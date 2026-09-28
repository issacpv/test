"""Phenotype harmonizer: bring HCP-YA, HCP Lifespan (NDA), OASIS-3 and BIDS tables to one schema.

Common schema (one row per scan session):

    subject_id, session_id, dataset, site, scanner, age, sex, age_is_binned, group, diagnosis

``sex`` is 'F'/'M'; ``age`` in years (HCP-YA open data only have bins, so the midpoint is used and
``age_is_binned`` is True); ``group`` is 'control' for reference subjects and anything else for
scored-only subjects.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

SCHEMA: tuple[str, ...] = ("subject_id", "session_id", "dataset", "site", "scanner", "age", "sex",
                          "age_is_binned", "group", "diagnosis")

PathOrFrame = Union[str, Path, pd.DataFrame]

_AGE_BIN_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_AGE_PLUS_RE = re.compile(r"^\s*(\d+)\s*\+\s*$")


def _frame(x: PathOrFrame, **kw) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    p = Path(x)
    if p.suffix.lower() in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t", **kw)
    return pd.read_csv(p, **kw)


def normalize_sex(value: object) -> Optional[str]:
    """Map common codings ('M', 'male', 1?) to 'M'/'F'; numeric codes are refused (ambiguous)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = str(value).strip().lower()
    if v in ("m", "male", "man"):
        return "M"
    if v in ("f", "female", "woman"):
        return "F"
    return None


def parse_age_bin(value: object, open_bin_width: float = 2.0) -> tuple[float, bool]:
    """Age from a number or an HCP-YA style bin ('22-25' → 23.5, '36+' → 36 + width/2).

    Returns (age, is_binned).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, False
    s = str(value).strip()
    try:
        return float(s), False
    except ValueError:
        pass
    m = _AGE_BIN_RE.match(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2 + 0.5, True  # bins are inclusive integer years: 22-25 covers [22, 26)
    m = _AGE_PLUS_RE.match(s)
    if m:
        return float(m.group(1)) + open_bin_width / 2, True
    return np.nan, False


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for c in SCHEMA:
        if c not in df:
            df[c] = np.nan
    df["sex"] = df["sex"].map(normalize_sex)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_is_binned"] = df["age_is_binned"].fillna(False).astype(bool)
    df["site"] = df["site"].fillna(df["dataset"])
    df["session_id"] = df["session_id"].fillna("ses-01")
    df["group"] = df["group"].fillna("control")
    return df[list(SCHEMA)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
def from_hcp_ya(unrestricted: PathOrFrame, restricted: Optional[PathOrFrame] = None,
                site: str = "HCP-YA_WashU") -> pd.DataFrame:
    """HCP-YA unrestricted CSV (``Subject``, ``Gender``, ``Age`` bins); exact ``Age_in_Yrs`` from the
    restricted file when available (never commit that file)."""
    u = _frame(unrestricted)
    ages = u["Age"].map(parse_age_bin)
    df = pd.DataFrame({"subject_id": u["Subject"].astype(str), "dataset": "HCP-YA", "site": site,
                       "scanner": "Siemens Skyra 3T", "age": [a for a, _ in ages],
                       "age_is_binned": [b for _, b in ages], "sex": u["Gender"], "group": "control"})
    if restricted is not None:
        r = _frame(restricted)
        exact = r.set_index(r["Subject"].astype(str))["Age_in_Yrs"]
        df["age"] = df["subject_id"].map(exact).fillna(df["age"])
        df["age_is_binned"] = df["subject_id"].map(exact).isna() & df["age_is_binned"]
    return _finish(df)


def from_nda(ndar_subject: PathOrFrame, dataset: str, site: Optional[str] = None,
             scanner: str = "Siemens Prisma 3T") -> pd.DataFrame:
    """NDA ``ndar_subject01``-style table (HCP-A / HCP-D): ``src_subject_id``, ``interview_age`` (months),
    ``sex``; optional ``site`` column. NDA text exports carry a second header row that is skipped."""
    df = _frame(ndar_subject)
    if len(df) and str(df.iloc[0, 0]).lower().startswith(("the ", "subject", "gender")):
        df = df.iloc[1:].copy()
    sub = df["src_subject_id"].astype(str)
    out = pd.DataFrame({"subject_id": sub, "dataset": dataset,
                        "site": df["site"] if "site" in df else (site or dataset), "scanner": scanner,
                        "age": pd.to_numeric(df["interview_age"], errors="coerce") / 12.0,
                        "age_is_binned": False, "sex": df["sex"], "group": "control"})
    if "session_id" in df:
        out["session_id"] = df["session_id"]
    return _finish(out)


def from_oasis3(clinical: PathOrFrame, mr_sessions: PathOrFrame, control_rule: str = "cdr0",
                amyloid: Optional[pd.DataFrame] = None, max_clinical_gap: int = 365) -> pd.DataFrame:
    """OASIS-3: one row per MR session; age = ageAtEntry + days/365.25; group from the nearest CDR.

    ``clinical`` needs ``Subject``/``ADRC_ADRCCLINICALDATA ID``, ``ageAtEntry``, ``M/F``, ``cdr``;
    ``mr_sessions`` needs ``MR ID`` (``OAS30001_MR_d0129``) and optionally ``Scanner``.
    ``control_rule`` 'cdr0' or 'cdr0_amyloid_negative' (then ``amyloid`` = DataFrame with
    ``subject``, ``amyloid_pos``).
    """
    c = _frame(clinical)
    m = _frame(mr_sessions)
    id_col = "ADRC_ADRCCLINICALDATA ID" if "ADRC_ADRCCLINICALDATA ID" in c else next(
        (k for k in c.columns if "CLINICALDATA" in k.upper()), None)
    if id_col is not None:
        c["subject"] = c[id_col].str.split("_").str[0]
        c["day"] = c[id_col].str.extract(r"_d(\d+)$")[0].astype(int)
    else:
        c["subject"] = c["Subject"].astype(str)
        c["day"] = pd.to_numeric(c["day"], errors="coerce")
    mr_col = "MR ID" if "MR ID" in m else next(k for k in m.columns if "MR" in k.upper() and "ID" in k.upper())
    m["subject"] = m[mr_col].str.split("_").str[0]
    m["day"] = m[mr_col].str.extract(r"_d(\d+)$")[0].astype(int)
    static = c.groupby("subject").agg(age_at_entry=("ageAtEntry", "first"), sex=("M/F", "first"))
    # nearest clinical visit for CDR
    mm = m.merge(c[["subject", "day", "cdr"]].rename(columns={"day": "cday"}), on="subject", how="left")
    mm["gap"] = (mm["day"] - mm["cday"]).abs()
    mm = mm.sort_values(["subject", "day", "gap"]).drop_duplicates([mr_col])
    mm.loc[mm["gap"] > max_clinical_gap, "cdr"] = np.nan
    out = pd.DataFrame({"subject_id": mm["subject"], "session_id": mm[mr_col], "dataset": "OASIS-3",
                        "site": "OASIS-3_WashU", "scanner": mm["Scanner"] if "Scanner" in mm else np.nan,
                        "age": mm["subject"].map(static["age_at_entry"]) + mm["day"] / 365.25,
                        "age_is_binned": False, "sex": mm["subject"].map(static["sex"]),
                        "diagnosis": np.where(mm["cdr"].isna(), None, np.where(mm["cdr"] == 0, "CDR0", "CDR>=0.5"))})
    grp = np.where(mm["cdr"] == 0, "control", np.where(mm["cdr"].isna(), "unknown", "impaired"))
    if control_rule == "cdr0_amyloid_negative" and amyloid is not None:
        a = amyloid.set_index("subject")["amyloid_pos"]
        apos = mm["subject"].map(a)
        grp = np.where((grp == "control") & (apos == 1.0), "preclinical_amyloid_positive",
                       np.where((grp == "control") & apos.isna(), "control_amyloid_unknown", grp))
    out["group"] = grp
    return _finish(out)


def from_bids_participants(participants: PathOrFrame, dataset: str, site: Optional[str] = None,
                           scanner: Optional[str] = None, diagnosis_col: Optional[str] = None,
                           control_values: Sequence[str] = ("CONTROL", "control", "HC", "healthy")) -> pd.DataFrame:
    """BIDS ``participants.tsv`` (``participant_id``, ``age``, ``sex``; optional diagnosis column)."""
    p = _frame(participants)
    ages = p["age"].map(parse_age_bin) if "age" in p else [(np.nan, False)] * len(p)
    out = pd.DataFrame({"subject_id": p["participant_id"].astype(str), "dataset": dataset,
                        "site": site or dataset, "scanner": scanner,
                        "age": [a for a, _ in ages], "age_is_binned": [b for _, b in ages],
                        "sex": p["sex"] if "sex" in p else None})
    if diagnosis_col and diagnosis_col in p:
        out["diagnosis"] = p[diagnosis_col]
        out["group"] = np.where(p[diagnosis_col].isin(control_values), "control", "patient")
    if "family_id" in p:
        out["family_id"] = p["family_id"]
    fin = _finish(out)
    if "family_id" in out:
        fin["family_id"] = out["family_id"].to_numpy()
    return fin


def harmonize(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate adapter outputs, validate the schema and drop rows without age or sex."""
    parts = [f for f in frames]
    for f in parts:
        missing = [c for c in SCHEMA if c not in f.columns]
        if missing:
            raise ValueError(f"frame missing schema columns {missing}")
    df = pd.concat(parts, ignore_index=True)
    df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)
    return df


def site_summary(pheno: pd.DataFrame, decade: int = 10) -> pd.DataFrame:
    """Counts of reference subjects per site × sex × age decade (cells with n < 10 should be flagged)."""
    ctrl = pheno[pheno["group"] == "control"].copy()
    ctrl["age_bin"] = (ctrl["age"] // decade * decade).astype(int)
    return ctrl.pivot_table(index=["site", "sex"], columns="age_bin", values="subject_id", aggfunc="count",
                            fill_value=0)


# ---------------------------------------------------------------------------
# feature tables
# ---------------------------------------------------------------------------
def read_freesurfer_table(path: PathOrFrame, id_col: Optional[str] = None, measure: str = "thickness") -> pd.DataFrame:
    """Read an ``aparcstats2table``/``asegstats2table`` output (first column = subject id) and keep
    columns for ``measure`` ('thickness', 'volume', 'area'); the index is the subject/session id."""
    df = _frame(path)
    id_col = id_col or df.columns[0]
    df = df.set_index(df[id_col].astype(str)).drop(columns=[id_col])
    if measure == "volume":
        keep = [c for c in df.columns if not c.lower().endswith(("_thickness", "_area")) and "eTIV" not in c]
        keep += [c for c in df.columns if "eTIV" in c or "EstimatedTotalIntraCranialVol" in c]
    else:
        keep = [c for c in df.columns if c.lower().endswith(f"_{measure}")]
    return df[list(dict.fromkeys(keep))].apply(pd.to_numeric, errors="coerce")


def icv_normalize(volumes: pd.DataFrame, icv_col: Optional[str] = None, scale: float = 1e3) -> pd.DataFrame:
    """Divide every volume column by ICV (column containing 'eTIV' or 'IntraCranial') × scale."""
    icv_col = icv_col or next(c for c in volumes.columns if "etiv" in c.lower() or "intracranial" in c.lower())
    out = volumes.drop(columns=[icv_col]).div(volumes[icv_col], axis=0) * scale
    return out


def read_tract_table(path: PathOrFrame, metric: str, id_col: Optional[str] = None) -> pd.DataFrame:
    """Read a wide tract table (e.g. TractSeg ``Tractometry`` means or JHU-atlas means) and prefix
    columns with the metric name (``FA_CST_left``)."""
    df = _frame(path)
    id_col = id_col or df.columns[0]
    df = df.set_index(df[id_col].astype(str)).drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    df.columns = [c if c.upper().startswith(metric.upper() + "_") else f"{metric}_{c}" for c in df.columns]
    return df


def attach_features(pheno: pd.DataFrame, features: pd.DataFrame, on: str = "subject_id") -> pd.DataFrame:
    """Left-join a feature table (indexed by subject or session id) onto the phenotype table."""
    f = features.copy()
    f.index = f.index.astype(str)
    return pheno.merge(f, left_on=on, right_index=True, how="left")
