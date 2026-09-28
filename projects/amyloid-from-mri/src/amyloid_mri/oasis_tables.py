"""Loaders and joiners for OASIS-3 clinical, PET (PUP/Centiloid) and MR-session tables.

OASIS-3 identifiers encode the subject and the number of days since study entry, e.g.
``OAS30001_MR_d0129`` (MR session, day 129), ``OAS30001_AV45_PUPTIMECOURSE_d2430`` (PUP output
for an AV45 PET on day 2430), ``OAS30001_ClinicalData_d0000`` (clinical visit). Everything here
works on that convention, so the same functions serve OASIS-4 ids (``OAS4xxxx``).

Column names differ slightly between OASIS-3 releases; loaders therefore accept a list of
candidate names and match case-insensitively.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

OASIS_ID_RE = re.compile(r"^(?P<subject>OAS\d{5})_(?P<kind>[^_]+)(?:_.*)?_d(?P<day>\d+)$")

#: Tracer-specific Centiloid positivity thresholds used by the OASIS-3 documentation
#: (PiB 16.4 CL ~ SUVR 1.42 with RSF PVC; AV45 20.6 CL). Override per analysis.
DEFAULT_CENTILOID_THRESHOLDS: Mapping[str, float] = {"PIB": 16.4, "AV45": 20.6}

PathOrFrame = Union[str, Path, pd.DataFrame]


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def parse_oasis_id(oasis_id: str) -> tuple[str, str, int]:
    """Split an OASIS id into (subject, kind, day).

    >>> parse_oasis_id("OAS30001_AV45_PUPTIMECOURSE_d2430")
    ('OAS30001', 'AV45', 2430)
    """
    m = OASIS_ID_RE.match(str(oasis_id).strip())
    if not m:
        raise ValueError(f"not an OASIS id: {oasis_id!r}")
    return m.group("subject"), m.group("kind"), int(m.group("day"))


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Return the first column matching a candidate (exact, then case-insensitive, then substring)."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    if required:
        raise KeyError(f"none of {list(candidates)} found in columns {cols}")
    return None


def _as_frame(x: PathOrFrame) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.read_csv(x)


def apoe_e4_count(genotype: object) -> float:
    """Number of APOE ε4 alleles from a genotype coded like ``34``, ``'3/4'`` or ``'E3E4'``.

    Returns NaN when the genotype is missing or unparsable.
    """
    if genotype is None or (isinstance(genotype, float) and np.isnan(genotype)):
        return np.nan
    digits = re.findall(r"[234]", str(genotype))
    if len(digits) != 2:
        return np.nan
    return float(sum(d == "4" for d in digits))


def sex_to_male(value: object) -> float:
    """Map 'M'/'F' (or 'male'/'female', 1/2 codings are refused) to 1.0/0.0, NaN otherwise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    v = str(value).strip().lower()
    if v in ("m", "male"):
        return 1.0
    if v in ("f", "female"):
        return 0.0
    return np.nan


# ---------------------------------------------------------------------------
# loaders -> standardized long tables (one row per record; columns subject, day, ...)
# ---------------------------------------------------------------------------
def load_clinical(source: PathOrFrame) -> pd.DataFrame:
    """Standardize the ADRC clinical-data table.

    Output columns: subject, day, age_at_entry, sex_male, apoe_genotype, apoe_e4_count, mmse, cdr,
    plus any others left untouched.
    """
    df = _as_frame(source)
    id_col = find_column(df, ["ADRC_ADRCCLINICALDATA ID", "ADRC_ADRCCLINICALDATA_ID", "clinical_id", "ID"],
                         required=False)
    if id_col is not None:
        parsed = df[id_col].map(parse_oasis_id)
        df["subject"] = [p[0] for p in parsed]
        df["day"] = [p[2] for p in parsed]
    else:
        df["subject"] = df[find_column(df, ["Subject", "subject_id"])]
        df["day"] = pd.to_numeric(df[find_column(df, ["day", "days_to_visit"])], errors="coerce")
    out = pd.DataFrame({
        "subject": df["subject"].astype(str),
        "day": pd.to_numeric(df["day"], errors="coerce").astype(int),
    })
    out["age_at_entry"] = pd.to_numeric(df[find_column(df, ["ageAtEntry", "age_at_entry"])], errors="coerce")
    sex_col = find_column(df, ["M/F", "sex", "gender"], required=False)
    out["sex_male"] = df[sex_col].map(sex_to_male) if sex_col else np.nan
    apoe_col = find_column(df, ["apoe", "APOE"], required=False)
    out["apoe_genotype"] = df[apoe_col] if apoe_col else np.nan
    out["apoe_e4_count"] = out["apoe_genotype"].map(apoe_e4_count)
    mmse_col = find_column(df, ["mmse", "MMSE"], required=False)
    out["mmse"] = pd.to_numeric(df[mmse_col], errors="coerce") if mmse_col else np.nan
    cdr_col = find_column(df, ["cdr", "CDRTOT", "cdr_global"], required=False)
    out["cdr"] = pd.to_numeric(df[cdr_col], errors="coerce") if cdr_col else np.nan
    return out


def load_pup(source: PathOrFrame) -> pd.DataFrame:
    """Standardize the PUP time-course table (one row per amyloid PET session).

    Output columns: subject, day, pup_id, tracer (upper-case, e.g. PIB/AV45), centiloid.
    """
    df = _as_frame(source)
    id_col = find_column(df, ["PUP_PUPTIMECOURSEDATA ID", "PUP_PUPTIMECOURSEDATA_ID", "pup_id", "PUP ID"])
    parsed = df[id_col].map(parse_oasis_id)
    out = pd.DataFrame({
        "subject": [p[0] for p in parsed],
        "day": [p[2] for p in parsed],
        "pup_id": df[id_col].astype(str).values,
    })
    tracer_col = find_column(df, ["tracer", "Tracer"], required=False)
    tracer = df[tracer_col].astype(str) if tracer_col else pd.Series([p[1] for p in parsed])
    out["tracer"] = tracer.str.upper().str.replace("FBP", "AV45", regex=False).values
    cl_col = find_column(df, ["Centiloid_fSUVR_TOT_CORTMEAN", "centiloid", "Centiloid"])
    out["centiloid"] = pd.to_numeric(df[cl_col], errors="coerce").values
    return out


def load_mr_sessions(source: PathOrFrame) -> pd.DataFrame:
    """Standardize the MR-session table. Output columns: subject, day, mr_id, scanner."""
    df = _as_frame(source)
    id_col = find_column(df, ["MR ID", "MR_ID", "experiment_id", "mr_id", "Label"])
    parsed = df[id_col].map(parse_oasis_id)
    out = pd.DataFrame({
        "subject": [p[0] for p in parsed],
        "day": [p[2] for p in parsed],
        "mr_id": df[id_col].astype(str).values,
    })
    sc = find_column(df, ["Scanner", "scanner"], required=False)
    out["scanner"] = df[sc].astype(str).values if sc else "unknown"
    return out


# ---------------------------------------------------------------------------
# joining
# ---------------------------------------------------------------------------
def match_nearest(left: pd.DataFrame, right: pd.DataFrame, max_gap: int,
                  left_day: str = "day", right_day: str = "day", suffix: str = "_r") -> pd.DataFrame:
    """For each left row, attach the right row of the same subject with the nearest day.

    Rows without a right record within ``max_gap`` days keep NaN in the right-hand columns.
    Ties are broken towards the earlier right record. The returned frame preserves the left order.
    """
    l = left.reset_index(drop=True).copy()
    l["_li"] = np.arange(len(l))
    r = right.copy()
    r = r.rename(columns={c: (c + suffix if c != "subject" else c) for c in r.columns})
    m = l.merge(r, on="subject", how="inner")
    m["_gap"] = (m[left_day] - m[right_day + suffix]).abs()
    m = m[m["_gap"] <= max_gap].sort_values(["_li", "_gap", right_day + suffix]).drop_duplicates("_li")
    out = l.merge(m.drop(columns=[c for c in l.columns if c != "_li"]), on="_li", how="left")
    return out.drop(columns=["_li"])


def centiloid_positive(centiloid: pd.Series, tracer: pd.Series,
                       thresholds: Mapping[str, float] = DEFAULT_CENTILOID_THRESHOLDS) -> pd.Series:
    """Tracer-specific amyloid positivity (1.0/0.0, NaN when Centiloid or tracer is unknown)."""
    thr = tracer.astype(str).str.upper().map(dict(thresholds))
    cl = pd.to_numeric(centiloid, errors="coerce")
    out = (cl >= thr).astype(float)
    out[cl.isna() | thr.isna()] = np.nan
    return out


def build_subject_table(clinical: pd.DataFrame, pup: pd.DataFrame, mr: pd.DataFrame,
                        max_pet_gap: int = 365, max_clinical_gap: int = 180,
                        one_per_subject: bool = True,
                        thresholds: Mapping[str, float] = DEFAULT_CENTILOID_THRESHOLDS,
                        grey_zone: Optional[tuple[float, float]] = None) -> pd.DataFrame:
    """Build the analysis table: one row per MR session with matched PET label and clinical covariates.

    Parameters
    ----------
    clinical, pup, mr
        Outputs of :func:`load_clinical`, :func:`load_pup`, :func:`load_mr_sessions`.
    max_pet_gap, max_clinical_gap
        Maximum |days| between the MR session and the PET / clinical visit.
    one_per_subject
        Keep only the earliest MR session with a PET label per subject (primary analysis). With
        ``False`` all labelled sessions are kept and ``subject`` must be used as the CV group.
    grey_zone
        Optional (low, high) Centiloid interval; sessions inside it get label NaN (sensitivity analysis).

    Returns
    -------
    DataFrame with columns: subject, mr_id, mr_day, scanner, pet_day, pup_id, tracer, centiloid,
    amyloid_pos, pet_gap_days, clin_day, age, sex_male, apoe_e4_count, mmse, cdr, cog_status.
    """
    t = match_nearest(mr.rename(columns={"day": "mr_day"}), pup, max_gap=max_pet_gap,
                      left_day="mr_day", right_day="day", suffix="_pet")
    t = t.rename(columns={"day_pet": "pet_day", "pup_id_pet": "pup_id", "tracer_pet": "tracer",
                          "centiloid_pet": "centiloid"})
    t["pet_gap_days"] = (t["mr_day"] - t["pet_day"]).abs()
    t = match_nearest(t, clinical, max_gap=max_clinical_gap, left_day="mr_day", right_day="day",
                      suffix="_clin")
    t = t.rename(columns={"day_clin": "clin_day", "age_at_entry_clin": "age_at_entry",
                          "sex_male_clin": "sex_male", "apoe_e4_count_clin": "apoe_e4_count",
                          "mmse_clin": "mmse", "cdr_clin": "cdr", "apoe_genotype_clin": "apoe_genotype"})
    # demographics that do not change over visits: fill from any visit of the subject
    static = clinical.groupby("subject").agg(age_at_entry=("age_at_entry", "first"),
                                             sex_male=("sex_male", "first"),
                                             apoe_e4_count=("apoe_e4_count", "first"))
    for col in ("age_at_entry", "sex_male", "apoe_e4_count"):
        t[col] = t[col].fillna(t["subject"].map(static[col]))
    t["age"] = t["age_at_entry"] + t["mr_day"] / 365.25
    t["amyloid_pos"] = centiloid_positive(t["centiloid"], t["tracer"].fillna(""), thresholds)
    if grey_zone is not None:
        lo, hi = grey_zone
        t.loc[(t["centiloid"] > lo) & (t["centiloid"] < hi), "amyloid_pos"] = np.nan
    t["cog_status"] = np.where(t["cdr"].isna(), None, np.where(t["cdr"] == 0, "CN", "CI"))
    t = t[t["amyloid_pos"].notna()].copy()
    if one_per_subject:
        t = t.sort_values(["subject", "mr_day"]).drop_duplicates("subject", keep="first")
    cols = ["subject", "mr_id", "mr_day", "scanner", "pet_day", "pup_id", "tracer", "centiloid",
            "amyloid_pos", "pet_gap_days", "clin_day", "age", "sex_male", "apoe_e4_count", "mmse",
            "cdr", "cog_status"]
    return t[[c for c in cols if c in t.columns]].reset_index(drop=True)
