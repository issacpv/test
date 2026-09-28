"""Build subject timelines from OASIS-3-style session tables.

Every OASIS-3 id ends in ``_dNNNN`` = days since the subject's study entry, which makes session
alignment trivial. The functions here take standardized long tables (one row per record with
``subject`` and ``day``) and return one row per MR session with clinical covariates, amyloid status
over time, and group labels, plus helpers for consecutive-session pairs.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd

OASIS_ID_RE = re.compile(r"^(?P<subject>OAS\d{5})_(?P<kind>[^_]+)(?:_.*)?_d(?P<day>\d+)$")
DEFAULT_CENTILOID_THRESHOLDS: Mapping[str, float] = {"PIB": 16.4, "AV45": 20.6}


def parse_oasis_id(oasis_id: str) -> tuple[str, str, int]:
    """``'OAS30001_MR_d0129' -> ('OAS30001', 'MR', 129)``."""
    m = OASIS_ID_RE.match(str(oasis_id).strip())
    if not m:
        raise ValueError(f"not an OASIS id: {oasis_id!r}")
    return m.group("subject"), m.group("kind"), int(m.group("day"))


def ids_to_subject_day(ids: pd.Series) -> pd.DataFrame:
    parsed = ids.map(parse_oasis_id)
    return pd.DataFrame({"subject": [p[0] for p in parsed], "day": [p[2] for p in parsed]}, index=ids.index)


def nearest_within(left: pd.DataFrame, right: pd.DataFrame, max_gap: int, left_day: str = "day",
                   right_day: str = "day", suffix: str = "_r") -> pd.DataFrame:
    """Attach, per left row, the same-subject right row with the nearest day (≤ max_gap) or NaN."""
    l = left.reset_index(drop=True).copy()
    l["_li"] = np.arange(len(l))
    r = right.rename(columns={c: c + suffix for c in right.columns if c != "subject"})
    m = l.merge(r, on="subject", how="inner")
    m["_gap"] = (m[left_day] - m[right_day + suffix]).abs()
    m = m[m["_gap"] <= max_gap].sort_values(["_li", "_gap", right_day + suffix]).drop_duplicates("_li")
    keep = [c for c in m.columns if c not in l.columns or c == "_li"]
    return l.merge(m[keep], on="_li", how="left").drop(columns=["_li", "_gap"], errors="ignore")


def amyloid_status_over_time(sessions: pd.DataFrame, pet: pd.DataFrame,
                             thresholds: Mapping[str, float] = DEFAULT_CENTILOID_THRESHOLDS,
                             window_days: int = 730) -> pd.Series:
    """Amyloid status (1/0/NaN) for each session by the "once positive, always positive" rule.

    A session is A+ if any PET at or before it (or within ``window_days`` after it) is positive.
    It is A- if it has at least one negative PET within ``window_days`` and no positive PET at or
    before it. Otherwise NaN.

    Parameters
    ----------
    sessions : DataFrame with ``subject`` and ``day``.
    pet : DataFrame with ``subject``, ``day``, ``tracer``, ``centiloid``.
    """
    thr = pet["tracer"].astype(str).str.upper().map(dict(thresholds))
    pet = pet.assign(pos=(pd.to_numeric(pet["centiloid"], errors="coerce") >= thr).astype(float))
    pet.loc[thr.isna() | pet["centiloid"].isna(), "pos"] = np.nan
    pet = pet.dropna(subset=["pos"])
    out = np.full(len(sessions), np.nan)
    by_subject = {s: g for s, g in pet.groupby("subject")}
    for i, (subj, day) in enumerate(zip(sessions["subject"].to_numpy(), sessions["day"].to_numpy())):
        g = by_subject.get(subj)
        if g is None:
            continue
        first_pos = g.loc[g["pos"] == 1, "day"].min()
        if pd.notna(first_pos) and first_pos <= day + window_days:
            out[i] = 1.0
        elif ((g["pos"] == 0) & ((g["day"] - day).abs() <= window_days)).any():
            out[i] = 0.0
    return pd.Series(out, index=sessions.index, name="amyloid_pos")


def build_timeline(mr: pd.DataFrame, clinical: pd.DataFrame, pet: pd.DataFrame,
                   features: Optional[pd.DataFrame] = None, wmh: Optional[pd.DataFrame] = None,
                   max_clinical_gap: int = 180, pet_window_days: int = 730,
                   thresholds: Mapping[str, float] = DEFAULT_CENTILOID_THRESHOLDS) -> pd.DataFrame:
    """One row per MR session with covariates, amyloid status, group label and features.

    Parameters
    ----------
    mr : columns ``mr_id`` (OASIS id), optional ``scanner``.
    clinical : columns ``subject, day, cdr, cdr_sb, mmse, age_at_entry, sex_male`` (missing ones tolerated).
    pet : columns ``subject, day, tracer, centiloid``.
    features : optional, indexed by ``mr_id`` (or with an ``mr_id`` column), e.g. FreeSurfer regions + eTIV.
    wmh : optional, columns ``mr_id, wmh_mm3``.

    Returns
    -------
    DataFrame sorted by subject and day with ``years`` (from first session), ``visit`` (0, 1, ...),
    ``n_sessions``, ``amyloid_pos``, ``cog_impaired`` (CDR ≥ 0.5), ``group`` in
    {A-CN, A+CN, A+CI, A-CI, NaN}.
    """
    t = mr.copy()
    sd = ids_to_subject_day(t["mr_id"])
    t["subject"], t["day"] = sd["subject"], sd["day"]
    t = nearest_within(t, clinical, max_gap=max_clinical_gap, suffix="_c")
    ren = {c + "_c": c for c in ("cdr", "cdr_sb", "mmse", "age_at_entry", "sex_male")}
    t = t.rename(columns={k: v for k, v in ren.items() if k in t.columns})
    static = clinical.groupby("subject").agg({c: "first" for c in ("age_at_entry", "sex_male") if c in clinical})
    for c in static.columns:
        t[c] = t[c].fillna(t["subject"].map(static[c])) if c in t else t["subject"].map(static[c])
    if "age_at_entry" in t:
        t["age"] = t["age_at_entry"] + t["day"] / 365.25
    t["amyloid_pos"] = amyloid_status_over_time(t[["subject", "day"]], pet, thresholds, pet_window_days).to_numpy()
    t["cog_impaired"] = np.where(t["cdr"].isna(), np.nan, (t["cdr"] >= 0.5).astype(float)) if "cdr" in t else np.nan
    a = t["amyloid_pos"].map({1.0: "A+", 0.0: "A-"})
    c = t["cog_impaired"].map({1.0: "CI", 0.0: "CN"})
    t["group"] = (a + c).where(a.notna() & c.notna())
    if features is not None:
        f = features.reset_index() if "mr_id" not in features.columns else features
        t = t.merge(f, on="mr_id", how="left")
    if wmh is not None:
        t = t.merge(wmh[["mr_id", "wmh_mm3"]], on="mr_id", how="left")
    t = t.sort_values(["subject", "day"]).reset_index(drop=True)
    t["visit"] = t.groupby("subject").cumcount()
    t["n_sessions"] = t.groupby("subject")["day"].transform("size")
    t["years"] = (t["day"] - t.groupby("subject")["day"].transform("min")) / 365.25
    return t


def baseline_sessions(timeline: pd.DataFrame) -> pd.DataFrame:
    """First session per subject (the only rows models may be fitted on)."""
    return timeline[timeline["visit"] == 0].copy()


def consecutive_pairs(timeline: pd.DataFrame, min_interval_years: float = 0.5,
                      same_scanner_only: bool = False) -> pd.DataFrame:
    """Consecutive-session pairs per subject with the interval in years.

    Columns: subject, mr_id_0, mr_id_1, day_0, day_1, interval_years, plus ``group_0``/``group_1``.
    """
    t = timeline.sort_values(["subject", "day"])
    nxt = t.groupby("subject").shift(-1)
    pairs = pd.DataFrame({
        "subject": t["subject"], "mr_id_0": t["mr_id"], "mr_id_1": nxt["mr_id"],
        "day_0": t["day"], "day_1": nxt["day"],
        "group_0": t.get("group"), "group_1": nxt.get("group"),
    }).dropna(subset=["mr_id_1"])
    pairs["interval_years"] = (pairs["day_1"] - pairs["day_0"]) / 365.25
    if same_scanner_only and "scanner" in t:
        pairs = pairs[(t.loc[pairs.index, "scanner"] == nxt.loc[pairs.index, "scanner"])]
    return pairs[pairs["interval_years"] >= min_interval_years].reset_index(drop=True)


def group_counts(timeline: pd.DataFrame) -> pd.DataFrame:
    """Sessions / subjects / subjects with ≥ 2 sessions per group (baseline group)."""
    b = baseline_sessions(timeline)
    return pd.DataFrame({
        "sessions": timeline.groupby("group").size(),
        "subjects": b.groupby("group").size(),
        "subjects_2plus": b[b["n_sessions"] >= 2].groupby("group").size(),
    }).fillna(0).astype(int)
