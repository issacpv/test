"""Cohort construction for OASIS-3-style tables.

OASIS-3 identifiers carry the subject and the day from entry (``OAS30001_MR_d0129``); PET
Centiloids come from the PUP table (``OAS30001_PIB_PUPTIMECOURSE_d2430``); clinical rows are
``OAS30001_ClinicalData_d0000``. All matching is done on days.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_ID_RE = re.compile(r"^(OAS3\d{4})_(.+?)_d(\d{4,5})$")

#: tracer-specific Centiloid positivity thresholds used in the OASIS-3 documentation
CENTILOID_THRESHOLDS: Dict[str, float] = {"PIB": 16.4, "AV45": 20.6}


def parse_oasis_id(identifier: str) -> Tuple[str, str, int]:
    """Return ``(subject, kind, days_from_entry)`` for any OASIS-3 experiment/row ID."""
    m = _ID_RE.match(identifier.strip())
    if not m:
        raise ValueError(f"not an OASIS-3 id: {identifier!r}")
    return m.group(1), m.group(2), int(m.group(3))


def amyloid_positive(centiloid: np.ndarray, tracer: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Binary amyloid status from Centiloids with tracer-specific thresholds (NaN preserved)."""
    thr = {**CENTILOID_THRESHOLDS, **(thresholds or {})}
    cl = np.asarray(centiloid, float)
    tr = np.asarray(tracer).astype(str)
    cut = np.array([thr.get(t.upper(), np.nan) for t in tr])
    out = (cl >= cut).astype(float)
    out[~np.isfinite(cl) | ~np.isfinite(cut)] = np.nan
    return out


def match_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    window_days: int,
    subject_col: str = "subject",
    day_col: str = "day",
    suffix: str = "_r",
) -> pd.DataFrame:
    """For each row of ``left`` attach the row of ``right`` (same subject) closest in days.

    Rows further than ``window_days`` apart get NaN for the right-hand columns. Columns from
    ``right`` other than ``subject_col`` are suffixed with ``suffix``; ``gap_days{suffix}`` is added.
    """
    out_rows = []
    r_groups = {s: g.sort_values(day_col) for s, g in right.groupby(subject_col)}
    r_cols = [c for c in right.columns if c != subject_col]
    for _, row in left.iterrows():
        g = r_groups.get(row[subject_col])
        rec = {c + suffix: np.nan for c in r_cols}
        rec["gap_days" + suffix] = np.nan
        if g is not None and len(g):
            gaps = (g[day_col] - row[day_col]).abs()
            j = int(gaps.idxmin())
            if gaps.loc[j] <= window_days:
                for c in r_cols:
                    rec[c + suffix] = g.loc[j, c]
                rec["gap_days" + suffix] = int(g.loc[j, day_col] - row[day_col])
        out_rows.append(rec)
    return pd.concat([left.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def build_longitudinal_table(
    wmh_sessions: pd.DataFrame,
    pet: pd.DataFrame,
    clinical: pd.DataFrame,
    demographics: pd.DataFrame,
    pet_window_days: int = 365,
    clin_window_days: int = 180,
    carry_baseline_pet: bool = True,
) -> pd.DataFrame:
    """Assemble one row per WMH-scored MR session with amyloid status, clinical scores and time.

    Parameters
    ----------
    wmh_sessions
        ``subject, day, wmh_ml, icv_ml`` (+ any regional columns) per FLAIR session.
    pet
        ``subject, day, tracer, centiloid``.
    clinical
        ``subject, day, cdr, sumbox, mmse`` (+ psychometrics).
    demographics
        ``subject, age_at_entry, sex, education, apoe4``.
    """
    df = wmh_sessions.copy()
    df = match_nearest(df, pet[["subject", "day", "tracer", "centiloid"]], pet_window_days, suffix="_pet")
    if carry_baseline_pet:
        base = pet.sort_values("day").groupby("subject").first()
        missing = df["centiloid_pet"].isna()
        df.loc[missing, "centiloid_pet"] = df.loc[missing, "subject"].map(base["centiloid"])
        df.loc[missing, "tracer_pet"] = df.loc[missing, "subject"].map(base["tracer"])
        df.loc[missing, "gap_days_pet"] = df.loc[missing, "subject"].map(base["day"]) - df.loc[missing, "day"]
    df["amyloid_pos"] = amyloid_positive(df["centiloid_pet"], df["tracer_pet"].fillna("NA"))
    df = match_nearest(df, clinical, clin_window_days, suffix="_clin")
    df = df.merge(demographics, on="subject", how="left")
    df["age"] = df["age_at_entry"] + df["day"] / 365.25
    first = df.groupby("subject")["day"].transform("min")
    df["time"] = (df["day"] - first) / 365.25
    df["wmh_log"] = np.log1p(df["wmh_ml"])
    if "icv_ml" in df:
        df["wmh_pct_icv"] = 100 * df["wmh_ml"] / df["icv_ml"]
    return df


def progression_events(
    clinical: pd.DataFrame,
    baseline_cdr: float = 0.0,
    event_cdr: float = 0.5,
    subject_col: str = "subject",
    day_col: str = "day",
    cdr_col: str = "cdr",
) -> pd.DataFrame:
    """Time-to-event table for CDR progression among subjects with ``cdr == baseline_cdr`` at entry.

    Returns ``subject, day0, event, day_event_or_censor, followup_years``.
    """
    rows = []
    for s, g in clinical.sort_values(day_col).groupby(subject_col):
        g = g.dropna(subset=[cdr_col])
        if len(g) == 0 or g[cdr_col].iloc[0] != baseline_cdr:
            continue
        day0 = int(g[day_col].iloc[0])
        prog = g[g[cdr_col] >= event_cdr]
        if len(prog):
            day_ev, event = int(prog[day_col].iloc[0]), 1
        else:
            day_ev, event = int(g[day_col].iloc[-1]), 0
        rows.append({"subject": s, "day0": day0, "event": event, "day_end": day_ev, "followup_years": (day_ev - day0) / 365.25})
    return pd.DataFrame(rows, columns=["subject", "day0", "event", "day_end", "followup_years"])
