"""Cohort construction: pregnancy / HDP flags from ICD codes and matched controls.

Inputs are plain DataFrames in the MIMIC column conventions so the same code
serves MIMIC-III (``DIAGNOSES_ICD``: hadm_id, icd9_code) and MIMIC-IV
(``diagnoses_icd``: hadm_id, icd_code, icd_version).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ICD-9: 640-679 complications of pregnancy, childbirth and the puerperium;
# V22/V23 supervision of pregnancy; V27 outcome of delivery.
ICD9_PREGNANCY_PREFIXES: tuple[str, ...] = tuple(str(c) for c in range(640, 680)) + ("V22", "V23", "V27")
# ICD-9 642: hypertension complicating pregnancy. 642.0-642.2 pre-existing, 642.3 transient,
# 642.4 mild/unspecified preeclampsia, 642.5 severe preeclampsia (incl. HELLP), 642.6 eclampsia,
# 642.7 preeclampsia superimposed on chronic hypertension, 642.9 unspecified.
ICD9_HDP_PREFIX = "642"
ICD9_HDP_SEVERE_PREFIXES: tuple[str, ...] = ("6425", "6426", "6427")

# ICD-10-CM: O00-O9A pregnancy chapter; Z33/Z34 pregnancy state/supervision; Z37 outcome of delivery.
ICD10_PREGNANCY_PREFIXES: tuple[str, ...] = ("O", "Z33", "Z34", "Z37")
# O10-O16 hypertensive disorders; O11 pre-existing HTN with preeclampsia; O13 gestational HTN;
# O14 preeclampsia (O14.1 severe, O14.2 HELLP); O15 eclampsia; O16 unspecified.
ICD10_HDP_PREFIXES: tuple[str, ...] = ("O10", "O11", "O13", "O14", "O15", "O16")
ICD10_HDP_SEVERE_PREFIXES: tuple[str, ...] = ("O11", "O141", "O142", "O15")


def _norm(code: object) -> str:
    return str(code).replace(".", "").strip().upper()


def flag_pregnancy(diag: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hadm_id with boolean columns ``pregnant``, ``hdp``, ``hdp_severe``.

    ``diag`` needs columns ``hadm_id`` and either ``icd9_code`` (MIMIC-III) or
    ``icd_code`` + ``icd_version`` (MIMIC-IV).
    """
    d = diag.copy()
    if "icd9_code" in d.columns and "icd_code" not in d.columns:
        d["icd_code"] = d["icd9_code"]
        d["icd_version"] = 9
    d["code"] = d["icd_code"].map(_norm)
    v9 = d["icd_version"].astype(int) == 9
    is_preg = np.where(
        v9,
        d["code"].str.startswith(ICD9_PREGNANCY_PREFIXES),
        d["code"].str.startswith(ICD10_PREGNANCY_PREFIXES),
    )
    is_hdp = np.where(
        v9, d["code"].str.startswith(ICD9_HDP_PREFIX), d["code"].str.startswith(ICD10_HDP_PREFIXES)
    )
    is_sev = np.where(
        v9,
        d["code"].str.startswith(ICD9_HDP_SEVERE_PREFIXES),
        d["code"].str.startswith(ICD10_HDP_SEVERE_PREFIXES),
    )
    d["pregnant"], d["hdp"], d["hdp_severe"] = is_preg, is_hdp, is_sev
    out = d.groupby("hadm_id")[["pregnant", "hdp", "hdp_severe"]].any().reset_index()
    # HDP implies pregnancy-related admission even if no other O/64x code present
    out["pregnant"] = out["pregnant"] | out["hdp"]
    return out


def assign_group(flags: pd.DataFrame) -> pd.Series:
    """Map flags to analysis group labels: 'control', 'pregnant_normotensive', 'hdp'."""
    g = pd.Series("control", index=flags.index, dtype=object)
    g[flags["pregnant"] & ~flags["hdp"]] = "pregnant_normotensive"
    g[flags["hdp"]] = "hdp"
    return g


def select_matched_controls(
    admissions: pd.DataFrame,
    case_hadm: pd.Series,
    ratio: int = 3,
    age_band: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Pick non-pregnant female controls matched on age band and first care unit.

    ``admissions`` needs columns hadm_id, subject_id, gender ('F'/'M'), age (years),
    first_careunit, and a boolean ``ever_pregnant`` (any pregnancy code on any
    admission of the subject).  Returns a DataFrame with columns case_hadm_id,
    control_hadm_id, stratum.  Controls are sampled without replacement and a
    subject is used at most once.
    """
    rng = np.random.default_rng(seed)
    adm = admissions.copy()
    adm["age_band"] = (adm["age"] // age_band).astype(int)
    adm["stratum"] = adm["age_band"].astype(str) + "|" + adm["first_careunit"].astype(str)
    cases = adm[adm["hadm_id"].isin(set(case_hadm))]
    pool = adm[(adm["gender"] == "F") & (~adm["ever_pregnant"]) & (adm["age"].between(18, 45))]
    pool = pool[~pool["hadm_id"].isin(set(case_hadm))]
    used_subjects: set = set()
    rows = []
    for _, c in cases.iterrows():
        cand = pool[(pool["stratum"] == c["stratum"]) & (~pool["subject_id"].isin(used_subjects))]
        take = cand.sample(n=min(ratio, len(cand)), random_state=int(rng.integers(0, 2**31 - 1))) if len(cand) else cand
        for _, t in take.iterrows():
            used_subjects.add(t["subject_id"])
            rows.append({"case_hadm_id": c["hadm_id"], "control_hadm_id": t["hadm_id"], "stratum": c["stratum"]})
    return pd.DataFrame(rows, columns=["case_hadm_id", "control_hadm_id", "stratum"])


def tier_records(record_index: pd.DataFrame, min_minutes_tier_a: float = 10.0, min_nibp: int = 5) -> pd.DataFrame:
    """Assign waveform records to Tier A (ECG+PLETH+ABP) or Tier B (ECG+PLETH+NIBP numerics).

    ``record_index`` columns: record, subject_id, signals (list/tuple of channel
    names as in the .hea header), duration_min, n_nibp (count of NIBP numerics).
    Adds a ``tier`` column with 'A', 'B' or '' (ineligible).
    """
    ri = record_index.copy()

    def _tier(row: pd.Series) -> str:
        sig = {str(s).upper() for s in row["signals"]}
        has_ecg = any(s.startswith("II") or s in {"ECG", "I", "V", "AVR", "MCL1"} for s in sig)
        has_ppg = "PLETH" in sig
        has_abp = bool(sig & {"ABP", "ART"})
        if has_ecg and has_ppg and has_abp and row["duration_min"] >= min_minutes_tier_a:
            return "A"
        if has_ecg and has_ppg and row.get("n_nibp", 0) >= min_nibp:
            return "B"
        return ""

    ri["tier"] = ri.apply(_tier, axis=1)
    return ri
