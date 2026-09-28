"""Timing features and chest-radiograph severity for linked reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHEXPERT_FINDINGS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
)
DAY_SHIFT_START = 7
NIGHT_START = 23


def timing_features(df: pd.DataFrame, time_col: str = "charttime") -> pd.DataFrame:
    """Add hour, hour_block (6 x 4 h), night, weekend, shift-distance and anchor-safe features.

    MIMIC date shifting preserves time of day and day of week, so these are valid;
    calendar date is not preserved, so no holiday or workload-across-patients features.
    """
    out = df.copy()
    t = pd.to_datetime(out[time_col])
    out["hour"] = t.dt.hour
    out["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    out["hour_block"] = (out["hour"] // 4).astype(int)
    out["night"] = ((out["hour"] >= NIGHT_START) | (out["hour"] < DAY_SHIFT_START)).astype(int)
    out["weekend"] = (t.dt.dayofweek >= 5).astype(int)
    # signed hours from the day-shift start, for regression discontinuity in time (range [-12, 12))
    d = (out["minute_of_day"] / 60.0) - DAY_SHIFT_START
    out["hours_from_shift_start"] = ((d + 12) % 24) - 12
    return out


def cxr_severity(linked: pd.DataFrame, cxr_metadata: pd.DataFrame, chexpert: pd.DataFrame, tol_min: float = 30.0) -> pd.DataFrame:
    """Number of CheXpert-positive findings for the MIMIC-CXR study acquired within +-tol_min of each CXR report's charttime.

    ``cxr_metadata``: subject_id, study_id, StudyDate (YYYYMMDD), StudyTime (HHMMSS.ffff).
    ``chexpert``: subject_id, study_id and the 14 label columns (1.0 positive, 0.0 negative, -1.0 uncertain, NaN).
    Returns note_id, study_id, n_positive_findings, study_to_chart_min (audit of charttime semantics).
    """
    md = cxr_metadata[["subject_id", "study_id", "StudyDate", "StudyTime"]].drop_duplicates("study_id").copy()
    date = md["StudyDate"].astype(int).astype(str)
    secs = md["StudyTime"].astype(float).fillna(0.0)
    hh = (secs // 10000).astype(int)
    mm = ((secs % 10000) // 100).astype(int)
    ss = (secs % 100).astype(int)
    md["study_time"] = pd.to_datetime(date, format="%Y%m%d") + pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s")
    labels = chexpert.set_index("study_id")[[c for c in CHEXPERT_FINDINGS if c in chexpert.columns]]
    md["n_positive_findings"] = md["study_id"].map((labels == 1.0).sum(axis=1))
    cx = linked[linked["modality"] == "cxr"][["note_id", "subject_id", "charttime"]].copy()
    cx["charttime"] = pd.to_datetime(cx["charttime"])
    j = cx.merge(md, on="subject_id", how="inner")
    j["study_to_chart_min"] = (j["charttime"] - j["study_time"]).dt.total_seconds() / 60.0
    j = j[j["study_to_chart_min"].abs() <= tol_min]
    j = j.reindex(j["study_to_chart_min"].abs().sort_values().index).drop_duplicates("note_id")
    return j[["note_id", "study_id", "n_positive_findings", "study_to_chart_min"]].reset_index(drop=True)


def stay_level_table(index_exams: pd.DataFrame, triage: pd.DataFrame, patients: pd.DataFrame, admissions: pd.DataFrame, icustays: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """Join the index exam per stay with triage acuity, age, admission outcome, ICU within 24 h and 72-h return."""
    d = index_exams.merge(triage[["stay_id", "acuity", "chiefcomplaint"]], on="stay_id", how="left")
    d = d.merge(patients[["subject_id", "anchor_age", "anchor_year", "anchor_year_group", "gender"]], on="subject_id", how="left")
    d["age"] = d["anchor_age"] + (pd.to_datetime(d["intime"]).dt.year - d["anchor_year"])
    d["admitted"] = d["disposition"].astype(str).str.upper().eq("ADMITTED").astype(int)
    adm = admissions[["hadm_id", "hospital_expire_flag"]].drop_duplicates("hadm_id")
    if "hadm_id" in d.columns:
        d = d.merge(adm, on="hadm_id", how="left")
    icu = icustays[["hadm_id", "intime"]].rename(columns={"intime": "icu_intime"})
    icu["icu_intime"] = pd.to_datetime(icu["icu_intime"])
    first = icu.groupby("hadm_id")["icu_intime"].min().reset_index()
    if "hadm_id" in d.columns:
        d = d.merge(first, on="hadm_id", how="left")
        h = (d["icu_intime"] - pd.to_datetime(d["outtime"])).dt.total_seconds() / 3600.0
        d["icu_24h"] = ((h >= -1) & (h <= 24)).fillna(False).astype(int)
    e = edstays[["stay_id", "subject_id", "intime", "outtime"]].copy()
    e["intime"], e["outtime"] = pd.to_datetime(e["intime"]), pd.to_datetime(e["outtime"])
    e = e.sort_values(["subject_id", "intime"])
    e["next_intime"] = e.groupby("subject_id")["intime"].shift(-1)
    gap = (e["next_intime"] - e["outtime"]).dt.total_seconds() / 3600.0
    e["ed_return_72h"] = ((gap >= 0) & (gap <= 72)).astype(int)
    d = d.merge(e[["stay_id", "ed_return_72h"]], on="stay_id", how="left")
    d["log_dwell"] = np.log(d["post_imaging_dwell_h"].clip(lower=1 / 60))
    d["arrival_to_exam_h"] = (pd.to_datetime(d["charttime"]) - pd.to_datetime(d["intime"])).dt.total_seconds() / 3600.0
    return timing_features(d)
