"""Exam table construction and linkage of radiology reports to ED stays."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cta_chest", r"\bCTA\b.*CHEST|CT ANGIO.*CHEST|PULMONARY ANGIO|\bCTPA\b"),
    ("ct_head", r"\bCT\b.*(HEAD|BRAIN)"),
    ("ct_abd_pelvis", r"\bCT\b.*(ABD|ABDOMEN|PELVIS)"),
    ("ct_other", r"\bCT\b|\bCTA\b"),
    ("mri", r"\bMRI?\b|\bMRA\b|MAGNETIC"),
    ("ultrasound", r"\bUS\b|ULTRASOUND|\bDUPLEX\b|\bECHO\b"),
    ("cxr", r"\bCHEST\b.*(PORT|AP|PA|LAT|VIEW|X-?RAY)|^CHEST|CXR"),
    ("xr_extremity_spine", r"\b(ANKLE|KNEE|HIP|FOOT|HAND|WRIST|ELBOW|SHOULDER|FEMUR|TIBIA|FOREARM|HUMERUS|FINGER|TOE|SPINE|LUMBAR|CERVICAL|THORACIC|PELVIS|CLAVICLE|RIBS?)\b"),
)
_MOD_RES = [(m, re.compile(p, re.IGNORECASE)) for m, p in MODALITY_PATTERNS]

PRELIM_RE = re.compile(r"wet read|preliminary|prelim\b", re.IGNORECASE)
COMMUNICATION_RE = re.compile(r"discussed with|communicated to|notified|critical (?:result|finding)|telephone|paged", re.IGNORECASE)


def classify_modality(exam_name: object) -> str:
    """Map a radiology exam name to a coarse modality label ('other' when unmatched)."""
    s = str(exam_name)
    for m, rx in _MOD_RES:
        if rx.search(s):
            return m
    return "other"


def pivot_detail(radiology_detail: pd.DataFrame) -> pd.DataFrame:
    """Pivot radiology_detail (note_id, field_name, field_value, field_ordinal) to one row per note_id.

    Keeps the first ordinal of exam_name, exam_code, cpt_code and parent_note_id.
    """
    keep = radiology_detail[radiology_detail["field_name"].isin(["exam_name", "exam_code", "cpt_code", "parent_note_id", "addendum_note_id"])]
    keep = keep.sort_values("field_ordinal").drop_duplicates(["note_id", "field_name"])
    wide = keep.pivot(index="note_id", columns="field_name", values="field_value")
    for c in ("exam_name", "exam_code", "cpt_code", "parent_note_id"):
        if c not in wide.columns:
            wide[c] = np.nan
    return wide.reset_index()


def build_exam_table(radiology: pd.DataFrame, radiology_detail: pd.DataFrame, max_tat_days: float = 7.0) -> pd.DataFrame:
    """One row per radiology note with modality, TAT (h), quality flags and text markers.

    Reports (note_type RR) and addenda (AR) are both kept; addenda carry
    ``parent_note_id``.  ``tat_quarantine`` flags storetime < charttime, TAT >
    ``max_tat_days`` or missing storetime; ``auto_stored`` flags storetime == charttime.
    """
    r = radiology.copy()
    r["charttime"] = pd.to_datetime(r["charttime"])
    r["storetime"] = pd.to_datetime(r["storetime"])
    r = r.merge(pivot_detail(radiology_detail), on="note_id", how="left")
    r["modality"] = r["exam_name"].map(classify_modality)
    r["tat_h"] = (r["storetime"] - r["charttime"]).dt.total_seconds() / 3600.0
    r["auto_stored"] = r["tat_h"].eq(0.0)
    r["tat_quarantine"] = r["storetime"].isna() | (r["tat_h"] < 0) | (r["tat_h"] > 24.0 * max_tat_days)
    txt = r["text"].fillna("").astype(str)
    r["prelim_marker"] = txt.str.contains(PRELIM_RE).astype(int)
    r["communication_marker"] = txt.str.contains(COMMUNICATION_RE).astype(int)
    r["is_addendum"] = r["note_type"].astype(str).str.upper().eq("AR")
    return r


def link_reports_to_edstays(exams: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """Attach the ED stay during which each report's charttime falls (same subject, intime <= charttime <= outtime).

    Returns the exams table restricted to linked reports with columns stay_id, intime,
    outtime, disposition, post_imaging_dwell_h, report_after_exit, report_to_exit_h,
    exam_seq and n_exams_in_stay (reports only, addenda excluded from counts).
    """
    e = edstays[["subject_id", "stay_id", "intime", "outtime", "disposition"]].copy()
    e["intime"], e["outtime"] = pd.to_datetime(e["intime"]), pd.to_datetime(e["outtime"])
    m = exams.merge(e, on="subject_id", how="inner")
    inside = (m["charttime"] >= m["intime"]) & (m["charttime"] <= m["outtime"])
    m = m[inside].copy()
    # if a report falls inside two overlapping stays, keep the stay with the latest intime
    m = m.sort_values(["note_id", "intime"]).drop_duplicates("note_id", keep="last")
    m["post_imaging_dwell_h"] = (m["outtime"] - m["charttime"]).dt.total_seconds() / 3600.0
    m["report_after_exit"] = (m["storetime"] > m["outtime"]).astype(int)
    m["report_to_exit_h"] = (m["outtime"] - m["storetime"]).dt.total_seconds() / 3600.0
    reports = m[~m["is_addendum"]].sort_values(["stay_id", "charttime"])
    reports = reports.assign(exam_seq=reports.groupby("stay_id").cumcount() + 1)
    reports["n_exams_in_stay"] = reports.groupby("stay_id")["note_id"].transform("size")
    m = m.merge(reports[["note_id", "exam_seq", "n_exams_in_stay"]], on="note_id", how="left")
    return m.reset_index(drop=True)


def addendum_flags(linked: pd.DataFrame, exams: pd.DataFrame) -> pd.DataFrame:
    """Per report note_id: has_addendum and addendum_after_exit (addendum storetime > ED outtime)."""
    add = exams[exams["is_addendum"] & exams["parent_note_id"].notna()][["note_id", "parent_note_id", "storetime"]]
    add = add.rename(columns={"note_id": "addendum_id", "storetime": "addendum_storetime"})
    rep = linked[~linked["is_addendum"]][["note_id", "outtime"]]
    j = rep.merge(add, left_on="note_id", right_on="parent_note_id", how="left")
    j["addendum_after_exit"] = (j["addendum_storetime"] > j["outtime"]).astype(int)
    g = j.groupby("note_id").agg(has_addendum=("addendum_id", lambda s: int(s.notna().any())), addendum_after_exit=("addendum_after_exit", "max"))
    return g.reset_index()


def index_exam_per_stay(linked: pd.DataFrame, priority: tuple[str, ...] = ("ct_abd_pelvis", "cta_chest", "ct_head", "ct_other", "cxr", "mri", "ultrasound", "xr_extremity_spine", "other")) -> pd.DataFrame:
    """Pick one index report per ED stay: highest-priority modality, then earliest charttime."""
    rep = linked[~linked["is_addendum"]].copy()
    rank = {m: i for i, m in enumerate(priority)}
    rep["_rank"] = rep["modality"].map(rank).fillna(len(priority))
    rep = rep.sort_values(["stay_id", "_rank", "charttime"]).drop_duplicates("stay_id", keep="first")
    return rep.drop(columns="_rank").reset_index(drop=True)
