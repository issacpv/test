"""Synthetic tests for rad_delay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rad_delay import analysis, features, linkage


def test_modality_classification():
    assert linkage.classify_modality("CHEST (PORTABLE AP)") == "cxr"
    assert linkage.classify_modality("CT HEAD W/O CONTRAST") == "ct_head"
    assert linkage.classify_modality("CT ABD & PELVIS WITH CONTRAST") == "ct_abd_pelvis"
    assert linkage.classify_modality("CTA CHEST W&W/O C&RECONS, NON-CORONARY") == "cta_chest"
    assert linkage.classify_modality("MR HEAD W & W/O CONTRAST") == "mri"
    assert linkage.classify_modality("US ABD LIMIT, SINGLE ORGAN") == "ultrasound"
    assert linkage.classify_modality("ANKLE (AP, MORTISE & LAT) RIGHT") == "xr_extremity_spine"
    assert linkage.classify_modality("DG FLUORO") == "other"


def _synth():
    t0 = pd.Timestamp("2150-03-01 10:00")
    edstays = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "stay_id": [10, 11, 20],
            "intime": [t0, t0 + pd.Timedelta(days=10), t0],
            "outtime": [t0 + pd.Timedelta(hours=6), t0 + pd.Timedelta(days=10, hours=3), t0 + pd.Timedelta(hours=4)],
            "disposition": ["HOME", "ADMITTED", "HOME"],
            "hadm_id": [np.nan, 100, np.nan],
        }
    )
    radiology = pd.DataFrame(
        {
            "note_id": ["a", "b", "c", "d", "e"],
            "subject_id": [1, 1, 1, 2, 2],
            "hadm_id": [np.nan] * 5,
            "note_type": ["RR", "RR", "AR", "RR", "RR"],
            "note_seq": [1, 2, 2, 1, 2],
            "charttime": [t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(days=3)],
            "storetime": [t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=8), t0 + pd.Timedelta(hours=30), t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(days=3, hours=1)],
            "text": ["Wet read discussed with ED.", "FINDINGS: ...", "ADDENDUM", "FINDINGS", "FINDINGS"],
        }
    )
    detail = pd.DataFrame(
        {
            "note_id": ["a", "b", "c", "d", "e"],
            "subject_id": [1, 1, 1, 2, 2],
            "field_name": ["exam_name", "exam_name", "parent_note_id", "exam_name", "exam_name"],
            "field_value": ["CHEST (PORTABLE AP)", "CT ABD & PELVIS WITH CONTRAST", "b", "CT HEAD W/O CONTRAST", "CHEST (PA & LAT)"],
            "field_ordinal": [1, 1, 1, 1, 1],
        }
    )
    return edstays, radiology, detail


def test_exam_table_linkage_and_flags():
    edstays, radiology, detail = _synth()
    ex = linkage.build_exam_table(radiology, detail)
    assert ex.set_index("note_id").loc["a", "tat_h"] == 1.0
    assert ex.set_index("note_id").loc["a", "prelim_marker"] == 1 and ex.set_index("note_id").loc["a", "communication_marker"] == 1
    assert ex.set_index("note_id").loc["d", "auto_stored"]
    assert ex.set_index("note_id").loc["c", "is_addendum"] and ex.set_index("note_id").loc["c", "parent_note_id"] == "b"
    linked = linkage.link_reports_to_edstays(ex, edstays)
    assert set(linked["note_id"]) == {"a", "b", "c", "d"}  # e is outside any stay
    li = linked.set_index("note_id")
    assert li.loc["a", "report_after_exit"] == 0 and li.loc["b", "report_after_exit"] == 1
    assert li.loc["b", "post_imaging_dwell_h"] == 4.0 and li.loc["b", "n_exams_in_stay"] == 2 and li.loc["b", "exam_seq"] == 2
    af = linkage.addendum_flags(linked, ex).set_index("note_id")
    assert af.loc["b", "has_addendum"] == 1 and af.loc["b", "addendum_after_exit"] == 1 and af.loc["a", "has_addendum"] == 0
    idx = linkage.index_exam_per_stay(linked)
    assert idx.set_index("stay_id").loc[10, "modality"] == "ct_abd_pelvis"
    dbr = analysis.decision_before_report(linked, by=("modality",))
    assert dbr.set_index("modality").loc["ct_abd_pelvis", "share"] == 1.0
    tf = features.timing_features(linked)
    assert set(tf["hour"]) <= set(range(24)) and tf["hours_from_shift_start"].between(-12, 12).all()
    desc = analysis.describe_tat(ex)
    assert "tat_q50" in desc.columns


def test_cxr_severity_join():
    linked = pd.DataFrame({"note_id": ["a"], "subject_id": [1], "modality": ["cxr"], "charttime": [pd.Timestamp("2150-03-01 11:00")]})
    md = pd.DataFrame({"subject_id": [1, 1], "study_id": [5, 6], "StudyDate": [21500301, 21500301], "StudyTime": [105500.0, 150000.0]})
    chex = pd.DataFrame({"subject_id": [1, 1], "study_id": [5, 6], "Pleural Effusion": [1.0, 0.0], "Edema": [1.0, 1.0], "Pneumonia": [-1.0, np.nan]})
    sev = features.cxr_severity(linked, md, chex)
    assert len(sev) == 1 and sev.loc[0, "study_id"] == 5 and sev.loc[0, "n_positive_findings"] == 2
    assert abs(sev.loc[0, "study_to_chart_min"] - 5.0) < 1e-6


def test_iv_recovers_effect_under_confounding_and_rd():
    rng = np.random.default_rng(0)
    n = 6000
    severity = rng.normal(0, 1, n)  # unobserved: sicker -> faster read AND longer dwell
    night = rng.binomial(1, 0.3, n)
    weekend = rng.binomial(1, 0.28, n)
    tat = 1.5 + 0.8 * night + 0.5 * weekend - 0.6 * severity + rng.normal(0, 0.7, n)
    dwell = 2.0 + 0.5 * tat + 1.0 * severity + rng.normal(0, 0.8, n)
    df = pd.DataFrame({"subject_id": np.arange(n), "tat_h": tat, "dwell": dwell, "night": night, "weekend": weekend, "age": rng.normal(50, 15, n)})
    res = analysis.two_stage_least_squares(df, "dwell", "tat_h", ["night", "weekend"], ["age"])
    assert res["first_stage_F"] > 50
    assert res["ols_coef"] < 0.3  # biased downwards by prioritisation
    assert abs(res["iv_coef"] - 0.5) < 3 * res["iv_se"] and abs(res["iv_coef"] - 0.5) < 0.15
    assert "sargan_p" in res
    # regression discontinuity in time: jump of +1 at the cutoff
    r = rng.uniform(-6, 6, n)
    y = 3 + 0.1 * r + 1.0 * (r >= 0) + rng.normal(0, 0.5, n)
    rd = analysis.rd_in_time(pd.DataFrame({"hours_from_shift_start": r, "y": y}), "y", bandwidth=2.0)
    assert abs(rd["rd_estimate"] - 1.0) < 0.25
    nc = analysis.negative_control_check(df.assign(arrival_to_exam_h=rng.normal(1, 0.3, n)), "tat_h", ["arrival_to_exam_h"], ["age"])
    assert abs(nc.loc[0, "coef"]) < 0.05
    pn = analysis.permutation_null(df.assign(hour_block=rng.integers(0, 6, n)), "dwell", "tat_h", ["age"], ["night", "hour_block"], n_perm=20)
    assert "p_value" in pn
