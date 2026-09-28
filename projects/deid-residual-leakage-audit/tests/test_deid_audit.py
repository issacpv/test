"""Tests on fully synthetic, fictitious notes (no real data, no network, no models required)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deid_audit import audit, estimate, patterns, sections  # noqa: E402

CLEAN_NOTE = """Name:  ___                     Unit No:   ___

Admission Date:  2154-03-12              Discharge Date:   2154-03-18

Date of Birth:  ___             Sex:   F

Service: MEDICINE

Allergies:
No Known Allergies

Attending: ___.

Chief Complaint:
Shortness of breath

History of Present Illness:
___ year old woman with a history of heart failure who presents with dyspnea. She was seen at ___ in 2153.

Social History:
___

Brief Hospital Course:
The patient was diuresed. Echocardiogram showed an ejection fraction of 35%.

Discharge Disposition:
Home With Service

Facility:
___

Discharge Instructions:
Weigh yourself every morning, call MD if weight goes up more than 3 lbs.

Followup Instructions:
___
"""


def test_section_split_and_placeholders():
    secs = sections.split_sections(CLEAN_NOTE)
    names = [s.name for s in secs]
    assert names[0] == "Name" and "Brief Hospital Course" in names and "Followup Instructions" in names
    assert all(secs[i].end == secs[i + 1].start for i in range(len(secs) - 1))
    st = sections.placeholder_stats(CLEAN_NOTE, secs)
    assert st["ph4_total"] >= 8 and st["ph3_total"] == 0 and st["ph4__Facility"] == 1
    m3 = "Seen by [**First Name (STitle) 123**] on [**2150-3-12**] at [**Hospital1 18**]."
    hist = sections.mimic3_placeholder_types(m3)
    assert hist["DATE"] == 1 and hist["First Name (STitle)"] == 1 and hist["Hospital"] == 1
    assert sections.placeholder_stats(m3)["ph3_total"] == 3


def test_clean_note_has_no_residual_candidates_except_expected_shifted_dates():
    findings = patterns.scan(CLEAN_NOTE)
    counts = patterns.count_by_category(findings)
    assert counts.get("date_shifted", 0) == 2
    for cat in patterns.RESIDUAL_CATEGORIES:
        assert counts.get(cat, 0) == 0, cat


def test_planted_identifiers_are_detected_by_category_and_never_returned_as_text():
    rng = np.random.default_rng(0)
    planted, truth = estimate.plant_synthetic_phi(CLEAN_NOTE, rng, n_per_category=2)
    findings = patterns.scan(planted)
    counts = patterns.count_by_category(findings)
    for cat, k in truth.items():
        assert counts.get(cat, 0) >= k, cat
    for f in findings:  # findings carry no text
        assert set(f.__dataclass_fields__) == {"category", "section", "start", "length"}
    flags = patterns.note_flags(findings)
    assert flags["phone"] == 1 and flags["email"] == 1
    by_sec = patterns.count_by_category(findings, by_section=True)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in by_sec)


def test_recall_calibration_and_rates():
    rng = np.random.default_rng(1)
    docs = [estimate.plant_synthetic_phi(CLEAN_NOTE, rng, 1) for _ in range(20)]
    rec = estimate.detector_recall(lambda t: patterns.count_by_category(patterns.scan(t)), docs)
    assert set(rec["category"]) == set(docs[0][1])
    assert (rec["recall"] >= 0.9).all()
    r = estimate.rate_per_10k(3, 10000)
    assert r.rate_per_10k == 3.0 and r.low_per_10k < 3.0 < r.high_per_10k
    lo, hi = estimate.wilson_interval(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    cal = estimate.calibrated_rate(30, 0.5, 10000)
    assert cal.k == 60
    with pytest.raises(ValueError):
        estimate.calibrated_rate(1, 0.0, 10)


def test_capture_recapture_and_suppression():
    lp = estimate.lincoln_petersen(50, 40, 20)
    assert lp.estimate == 100.0 and lp.low < 100 < lp.high
    ch = estimate.chapman(50, 40, 20)
    assert abs(ch.estimate - ((51 * 41) / 21 - 1)) < 1e-9
    assert np.isfinite(estimate.chapman(5, 5, 0).estimate)
    a = np.r_[np.ones(30), np.zeros(70)]
    b = np.r_[np.ones(15), np.zeros(15), np.ones(15), np.zeros(55)]
    cr = estimate.capture_recapture_from_flags(a, b)
    assert cr.n1 == 30 and cr.n2 == 30 and cr.m == 15 and cr.estimate > 30
    import pandas as pd

    df = pd.DataFrame({"g": ["a", "b", "c"], "k": [0, 4, 25]})
    s = estimate.suppress_small_cells(df, ["k"], 10)
    assert s["k"].tolist()[0] == 0 and np.isnan(s["k"].tolist()[1]) and s["k"].tolist()[2] == 25


def test_aggregate_and_summarise_contain_counts_only():
    rng = np.random.default_rng(2)
    recs = [("n0", "discharge", CLEAN_NOTE)]
    for i in range(1, 12):
        recs.append((f"n{i}", "discharge", estimate.plant_synthetic_phi(CLEAN_NOTE, rng, 1, ["phone", "email"])[0]))
    recs.append(("r0", "radiology", "EXAMINATION: CXR\nFINDINGS: Clear lungs.\nIMPRESSION: No acute process."))
    df = audit.aggregate_notes(recs, nlp=None)
    assert len(df) == 13 and "note_id" in df and not any(c == "text" for c in df.columns)
    assert df.loc[df.note_id == "n0", "flag_phone"].item() == 0 and df["flag_phone"].sum() == 11
    assert (df.drop(columns=["note_id", "note_type"]).dtypes == int).all()
    summ = audit.summarise(df, min_cell=10)
    row = summ[(summ.note_type == "discharge") & (summ.category == "phone")].iloc[0]
    assert row["notes_with_candidate"] == 11 and abs(row["rate_per_10k"] - 1e4 * 11 / 12) < 1e-6
    small = summ[(summ.note_type == "radiology") & (summ.category == "phone")].iloc[0]
    assert small["notes_with_candidate"] == 0
    with pytest.raises(ValueError):
        audit.regex_vs_ner_capture(df)


def test_optional_local_ner_is_skipped_gracefully():
    nlp = audit.load_local_ner("en_core_web_sm")
    if nlp is None:
        assert audit.ner_counts("Dr. John Doe was seen in Boston.", None) == Counter()
        pytest.skip("no local spaCy model installed")
    c = audit.ner_counts("Dr. John Doe was seen in Boston.", nlp)
    assert sum(c.values()) >= 1
