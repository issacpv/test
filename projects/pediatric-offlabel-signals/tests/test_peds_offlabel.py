"""Synthetic tests for peds_offlabel (no network)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from peds_offlabel.age import band_from_years, flatten_report, pediatric_band
from peds_offlabel.label_ages import classify_age, extract_age_floor, floor_from_label
from peds_offlabel.offlabel_signals import class_shares_by_period, classify_reports, offlabel_signal_table, segmented_poisson_its, seriousness_model


def test_age_bands_and_flatten():
    assert band_from_years(0.01) == "neonate" and band_from_years(1.0) == "infant" and band_from_years(5) == "child"
    assert band_from_years(15) == "adolescent" and band_from_years(30) == "adult" and band_from_years(None) is None
    assert pediatric_band(None, "3") == "child" and pediatric_band(None, None) is None
    rec = {
        "safetyreportid": "1",
        "receivedate": "20240110",
        "serious": "1",
        "seriousnessdeath": "2",
        "patient": {
            "patientsex": "1",
            "patientonsetage": "18",
            "patientonsetageunit": "802",
            "drug": [{"drugcharacterization": "1", "openfda": {"generic_name": ["QUETIAPINE FUMARATE"]}, "drugindication": "Insomnia"}],
            "reaction": [{"reactionmeddrapt": "Somnolence"}],
        },
    }
    row = flatten_report(rec)
    assert row["age_band"] == "infant" and row["pediatric"] and row["serious"] and not row["death"]
    assert row["suspect"] == ["QUETIAPINE FUMARATE"] and row["reactions"] == ["SOMNOLENCE"]


def test_extract_age_floor_patterns():
    t1 = "Safety and effectiveness of DRUG have been established in pediatric patients 6 years of age and older. Safety and effectiveness in pediatric patients below the age of 6 years have not been established."
    r = extract_age_floor(t1)
    assert r["status"] == "approved" and r["min_age_years"] == 6.0
    t2 = "Safety and effectiveness in pediatric patients have not been established."
    r = extract_age_floor(t2)
    assert r["status"] == "not_established" and r["min_age_years"] is None
    t3 = "DRUG is indicated for the treatment of hypertension in adults and pediatric patients aged 12 years and older."
    assert extract_age_floor(t3)["min_age_years"] == 12.0
    t4 = "The safety and effectiveness have been established in pediatric patients 1 month to less than 2 years of age."
    assert abs(extract_age_floor(t4)["min_age_years"] - 1 / 12) < 1e-9
    t5 = "Use in pediatric patients from birth is supported by evidence from adequate and well-controlled studies."
    assert extract_age_floor(t5)["min_age_years"] == 0.0
    t6 = "Safety and effectiveness in pediatric patients younger than two years of age have not been established."
    assert extract_age_floor(t6)["min_age_years"] == 2.0 and extract_age_floor(t6)["status"] == "approved"
    t7 = "Pediatric patients weighing at least 40 kg: same dose as adults. Safety in pediatric patients 2 years and older has been established."
    r = extract_age_floor(t7)
    assert r["weight_based"] and r["min_age_years"] == 2.0
    assert extract_age_floor("Renal impairment: no dose adjustment.")["status"] == "unknown"
    lab = {"pediatric_use": t2, "indications_and_usage": t3}
    f = floor_from_label(lab)
    assert f["status"] == "approved" and f["min_age_years"] == 12.0 and f["section"] == "indications_and_usage"
    assert classify_age(10, f) == "below_floor" and classify_age(14, f) == "on_label_age" and classify_age(30, f) == "adult"
    assert classify_age(10, {"status": "not_established", "min_age_years": None}) == "no_pediatric_labeling"


def simulate(n: int = 6000, seed: int = 0) -> List[Dict[str, Any]]:
    """Paediatric reports for drug D (floor 6 y) and other drugs.

    Event 'X' is enriched among D reports below the floor beyond the age
    background; 'Y' is an age effect only (more common in young children on
    any drug); D reports below floor are more often serious.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        age = float(rng.uniform(0.1, 17.9))
        on_d = rng.random() < 0.3
        young = age < 6
        px = 0.03 + (0.25 if (on_d and young) else 0.0) + (0.02 if on_d else 0.0)
        py = 0.05 + (0.20 if young else 0.0)
        pts = [e for e, p in (("X", px), ("Y", py), ("NAUSEA", 0.15)) if rng.random() < p]
        serious = rng.random() < (0.5 if (on_d and young) else 0.25)
        band = "infant" if age < 2 else ("child" if age < 12 else "adolescent")
        y, m = 2018 + i * 8 // n, (i * 96 // n) % 12 + 1
        out.append({"safetyreportid": str(i), "receivedate": f"{y}{m:02d}10", "age_years": age, "age_band": band, "sex": "female" if rng.random() < 0.5 else "male", "qualification": "1" if rng.random() < 0.5 else "5", "serious": serious, "death": False, "suspect": ["D"] if on_d else ["OTHER"], "reactions": pts})
    return out


def test_offlabel_signal_table_and_seriousness():
    reps = simulate()
    floors = {"D": {"status": "approved", "min_age_years": 6.0}}
    cl = classify_reports(reps, floors)
    assert set(cl["label_class"]) == {"below_floor", "on_label_age"}
    tab = offlabel_signal_table(reps, cl, "D", min_a=3).set_index("event")
    assert tab.loc["X", "ratio"] > 2 and tab.loc["X", "offlabel_excess"]
    # Y is an age effect only: age-band-specific backgrounds remove it
    assert not tab.loc["Y", "offlabel_excess"] and 0.5 < tab.loc["Y", "ratio"] < 2.0
    sm_tab = seriousness_model(cl)
    assert sm_tab.loc["below_floor", "or"] > 1.5 and sm_tab.loc["below_floor", "p"] < 0.01
    shares = class_shares_by_period(cl, reps, freq="Y")
    assert "share_below_floor" in shares.columns and len(shares) >= 8


def test_segmented_poisson_its_detects_level_change():
    rng = np.random.default_rng(1)
    base = 50 * np.exp(0.01 * np.arange(60))
    y = rng.poisson(np.where(np.arange(60) >= 36, base * 0.5, base))
    res = segmented_poisson_its(y, change_index=36)
    assert res["level_hi"] < 0.8 and res["p_level"] < 0.001
    assert res["slope_lo"] < 1.0 < res["slope_hi"]
