"""Synthetic-data tests for qt_dose core functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qt_dose import corrections, exposure, linkage, models, external


def test_corrections_normal_qtc():
    # QT 0.4 s at RR 1.0 s (HR 60) -> ~400 ms under every formula
    for m in ("bazett", "fridericia", "framingham", "hodges"):
        qtc = corrections.correct([0.4], [1.0], method=m)[0]
        assert 395 <= qtc <= 405


def test_fit_power_correction_recovers_alpha():
    rng = np.random.default_rng(0)
    rr = rng.uniform(0.6, 1.2, size=500)
    alpha_true = 0.35
    qt = 0.40 * rr ** alpha_true * np.exp(rng.normal(0, 0.01, size=rr.size))
    alpha = corrections.fit_power_correction(qt, rr)
    assert abs(alpha - alpha_true) < 0.05


def test_map_ingredient_and_dose():
    assert exposure.map_ingredient("Sotalol HCl 80 mg Tab") == "sotalol"
    assert exposure.map_ingredient("aspirin 81mg") == "aspirin"
    assert exposure.map_ingredient("insulin glargine") is None
    assert exposure.parse_dose_mg(80, "mg") == 80.0
    assert exposure.parse_dose_mg(0.5, "g") == 500.0
    assert abs(exposure.parse_dose_mg(500, "mcg") - 0.5) < 1e-9


def test_build_exposures():
    emar = pd.DataFrame(
        {
            "emar_id": [1, 2, 3],
            "subject_id": [10, 10, 11],
            "charttime": ["2150-01-01 08:00", "2150-01-01 20:00", "2150-02-02 09:00"],
            "medication": ["Sotalol 80mg", "Aspirin 81mg", "Insulin"],
            "event_txt": ["Administered", "Administered", "Administered"],
        }
    )
    detail = pd.DataFrame(
        {"emar_id": [1, 2, 3], "dose_given": [80, 81, 10],
         "dose_given_unit": ["mg", "mg", "unit"], "route": ["PO", "PO", "SC"]}
    )
    ex = exposure.build_exposures(emar, detail)
    assert set(ex["ingredient"]) == {"sotalol", "aspirin"}
    assert ex.loc[ex["ingredient"] == "sotalol", "dose_mg"].iloc[0] == 80.0


def _synthetic_pairs(true_slope=0.05, n_pat=40, seed=0):
    """Generate pre/post pairs where Delta-QTc = slope*dose + patient effect + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_pat):
        pat_eff = rng.normal(0, 5)
        for _ in range(rng.integers(2, 5)):
            dose = rng.choice([40, 80, 120, 160])
            h = rng.uniform(1, 12)
            dq = true_slope * dose + pat_eff + rng.normal(0, 4)
            rows.append({"subject_id": sid, "ingredient": "sotalol", "dose_mg": float(dose),
                         "delta_qtc": dq, "hours_since_dose": h})
    return pd.DataFrame(rows)


def test_dose_slope_recovers_positive_effect():
    pairs = _synthetic_pairs(true_slope=0.05)
    res = models.fit_dose_slope(pairs, "sotalol")
    assert res.slope > 0
    assert res.significant
    # dose for +10 ms should be near 10/0.05 = 200 mg
    assert 120 < models.dose_for_threshold(res, 10.0) < 320


def test_negative_control_null():
    # For a true-null drug the permutation p-value is ~uniform, so average over
    # several synthetic draws: the mean should sit well above the 0.05 line.
    ps = []
    for s in range(8):
        pairs = _synthetic_pairs(true_slope=0.0, seed=s)
        pairs["ingredient"] = "aspirin"
        ps.append(models.permutation_null(pairs, "aspirin", n_perm=300, seed=s))
    assert np.nanmean(ps) > 0.2


def test_link_pre_post_basic():
    ecgs = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "ecg_time": ["2150-01-01 06:00", "2150-01-01 10:00", "2150-01-01 14:00"],
            "qtc": [400.0, 430.0, 445.0],
        }
    )
    admins = pd.DataFrame(
        {"subject_id": [1], "ingredient": ["sotalol"], "charttime": ["2150-01-01 08:00"],
         "dose_mg": [80.0], "route": ["PO"]}
    )
    pairs = linkage.link_pre_post(ecgs, admins, qtc_col="qtc")
    assert len(pairs) == 2  # two post-dose ECGs
    assert pairs["baseline_qtc"].iloc[0] == 400.0
    assert pairs["delta_qtc"].iloc[0] == 30.0


def test_reporting_odds_ratio_and_concordance():
    r = external.reporting_odds_ratio(50, 950, 10, 9990)
    assert r["ror"] > 1
    slopes = pd.DataFrame({"ingredient": ["sotalol", "aspirin", "haloperidol"],
                           "slope": [0.06, 0.001, 0.04]})
    cm = pd.DataFrame({"drug": ["sotalol", "aspirin", "haloperidol"],
                       "risk_category": ["Known", "No known risk", "Known"]})
    out = external.slope_vs_crediblemeds(slopes, cm)
    assert out["n"] == 3
    assert out["rho"] > 0
