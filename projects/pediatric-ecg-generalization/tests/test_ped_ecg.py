"""Synthetic-data tests for ped_ecg."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ped_ecg import pediatric_norms as pn
from ped_ecg import labels, decay, adaptation


def test_age_band_boundaries():
    assert pn.age_band(0.02) == "neonate"
    assert pn.age_band(0.5) == "6-12mo"
    assert pn.age_band(10) == "8-12y"
    assert pn.age_band(40) == "adult"


def test_zscore_for_age_direction():
    # An adult resting HR of 70 is normal for an adult (z~0) but bradycardic for a neonate (z<0).
    z_adult = pn.zscore_for_age("hr", 70, 40)
    z_neonate = pn.zscore_for_age("hr", 70, 0.02)
    assert abs(z_adult) < 0.5
    assert z_neonate < -2


def test_age_normalise_frame():
    df = pd.DataFrame({"age_years": [0.02, 10, 40], "hr": [145, 85, 70], "qt": [280, 340, 390]})
    out = pn.age_normalise_frame(df)
    assert "hr_z" in out and "qt_z" in out
    assert abs(out["hr_z"].iloc[0]) < 0.5  # 145 is normal for a neonate


def test_label_maps():
    assert labels.map_ptbxl("CRBBB") == "RBBB"
    assert labels.map_snomed("164889003") == "AF"
    assert "PVC" in labels.map_zzu_statement("Frequent premature ventricular complexes")
    assert labels.is_shared("AF") and not labels.is_shared("CHD")


def _synthetic_age_decay(n=1200, seed=0):
    """Scores whose separability from label degrades toward young ages."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 14, n)
    y = rng.integers(0, 2, n)
    # signal strength grows with age -> AUROC increases with age
    strength = 0.2 + 0.12 * age
    scores = y * strength + rng.normal(0, 1, n)
    return age, y, scores


def test_auroc_vs_age_increases():
    age, y, scores = _synthetic_age_decay()
    res = decay.auroc_vs_age(age, y, scores, bw=2.5, n_boot=100)
    finite = np.isfinite(res["auroc"])
    # AUROC should be higher at older centers than younger
    lo_age = res["auroc"][finite][0]
    hi_age = res["auroc"][finite][-1]
    assert hi_age > lo_age


def test_age_permutation_null_detects_dependence():
    age, y, scores = _synthetic_age_decay()
    out = decay.age_permutation_null(age, y, scores, bw=2.5, n_perm=200)
    assert out["p"] < 0.05  # AUROC genuinely depends on age


def test_confident_error_rate():
    rng = np.random.default_rng(0)
    age = rng.uniform(0, 14, 400)
    # infants get confident-but-wrong positives
    y = np.where(age < 1, 0, rng.integers(0, 2, 400))
    scores = np.where(age < 1, 0.9, rng.uniform(0, 1, 400))
    bands = {"infant": (0, 1), "older": (1, 14)}
    out = decay.confident_error_rate(y, scores, age, bands)
    assert out["infant"] > out["older"]


def test_decompose_gap_sums_to_total():
    # A toy where each op adds a fixed increment.
    increments = {"covariate": 0.06, "prior": 0.03, "recal": 0.01}

    def base(active):
        return 0.7 + sum(increments[n] for n in active)

    contrib = adaptation.decompose_gap({k: None for k in increments}, base)
    total = base(frozenset(increments)) - base(frozenset())
    assert abs(sum(contrib.values()) - total) < 1e-9
    assert contrib["covariate"] > contrib["recal"]


def test_label_efficiency_curve():
    rng = np.random.default_rng(0)
    d = 20
    Ztr = rng.normal(size=(1500, d))
    w = rng.normal(size=d)
    ytr = (Ztr @ w + rng.normal(0, 1, 1500) > 0).astype(int)
    Zte = rng.normal(size=(400, d))
    yte = (Zte @ w + rng.normal(0, 1, 400) > 0).astype(int)
    res = adaptation.label_efficiency_curve(Ztr, ytr, Zte, yte)
    assert 0.5 < res["ceiling"] <= 1.0
    k = adaptation.labels_to_reach(res["curve"], res["ceiling"], tol=0.05)
    assert k is None or k <= 1000
