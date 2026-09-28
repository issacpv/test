"""Synthetic-data tests for neoclock (no downloads, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neoclock import clock, features_neural as fn, normative, outcomes  # noqa: E402

FS = 128.0


def synthetic_neonatal_eeg(pma: float, seconds: float = 120.0, n_ch: int = 4, seed: int = 0, delay_weeks: float = 0.0):
    """Discontinuous neonatal-like EEG: bursts whose fraction and rate grow with PMA; IBIs shrink.

    ``delay_weeks`` makes the signal look younger than its nominal PMA (for outcome tests).
    """
    rng = np.random.default_rng(seed)
    eff = pma - delay_weeks
    n = int(seconds * FS)
    t = np.arange(n) / FS
    continuity = float(np.clip(0.35 + 0.065 * (eff - 35.0), 0.2, 0.98))
    ibi_mean = float(np.clip(12.0 - 1.0 * (eff - 35.0), 1.5, 15.0))
    X = np.zeros((n_ch, n))
    # shared burst schedule with per-hemisphere jitter that decreases with age
    mask = np.zeros(n, bool)
    tt = 0.0
    while tt < seconds:
        burst = rng.uniform(2.0, 6.0)
        mask[int(tt * FS):int(min(seconds, tt + burst) * FS)] = True
        tt += burst + rng.exponential(ibi_mean) * (1 - continuity) * 2
    for c in range(n_ch):
        noise = rng.normal(size=n)
        f_delta = 1.0 + 0.1 * (eff - 35.0)  # delta rhythm slightly faster with maturation
        delta = np.sin(2 * np.pi * f_delta * t + rng.uniform(0, 2 * np.pi)) + 0.5 * np.sin(2 * np.pi * (2 * f_delta + 0.3) * t)
        amp = 50.0 * (1.0 + 0.3 * (45.0 - eff) / 10.0)  # higher-amplitude bursts when younger
        sig = 10.0 * noise + amp * delta
        jitter = int(rng.integers(0, max(2, int(FS * (0.6 - 0.05 * (eff - 35.0))))))
        m = np.roll(mask, jitter if c % 2 else 0)
        X[c] = np.where(m, sig, 3.0 * noise)
    return X


def test_features_track_maturation():
    ages = np.array([35, 37, 39, 41, 43, 45], float)
    rows = [fn.record_features(synthetic_neonatal_eeg(a, seed=i), FS, homologous_pairs=[(0, 1), (2, 3)]) for i, a in enumerate(ages)]
    names = fn.feature_names()
    F = fn.features_to_matrix(rows, names)
    assert F.shape == (6, len(names)) and np.isfinite(F[:, names.index("continuity")]).all()
    cont = F[:, names.index("continuity")]
    ibi = F[:, names.index("ibi_median_s")]
    assert np.corrcoef(cont, ages)[0, 1] > 0.7
    assert np.corrcoef(ibi, ages)[0, 1] < -0.5
    r = fn.reeg(synthetic_neonatal_eeg(40)[0], FS)
    assert r["reeg_p95"] >= r["reeg_p50"] >= r["reeg_p5"]
    stats_ = fn.interburst_stats(np.r_[np.ones(200, bool), np.zeros(100, bool), np.ones(200, bool)], 100.0)
    assert stats_["ibi_median_s"] == pytest.approx(1.0) and stats_["continuity"] == pytest.approx(0.8)


def _cohort(n_sub: int = 36, seed: int = 0, delay=None):
    rng = np.random.default_rng(seed)
    names = fn.feature_names()
    F, pma, groups, delays = [], [], [], []
    for s in range(n_sub):
        a = float(rng.uniform(35, 45))
        d = 0.0 if delay is None else float(delay(rng))
        for e in range(2):  # two epochs per infant
            X = synthetic_neonatal_eeg(a, seconds=90.0, seed=seed * 1000 + s * 10 + e, delay_weeks=d)
            F.append(fn.record_features(X, FS, homologous_pairs=[(0, 1), (2, 3)]))
            pma.append(a)
            groups.append(f"s{s}")
            delays.append(d)
    return fn.features_to_matrix(F, names), np.asarray(pma), np.asarray(groups), np.asarray(delays), names


def test_clock_accuracy_bias_correction_and_icc():
    F, pma, groups, _, names = _cohort()
    ck = clock.MaturationClock()
    cv = ck.cross_val_predict(F, pma, groups, n_splits=4)
    assert clock.mae(cv["pred"], pma) < 2.0
    assert np.corrcoef(cv["pred"], pma)[0, 1] > 0.7
    raw_bias = clock.delta_bias(cv["delta_raw"], pma)
    corr_bias = clock.delta_bias(cv["delta_corrected"], pma)
    assert abs(corr_bias["r"]) < abs(raw_bias["r"]) + 0.05 and abs(corr_bias["r"]) < 0.3
    icc = clock.icc_oneway(cv["delta_corrected"], groups)
    assert np.isfinite(icc)
    tab = clock.mae_by_age_bin(cv["pred"], pma)
    assert tab["n"].sum() == pma.size
    ck.fit(F, pma)
    assert ck.delta(F, pma).shape == pma.shape
    null = clock.permutation_mae_null(F, pma, groups, n_perm=5)
    assert null["null_mean"] > null["mae"]


def test_normative_centiles():
    F, pma, groups, _, names = _cohort(n_sub=30, seed=1)
    nm = normative.FeatureCentiles().fit(F, pma, names)
    Z = nm.zscore(F, pma)
    assert Z.shape == F.shape
    assert abs(np.nanmean(Z)) < 0.3 and 0.5 < np.nanstd(Z) < 1.6
    C = nm.centile(F, pma)
    assert (C >= 0).all() and (C <= 100).all()
    tab = nm.centile_table([36, 40, 44])
    assert (tab["c97"] >= tab["c3"]).all() and len(tab) == 3 * len(names)
    dev = normative.multivariate_deviation(Z)
    assert dev.shape == (F.shape[0],)
    # a strongly delayed recording is flagged
    Xd = synthetic_neonatal_eeg(44.0, seconds=90.0, seed=77, delay_weeks=8.0)
    Fd = fn.features_to_matrix([fn.record_features(Xd, FS, homologous_pairs=[(0, 1), (2, 3)])], names)
    assert normative.multivariate_deviation(nm.zscore(Fd, np.array([44.0])))[0] > dev.mean()
    assert len(normative.calibration_check(Z, pma)) == 4


def test_outcomes_trend_and_burden():
    rng = np.random.default_rng(0)
    grade = np.repeat([0, 1, 2, 3], 15)
    delta = -1.0 * grade + rng.normal(0, 1.0, grade.size)
    jt = outcomes.jonckheere_terpstra(delta, grade, n_perm=300)
    assert jt["p"] < 0.01 and jt["z"] < 0
    jt0 = outcomes.jonckheere_terpstra(rng.normal(size=60), grade, n_perm=300)
    assert jt0["p"] > 0.05
    x = rng.normal(size=80)
    cov = rng.normal(size=80)
    y = -0.8 * x + 2.0 * cov + rng.normal(size=80)
    ps = outcomes.partial_spearman(x, y, cov, n_perm=300)
    assert ps["rho"] < -0.4 and ps["p"] < 0.01
    assert outcomes.auroc_binary(delta, (grade >= 2).astype(int)) < 0.3
    assert outcomes.seizure_burden([(0, 60), (30, 120), (600, 660)], 3600) == pytest.approx(3.0)
    cons = outcomes.consensus_events([[(10, 20), (40, 50)], [(12, 25)], [(8, 22), (45, 49)]], 60)
    assert cons == [(12.0, 20.0)]
    lo, hi = outcomes.bootstrap_ci(delta, n_boot=200)
    assert lo < np.median(delta) < hi
