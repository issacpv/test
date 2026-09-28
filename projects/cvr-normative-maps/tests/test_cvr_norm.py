"""Synthetic tests for cvr_norm (PYTHONPATH=src)."""
import numpy as np
import pytest

from cvr_norm import cvr_model, normative, physio, reliability


def test_end_tidal_detection_and_interpolation():
    fs = 40.0
    t = np.arange(0, 120, 1 / fs)
    breaths = 0.25  # Hz (15 breaths / min)
    petco2_true = 38 + 4 * (t > 60)                       # step increase at 60 s
    co2 = petco2_true * np.clip(np.sin(2 * np.pi * breaths * t), 0, None) ** 3
    t_pk, v_pk, pet = physio.end_tidal_co2(co2, fs)
    assert 25 <= t_pk.size <= 32
    assert abs(np.median(pet[t < 55]) - 38) < 1.0 and abs(np.median(pet[t > 65]) - 42) < 1.0


def test_rvt_and_compliance():
    fs = 40.0
    t = np.arange(0, 100, 1 / fs)
    resp = np.sin(2 * np.pi * 0.25 * t)
    onsets, durs = np.array([30.0, 70.0]), np.array([15.0, 15.0])
    hold = physio.boxcar(t.size, fs, onsets, durs).astype(bool)
    resp[hold] = 0.02 * np.random.default_rng(0).standard_normal(hold.sum())
    r = physio.rvt(resp, fs)
    assert r.shape == t.shape and np.all(r >= 0)
    frac, ratios = physio.breath_hold_compliance(resp, fs, onsets, durs)
    assert frac == 1.0 and np.all(ratios < 0.05)
    resp_bad = np.sin(2 * np.pi * 0.25 * t)               # never held breath
    frac_bad, _ = physio.breath_hold_compliance(resp_bad, fs, onsets, durs)
    assert frac_bad == 0.0


def test_cvr_fit_recovers_amplitude_and_delay():
    rng = np.random.default_rng(1)
    tr, n_vols = 1.4, 200
    onsets = np.array([20.0, 60.0, 100.0, 140.0, 180.0, 220.0])
    durs = np.full(onsets.size, 15.0)
    amps = np.array([2.0, 1.0, 0.5, 3.0])
    delays = np.array([0.0, 3.0, -2.1, 6.0])
    bold, reg = cvr_model.simulate_breath_hold_bold(n_vols, tr, onsets, durs, amps, delays, reg_fs=10.0,
                                                    noise_sd=0.2, rng=rng)
    res = cvr_model.fit_cvr(bold, reg, reg_fs=10.0, tr=tr, lag_range=(-9, 9), lag_step=0.3, drift_order=3)
    assert np.allclose(res.amplitude, amps, rtol=0.15, atol=0.15)
    assert np.allclose(res.delay, delays, atol=0.7)
    assert np.all(res.r2 > 0.3) and np.all(res.tstat > 3)
    floor = cvr_model.r2_floor_from_phase_randomization(bold, reg, 10.0, tr, n_null=3, rng=rng, lag_step=1.0)
    assert 0.0 <= floor < res.r2.min()


def test_normative_model_centiles_and_covariates():
    rng = np.random.default_rng(2)
    n = 2000
    age = rng.uniform(6, 85, n)
    sex = rng.integers(0, 2, n)
    mu = 0.25 + 0.012 * age - 0.00022 * age**2 - 0.03 * sex      # inverted U, peak ~27 y
    sd = 0.04 + 0.0006 * age
    y = mu + sd * rng.standard_normal(n)
    m = normative.CVRNormativeModel(df=6).fit(age, y, cov=sex)
    z = m.zscore(age, y, cov=sex)
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.15
    grid = np.linspace(6, 85, 200)
    pk = normative.peak_age(m, grid, cov=np.zeros(200))
    assert 18 <= pk <= 38
    curves = m.curves(grid, cov=np.zeros(200))
    width = curves[0.95] - curves[0.05]
    assert width[-1] > width[0]                                   # heteroscedastic
    lo, hi = normative.extreme_deviation_rate(z)
    assert 0.005 < lo < 0.06 and 0.005 < hi < 0.06
    c = m.centile(np.array([30.0]), np.array([mu[0] * 0 + 0.45]), cov=np.array([0]))
    assert 0 <= c[0] <= 100


def test_reliability_metrics():
    rng = np.random.default_rng(3)
    subj = rng.normal(1.0, 0.3, 40)
    sessions = np.stack([subj + 0.1 * rng.standard_normal(40) for _ in range(5)], axis=1)
    summ = reliability.retest_summary(sessions, n_boot=100, rng=rng)
    assert 0.8 < summ["icc"] < 1.0 and summ["icc_ci_low"] < summ["icc"] < summ["icc_ci_high"]
    assert 0 < summ["cov_within_pct"] < 25
    assert 0 < summ["mdc95"] < 0.5
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 0, 0], bool)
    assert reliability.dice(a, b) == pytest.approx(2 / 3)
