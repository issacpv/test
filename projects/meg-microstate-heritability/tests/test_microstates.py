"""Synthetic-data tests for microstate segmentation, HRV and twin statistics."""
import numpy as np
import pytest

from meg_microstates.heritability import (
    ace_model_comparison,
    falconer,
    fit_ace,
    permutation_mz_dz_difference,
    power_simulation,
    twin_icc,
    twin_identification,
)
from meg_microstates.hrv import cardiac_phase_histogram, clean_rr, detect_r_peaks, frequency_domain_hrv, rr_intervals, time_domain_hrv
from meg_microstates.microstates import (
    dfa_exponent,
    gfp,
    gfp_peaks,
    match_templates,
    microstate_parameters,
    segment,
    smooth_labels,
)
from meg_microstates.synthetic import simulate_ecg, simulate_microstate_data, simulate_twin_phenotypes


def test_gfp_peaks_min_distance():
    g = np.array([0, 1, 0, 2, 0, 3, 0, 1, 0], dtype=float)
    assert gfp_peaks(g).tolist() == [1, 3, 5, 7]
    assert gfp_peaks(g, min_distance=3).tolist() == [1, 5]  # keeps the larger of close peaks


def test_segmentation_recovers_templates_and_labels():
    sim = simulate_microstate_data(n_channels=24, sfreq=250.0, duration_s=40.0, n_states=4, snr=4.0, seed=1)
    out = segment(sim["data"], 250.0, n_states=4, min_duration_ms=12.0, seed=0)
    perm, corrs = match_templates(out["templates"], sim["templates"])
    assert np.all(np.abs(corrs) > 0.9)
    mapped = perm[out["labels"]]
    assert np.mean(mapped == sim["labels"]) > 0.85
    assert out["gev"] > 0.5
    params = out["params"]
    assert params["coverage"].sum() == pytest.approx(1.0)
    assert np.all(params["duration"] > 0.02)  # ~80 ms planted mean durations
    rows = params["transition"].sum(axis=1)
    assert np.allclose(rows[rows > 0], 1.0)
    assert np.allclose(np.diag(params["transition"]), 0.0)


def test_smooth_labels_removes_short_segments():
    labels = np.array([0] * 10 + [1] * 2 + [0] * 10 + [2] * 8)
    sm = smooth_labels(labels, min_samples=4)
    assert np.all(sm[:22] == 0) and np.all(sm[22:] == 2)


def test_microstate_parameters_and_dfa():
    rng = np.random.default_rng(0)
    labels = np.repeat(rng.integers(0, 4, size=400), 20)
    p = microstate_parameters(labels, sfreq=250.0, n_states=4)
    assert p["duration"].shape == (4,) and np.isfinite(p["entropy_rate"])
    white = rng.normal(size=4096)
    assert 0.35 < dfa_exponent(white) < 0.65
    assert dfa_exponent(np.cumsum(white)) > 1.2


def test_r_peak_detection_and_hrv():
    sim = simulate_ecg(duration_s=150.0, sfreq=500.0, hr_bpm=70, rr_sd_s=0.04, seed=2)
    peaks = detect_r_peaks(sim["ecg"], 500.0)
    assert abs(len(peaks) - len(sim["peaks"])) <= 2
    # every true peak matched within 20 ms
    d = np.min(np.abs(peaks[:, None] - sim["peaks"][None, :]), axis=0)
    assert np.percentile(d, 95) <= 10
    rr, t = rr_intervals(peaks, 500.0)
    rr, t, dropped = clean_rr(rr, t)
    assert dropped < 0.1
    td = time_domain_hrv(rr)
    assert 60 < td["hr_bpm"] < 80 and 20 < td["sdnn_ms"] < 70
    fd = frequency_domain_hrv(rr, t)
    assert np.isfinite(fd["lf_hf"]) and fd["lf"] >= 0


def test_cardiac_phase_histogram_null_is_calibrated():
    rng = np.random.default_rng(3)
    labels = np.repeat(rng.integers(0, 3, size=600), 25)  # 15000 samples
    peaks = np.arange(100, 15000, 220)
    res = cardiac_phase_histogram(labels, peaks, n_states=3, n_bins=6, n_perm=100)
    assert res["observed"].shape == (3, 6)
    assert np.all(res["p_perm"] > 0.01)  # no coupling planted


def test_ace_recovers_heritability():
    mz, dz = simulate_twin_phenotypes(400, 400, h2=0.6, c2=0.1, seed=5)
    r_mz, r_dz = twin_icc(mz), twin_icc(dz)
    assert r_mz > r_dz
    f = fit_ace(mz, dz, "ACE")
    assert f["h2"] == pytest.approx(0.6, abs=0.15)
    assert f["c2"] == pytest.approx(0.1, abs=0.15)
    cmp_ = ace_model_comparison(mz, dz)
    assert cmp_["lrt_drop_A"]["p"] < 0.01
    fal = falconer(r_mz, r_dz)
    assert fal["h2"] == pytest.approx(0.6, abs=0.2)
    perm = permutation_mz_dz_difference(mz, dz, n_perm=100)
    assert perm["p_perm"] < 0.05


def test_ace_model_comparison_null_heritability():
    mz, dz = simulate_twin_phenotypes(300, 300, h2=0.0, c2=0.4, seed=8)
    cmp_ = ace_model_comparison(mz, dz)
    assert cmp_["fits"]["ACE"]["h2"] < 0.2
    assert cmp_["lrt_drop_C"]["p"] < 0.01


def test_twin_identification_and_power():
    rng = np.random.default_rng(1)
    n_pairs, n_feat = 20, 12
    shared = rng.normal(size=(n_pairs, n_feat))
    feats = np.vstack([shared + 0.4 * rng.normal(size=shared.shape), shared + 0.4 * rng.normal(size=shared.shape)])
    pairs = [(i, i + n_pairs) for i in range(n_pairs)]
    res = twin_identification(feats, pairs, n_perm=200)
    assert res["accuracy"] > 0.7 and res["p_perm"] < 0.05
    pw = power_simulation(17, 12, h2=0.9, c2=0.0, n_sim=6, seed=0)
    assert 0 <= pw["power"] <= 1
