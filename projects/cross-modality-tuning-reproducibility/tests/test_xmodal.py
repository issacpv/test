"""Synthetic-data tests for xmodal."""
import numpy as np
import pandas as pd
import pytest

from xmodal import forward_model as fm
from xmodal import multiverse as mv
from xmodal import reliability as rl
from xmodal import tuning as tn


def test_osi_dsi_and_sparseness():
    dirs = np.arange(0, 360, 45)
    sharp = 1.0 + 20.0 * np.exp(6.0 * (np.cos(np.deg2rad(dirs - 90)) - 1))
    sel = tn.osi_dsi(sharp, dirs)
    assert sel["osi_ratio"] > 0.8 and sel["dsi_ratio"] > 0.8 and sel["osi_vec"] > 0.5
    assert sel["pref_dir_deg"] == 90.0 and sel["pref_ori_deg"] == 90.0
    flat = tn.osi_dsi(np.full(8, 5.0), dirs)
    assert flat["osi_vec"] == pytest.approx(0.0, abs=1e-9) and flat["osi_ratio"] == pytest.approx(0.0)
    assert tn.lifetime_sparseness(np.eye(8)[0]) == pytest.approx(1.0)
    assert tn.lifetime_sparseness(np.ones(8)) == pytest.approx(0.0)
    assert tn.preferred_condition([1, 5, 2], [1, 2, 4]) == 2


def test_tuning_metrics_with_secondary_and_responsiveness():
    rng = np.random.default_rng(0)
    dirs = np.tile(np.arange(0, 360, 45), 15 * 3)
    tf = np.repeat([1, 2, 4], 15 * 8)
    resp = tn.synthetic_tuned_responses(rng, dirs, pref_deg=180.0, peak=10.0, base=1.0, noise_sd=0.5)
    resp[tf == 2] *= 2.0  # preferred TF = 2
    m = tn.tuning_metrics(resp, dirs, secondary=tf, secondary_levels=[1, 2, 4])
    assert m["pref_secondary"] == 2 and abs(m["pref_dir_deg"] - 180.0) < 1e-9
    assert m["osi_ratio"] > 0.5
    ev = rng.normal(2.0, 1.0, 40)
    bl = rng.normal(0.0, 1.0, 40)
    assert tn.responsiveness_permutation(ev, bl, n_perm=300)["p"] < 0.01
    assert tn.responsiveness_permutation(bl, rng.normal(0.0, 1.0, 40), n_perm=300)["p"] > 0.05
    r = tn.responsiveness_anova_like(resp[:80], dirs[:80], rng.normal(1.0, 0.5, 30), n_perm=100)
    assert r["p"] < 0.05


def test_forward_model_and_deconvolution():
    rng = np.random.default_rng(1)
    fs, T = 30.0, 20.0
    spikes = np.array([2.0, 2.05, 2.1, 8.0, 14.0])
    f, t = fm.spikes_to_fluorescence(spikes, fs, T, fm.INDICATORS["GCaMP6f"], noise_sd=0.0, rng=rng)
    assert f.shape == t.shape == (600,)
    assert f[int(1.9 * fs)] == pytest.approx(0.0, abs=1e-9)
    peak_t = t[np.argmax(f)]
    assert 2.05 < peak_t < 2.6  # burst peak shortly after the three spikes
    # supralinearity amplifies the burst relative to single spikes
    f_lin, _ = fm.spikes_to_fluorescence(spikes, fs, T, noise_sd=0.0, gamma=1.0)
    f_sup, _ = fm.spikes_to_fluorescence(spikes, fs, T, noise_sd=0.0, gamma=1.5)
    burst_ratio = lambda x: x[int(2.0 * fs):int(3.0 * fs)].max() / x[int(8.0 * fs):int(9.0 * fs)].max()  # noqa: E731
    assert burst_ratio(f_sup) > burst_ratio(f_lin)
    g = fm.ar1_gamma(fs, fm.INDICATORS["GCaMP6f"].tau_decay_s)
    # deconvolve an AR(1)-generated trace exactly
    s_true = fm.bin_spikes(spikes, fs, T)
    y = np.zeros_like(s_true)
    for i in range(len(y)):
        y[i] = s_true[i] + (g * y[i - 1] if i > 0 else 0.0)
    s_hat = fm.deconvolve_ar1(y, g, chunk=256)
    assert np.allclose(s_hat, s_true, atol=1e-6)
    resp = fm.event_responses(f, fs, np.array([2.0, 8.0, 14.0, 17.0]), window=(0.0, 0.5))
    assert resp[0] > resp[1] > resp[3]
    best = fm.fit_forward_parameters(lambda traces, fs_: float(np.mean([tr.max() for tr in traces])), [spikes], fs, T,
                                     target_value=f_sup.max(), indicator=fm.INDICATORS["GCaMP6f"],
                                     gammas=(1.0, 1.5), c_sats=(np.inf,))
    assert best["gamma"] == 1.5


def test_reliability_and_disattenuation():
    rng = np.random.default_rng(2)
    dirs = np.tile(np.arange(0, 360, 45), 20)
    clean = tn.synthetic_tuned_responses(rng, dirs, 45.0, 10.0, 1.0, noise_sd=0.5)
    noisy = tn.synthetic_tuned_responses(rng, dirs, 45.0, 10.0, 1.0, noise_sd=8.0)
    r_clean = rl.split_half_reliability(clean, dirs, n_splits=30, rng=rng)["reliability"]
    r_noisy = rl.split_half_reliability(noisy, dirs, n_splits=30, rng=rng)["reliability"]
    assert r_clean > 0.9 > r_noisy
    assert rl.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert rl.disattenuated_correlation(0.4, 0.5, 0.5) == pytest.approx(0.8)
    assert rl.reliability_ceiling(0.64, 0.81) == pytest.approx(0.72)
    a = rng.normal(size=100)
    assert rl.test_retest_reliability(a, a + 0.1 * rng.normal(size=100)) > 0.9
    pt, lo, hi = rl.bootstrap_ci(np.median, a, n_boot=100, rng=rng)
    assert lo <= pt <= hi
    rn = rl.reliability_normalised_selectivity(np.array([0.5, 0.5]), np.array([0.25, 0.01]))
    assert rn[0] == pytest.approx(1.0) and np.isnan(rn[1])


def test_distribution_shift_reweighting_and_decomposition():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 2000)
    assert mv.distribution_shift(a, a)["wasserstein"] == pytest.approx(0.0)
    s = mv.distribution_shift(a, a + 1.0)
    assert 0.9 < s["median_diff"] < 1.1 and s["ks"] > 0.3
    # reweight a target sample so its covariates match the reference
    ref = pd.DataFrame({"depth": rng.normal(300, 50, 1000), "rate": rng.normal(2, 0.5, 1000)})
    tgt = pd.DataFrame({"depth": rng.normal(450, 80, 1500), "rate": rng.normal(4, 1.0, 1500)})
    w = mv.propensity_weights(ref, tgt)
    before = mv.covariate_balance(ref, tgt).abs().mean()
    after = mv.covariate_balance(ref, tgt, w).abs().mean()
    assert after < 0.5 * before
    assert 0 < mv.effective_sample_size(w) <= len(tgt)
    # sequential decomposition on an additive toy gap
    contributions = {"sampling": 0.3, "measurement": 0.5, "analysis": 0.1}
    gap_fn = lambda flags: 1.0 - sum(v for k, v in contributions.items() if flags[k])  # noqa: E731
    dec = mv.sequential_gap_decomposition(gap_fn, list(contributions))
    assert dec.set_index("term")["closed_mean"].to_dict() == pytest.approx(contributions)
    assert dec.attrs["raw_gap"] == 1.0 and dec.attrs["residual_gap"] == pytest.approx(0.1)
    chk = mv.invariance_check(a, a + 0.05)
    assert chk["invariant"] and not mv.invariance_check(a, a + 1.0)["invariant"]
    grid = mv.specification_grid(signal=["dff", "events", "deconv"], qc=["strict", "lenient"])
    assert len(grid) == 6
