"""Synthetic-data tests for dmri_charts (run with PYTHONPATH=src)."""

import numpy as np
import pandas as pd
import pytest

from dmri_charts import (
    NormativeModel,
    age_explained_variance,
    age_of_peak,
    bootstrap_age_of_peak,
    deviation_auc,
    fibonacci_sphere,
    fit_dki,
    fit_dti,
    icc_2_1,
    identify_shells,
    protocol_summary,
    protocol_transfer_bias,
    rank_models,
    simulate_lifespan_dataset,
    simulate_multicompartment_signal,
    subsample_protocol,
)


def _hcp_like_table(n_dirs=60, n_b0=6, seed=0):
    rng = np.random.default_rng(seed)
    dirs = fibonacci_sphere(n_dirs)
    bvals = np.concatenate([np.zeros(n_b0), np.full(n_dirs, 1000.0), np.full(n_dirs, 2000.0), np.full(n_dirs, 3000.0)])
    bvecs = np.vstack([np.zeros((n_b0, 3)), dirs, dirs @ _rot(rng), dirs @ _rot(rng)])
    return bvals + rng.normal(0, 5, len(bvals)) * (bvals > 0), bvecs


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q


def test_shells_and_subsampling():
    bvals, bvecs = _hcp_like_table()
    sh = identify_shells(bvals)
    assert set(sh) == {0, 1000, 2000, 3000}
    assert len(sh[1000]) == 60
    idx = subsample_protocol(bvals, bvecs, {1000: 30, 2000: 60}, n_b0=3)
    sub = identify_shells(bvals[idx])
    assert set(sub) == {0, 1000, 2000} and len(sub[0]) == 3 and len(sub[1000]) == 30 and len(sub[2000]) == 60
    s = protocol_summary(bvals[idx])
    assert s["multishell"] == 1 and s["max_b"] == 2000
    with pytest.raises(KeyError):
        subsample_protocol(bvals, bvecs, {1500: 10})


def test_dti_recovers_gaussian_tensor():
    bvals, bvecs = _hcp_like_table()
    keep = bvals < 1200  # DTI regime
    S = simulate_multicompartment_signal(bvals[keep], bvecs[keep], f_stick=0.0, f_zeppelin=1.0, f_ball=0.0,
                                         d_par=1.7e-3, d_perp=0.4e-3, orientation=(0, 0, 1))
    res = fit_dti(S[None, :], bvals[keep], bvecs[keep])
    assert res.AD[0] == pytest.approx(1.7e-3, rel=0.02)
    assert res.RD[0] == pytest.approx(0.4e-3, rel=0.05)
    fa_true = np.sqrt(0.5) * np.sqrt(2 * (1.7e-3 - 0.4e-3) ** 2) / np.sqrt(1.7e-3 ** 2 + 2 * 0.4e-3 ** 2)
    assert res.FA[0] == pytest.approx(fa_true, rel=0.03)
    assert abs(abs(res.evecs[0][:, 0] @ np.array([0, 0, 1.0])) - 1) < 0.02


def test_dki_kurtosis_sign():
    bvals, bvecs = _hcp_like_table()
    gauss = simulate_multicompartment_signal(bvals, bvecs, f_stick=0.0, f_zeppelin=1.0, f_ball=0.0)
    mixed = simulate_multicompartment_signal(bvals, bvecs, f_stick=0.5, f_zeppelin=0.35, f_ball=0.15)
    res = fit_dki(np.vstack([gauss, mixed]), bvals, bvecs)
    assert abs(res.MK[0]) < 0.05           # Gaussian tensor: no kurtosis
    assert res.MK[1] > 0.3                 # multi-compartment: positive kurtosis
    assert res.RK[1] > res.AK[1]           # sticks + zeppelins: radial kurtosis dominates
    noisy = simulate_multicompartment_signal(bvals, bvecs, snr=40, rng=np.random.default_rng(1))
    res_n = fit_dki(noisy[None, :], bvals, bvecs)
    assert np.isfinite(res_n.MK[0]) and 0.0 < res_n.FA[0] < 1.0


def test_normative_model_calibration_and_peak():
    df = simulate_lifespan_dataset(n=3000, peak_age=30.0, seed=1)
    model = NormativeModel(n_knots=6).fit(df.age, df.y, df.sex, df.site)
    z = model.zscore(df.age, df.y, df.sex, df.site)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.08
    # no age trend in z or z^2 (variance modelled)
    young, old = df.age < 30, df.age > 70
    assert abs(z[young].std() - z[old].std()) < 0.15
    peak = age_of_peak(model, sex=0)
    assert abs(peak - 30.0) < 6
    cent = model.centiles(np.array([20.0, 60.0]), sex=np.zeros(2), site=np.zeros(2, int))
    assert (cent["c95"] > cent["c50"]).all() and (cent["c50"] > cent["c05"]).all()
    point, (lo, hi), draws = bootstrap_age_of_peak(df.age, df.y, df.sex, df.site, n_boot=20, n_knots=6)
    assert lo <= point <= hi and len(draws) == 20
    metrics = age_explained_variance(model, df.age, df.y, df.sex, df.site)
    assert 0.3 < metrics["r2_age"] < 1.0
    assert metrics["slope_late_per_decade_sd"] < 0  # inverted U: declining after 40


def test_compare_metrics_and_leaderboard():
    rng = np.random.default_rng(2)
    z1 = rng.normal(size=45)
    z2 = 0.9 * z1 + 0.3 * rng.normal(size=45)
    icc = icc_2_1(z1, z2)
    assert 0.6 < icc < 1.0
    bias = protocol_transfer_bias(z1, z1 + 0.4)
    assert bias["mean_shift"] == pytest.approx(0.4, abs=1e-9) and bias["corr"] > 0.999
    auc = deviation_auc(rng.normal(size=200), rng.normal(size=100) - 1.0)
    assert auc["auc"] > 0.7
    table = pd.DataFrame({"r2_age": [0.5, 0.3], "icc": [0.7, 0.9], "cohen_d": [0.5, 0.1]}, index=["NODDI", "DTI"])
    lb = rank_models(table)
    assert "rank_sum" in lb.columns and set(lb.index) == {"NODDI", "DTI"}
