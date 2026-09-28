"""Synthetic-data tests for motion_causal (run with PYTHONPATH=src)."""

import numpy as np
import pytest

from motion_causal import (
    artifact_sensitivity,
    bootstrap_ci,
    crossval_run_predictions,
    decompose_association,
    fisher_z_fc,
    framewise_displacement,
    motion_matched_permutation,
    negative_control_exposure,
    permutation_pvalue,
    robustness_value,
    simulate_motion_cohort,
    vectorize_upper,
)
from motion_causal.decomposition import e_value


def test_framewise_displacement_basic():
    P = np.zeros((4, 6))
    P[1, 0] = 1.0            # 1 mm translation
    P[2, 3] = np.rad2deg(0.02)  # rotation of 0.02 rad -> 1 mm on a 50 mm sphere
    fd = framewise_displacement(P, rotation_units="deg")
    assert fd[0] == 0.0
    assert fd[1] == pytest.approx(1.0)
    assert fd[2] == pytest.approx(2.0, abs=1e-6)  # 1 mm back in x + 1 mm rotation arc


def test_fc_and_vectorize():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(100, 5))
    ts[:, 1] = ts[:, 0] + 0.1 * rng.normal(size=100)
    Z = fisher_z_fc(ts)
    assert Z.shape == (5, 5) and Z[0, 0] == 0.0
    assert Z[0, 1] > 1.0
    v = vectorize_upper(Z)
    assert v.shape == (10,)


def test_artifact_sensitivity_fe_and_iv():
    rng = np.random.default_rng(1)
    n, R = 300, 4
    trait = rng.normal(size=n)
    z = np.tile(np.arange(R, dtype=float), (n, 1))
    m = trait[:, None] + 0.3 * z + 0.5 * rng.normal(size=(n, R))
    subj_eff = rng.normal(size=n)
    yhat = subj_eff[:, None] - 0.8 * m + 0.2 * rng.normal(size=(n, R))
    fe = artifact_sensitivity(yhat, m)
    assert fe.beta_m == pytest.approx(-0.8, abs=3 * fe.se + 0.05)
    iv = artifact_sensitivity(yhat, m, instrument_runs=z)
    assert iv.method == "2sls" and iv.first_stage_f > 10
    assert iv.beta_m == pytest.approx(-0.8, abs=3 * iv.se + 0.05)


def test_decomposition_recovers_artifact_component():
    coh = simulate_motion_cohort(n_subjects=400, artifact_strength=0.6, trait_confounding=0.5, seed=2)
    pred = crossval_run_predictions(coh.X_runs, coh.y, groups=coh.families, n_splits=5, alphas=(10.0, 100.0, 1000.0))
    assert np.isfinite(pred.yhat).all()
    r = np.corrcoef(pred.yhat, coh.y)[0, 1]
    assert r > 0.3
    sens = artifact_sensitivity(pred.yhat_runs, coh.motion_runs, instrument_runs=coh.instrument)
    assert sens.beta_m != 0
    m_subj = coh.motion_runs.mean(axis=1)
    dec = decompose_association(pred.yhat, coh.y, m_subj, sens.beta_m, control_motion=coh.control_motion)
    # ground truth: covariance of the prediction that would have been made from artifact-free FC
    yhat_clean_runs = np.empty_like(pred.yhat_runs)
    # use the same folds' average weight as an approximation of the model applied to clean FC
    w = pred.weights.mean(axis=0)
    yhat_obs = (coh.X_runs.mean(axis=1) @ w)
    yhat_cln = (coh.X_clean_runs.mean(axis=1) @ w)
    true_art = np.cov(yhat_obs, coh.y)[0, 1] - np.cov(yhat_cln, coh.y)[0, 1]
    assert np.sign(dec.cov_artifact) == np.sign(true_art)
    assert abs(dec.cov_artifact - true_art) < 0.5 * abs(true_art) + 0.05
    assert abs(dec.cov_artifact + dec.cov_nonartifact - dec.cov_total) < 1e-9
    assert abs(dec.cov_trait + dec.cov_clean - dec.cov_nonartifact) < 1e-9
    assert dec.shares["artifact"] > 0


def test_decomposition_zero_artifact():
    coh = simulate_motion_cohort(n_subjects=300, artifact_strength=0.0, trait_confounding=0.5, seed=3)
    pred = crossval_run_predictions(coh.X_runs, coh.y, groups=coh.families, n_splits=5, alphas=(100.0, 1000.0))
    sens = artifact_sensitivity(pred.yhat_runs, coh.motion_runs)
    assert abs(sens.beta_m) < 3 * sens.se + 0.02
    dec = decompose_association(pred.yhat, coh.y, coh.motion_runs.mean(1), sens.beta_m)
    assert abs(dec.shares["artifact"]) < 0.15


def test_negative_control_exposure():
    rng = np.random.default_rng(4)
    n = 500
    trait = rng.normal(size=n)
    primary = trait + 0.3 * rng.normal(size=n)
    control = trait + 0.3 * rng.normal(size=n)
    yhat = 0.5 * trait - 0.7 * primary + 0.2 * rng.normal(size=n)  # artifact path = -0.7
    res = negative_control_exposure(yhat, primary, control)
    assert res["artifact_path"] == pytest.approx(-0.7, abs=3 * res["se_artifact_path"] + 0.05)


def test_nulls_and_sensitivity_helpers():
    rng = np.random.default_rng(5)
    motion = rng.normal(size=400)
    y = 0.6 * motion + rng.normal(size=400)
    yp = motion_matched_permutation(y, motion, n_bins=10, rng=rng)
    assert np.corrcoef(yp, motion)[0, 1] > 0.3  # motion relation preserved
    assert not np.allclose(yp, y)
    assert permutation_pvalue(2.0, np.zeros(99)) == pytest.approx(0.01)
    lo, hi, _ = bootstrap_ci(lambda a, b: float(np.corrcoef(a, b)[0, 1]), y, motion, n_boot=200, rng=rng)
    assert lo < np.corrcoef(y, motion)[0, 1] < hi
    rv = robustness_value(t_stat=4.0, dof=400)
    assert 0 < rv < 1
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2))
