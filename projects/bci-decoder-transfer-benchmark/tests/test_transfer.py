"""Synthetic-data tests for SPD geometry, alignment, the calibration protocol and the statistics."""
import numpy as np
import pytest

from bci_transfer.covariance import (
    MDM,
    airm_distance,
    balanced_accuracy,
    covariances,
    euclidean_alignment,
    expm_spd,
    geodesic,
    invsqrtm_spd,
    logm_spd,
    riemannian_alignment,
    riemannian_mean,
    sqrtm_spd,
    tangent_space,
)
from bci_transfer.learning_curves import (
    auc_learning_curve,
    calibration_curve,
    calibration_savings,
    fit_learning_curve,
    stratified_draw,
    trials_to_fraction,
)
from bci_transfer.mixed_effects import (
    fit_transfer_mixed_model,
    holm,
    meta_regression,
    negative_transfer_rate,
    paired_effect_size,
)
from bci_transfer.synthetic import make_multi_subject_dataset, make_results_table, make_subject


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def test_spd_functions_roundtrip():
    C = _spd(5, 0)
    assert np.allclose(sqrtm_spd(C) @ sqrtm_spd(C), C)
    assert np.allclose(invsqrtm_spd(C) @ C @ invsqrtm_spd(C), np.eye(5), atol=1e-8)
    assert np.allclose(expm_spd(logm_spd(C)), C)
    assert airm_distance(C, C) == pytest.approx(0.0, abs=1e-8)
    B = _spd(5, 1)
    assert airm_distance(C, B) == pytest.approx(airm_distance(B, C))


def test_riemannian_mean_is_geodesic_midpoint():
    A, B = _spd(4, 2), _spd(4, 3)
    M = riemannian_mean(np.stack([A, B]))
    assert np.allclose(M, geodesic(A, B, 0.5), atol=1e-6)
    assert airm_distance(M, A) == pytest.approx(airm_distance(M, B), rel=1e-5)


def test_tangent_space_norm_equals_distance():
    ref = _spd(4, 4)
    C = _spd(4, 5)
    v = tangent_space(C[None], ref)
    assert v.shape == (1, 10)
    assert np.linalg.norm(v[0]) == pytest.approx(airm_distance(ref, C), rel=1e-6)


def test_alignment_recentres_to_identity():
    sub = make_subject(n_trials=60, n_channels=6, n_times=200, shift=0.8, seed=1)
    C = covariances(sub["X"], shrinkage=0.0)
    Ca, M = riemannian_alignment(C)
    assert np.allclose(riemannian_mean(Ca), np.eye(6), atol=1e-4)
    Xa, R = euclidean_alignment(sub["X"], shrinkage=0.0)
    assert np.allclose(covariances(Xa, 0.0).mean(axis=0), np.eye(6), atol=1e-6)


def test_riemannian_alignment_enables_cross_subject_transfer():
    subs = make_multi_subject_dataset(n_subjects=5, n_trials=80, n_channels=6, n_times=200, shift=1.0, class_sep=1.5, seed=3)
    covs = [covariances(s["X"], shrinkage=0.05) for s in subs]
    target = 4
    src_raw = np.concatenate([covs[i] for i in range(4)])
    y_src = np.concatenate([subs[i]["y"] for i in range(4)])
    acc_raw = balanced_accuracy(subs[target]["y"], MDM().fit(src_raw, y_src).predict(covs[target]))
    src_al = np.concatenate([riemannian_alignment(covs[i])[0] for i in range(4)])
    tgt_al = riemannian_alignment(covs[target])[0]
    acc_al = balanced_accuracy(subs[target]["y"], MDM().fit(src_al, y_src).predict(tgt_al))
    assert acc_al >= 0.85
    assert acc_al > acc_raw


def test_stratified_draw_and_calibration_curve():
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 20)
    idx = stratified_draw(y, 10, rng)
    assert len(idx) == 10 and np.sum(y[idx] == 0) == 5
    assert stratified_draw(y, 0, rng).size == 0

    sub = make_subject(n_trials=120, n_channels=4, n_times=150, shift=0.3, seed=5)
    C = covariances(sub["X"], 0.05)
    cal, test = np.arange(60), np.arange(60, 120)
    source = make_subject(n_trials=100, n_channels=4, n_times=150, shift=0.3, seed=6)
    src_c = riemannian_alignment(covariances(source["X"], 0.05))[0]

    def fit_predict(X_cal, y_cal, X_test, src):
        if len(y_cal) == 0:
            return MDM().fit(src[0], src[1]).predict(X_test)
        return MDM().fit(np.concatenate([src[0], X_cal]), np.concatenate([src[1], y_cal])).predict(X_test)

    Ca_all = riemannian_alignment(C)[0]
    df = calibration_curve(fit_predict, Ca_all[cal], sub["y"][cal], Ca_all[test], sub["y"][test],
                           budgets=(0, 4, 8, 16, 60), n_draws=3, source=(src_c, source["y"]), seed=0)
    assert set(df["budget"]) == {0, 4, 8, 16, 60}
    assert (df["accuracy"] >= 0.5).mean() > 0.8
    assert df[df["budget"] == 0].shape[0] == 1  # zero budget is deterministic -> one draw


def test_learning_curve_fit_and_estimands():
    k = np.array([0, 5, 10, 20, 40, 80, 160])
    true = {"a": 0.9, "b": 0.35, "c": 0.7}
    acc = true["a"] - true["b"] * (k + 1.0) ** (-true["c"])
    fit = fit_learning_curve(k, acc)
    assert fit["a"] == pytest.approx(0.9, abs=0.01) and fit["c"] == pytest.approx(0.7, abs=0.05)
    k90 = trials_to_fraction(true, 0.9, chance=0.5)
    assert acc_at(true, k90) == pytest.approx(0.5 + 0.9 * 0.4, rel=1e-6)
    transfer = {"a": 0.9, "b": 0.15, "c": 0.7}
    sav = calibration_savings(transfer, true, 0.9, 0.5)
    assert sav["k_transfer"] < sav["k_scratch"] and sav["saved"] > 0 and 0 < sav["ratio"] < 1
    assert trials_to_fraction({"a": 0.4, "b": 0.1, "c": 0.5}, 0.9, 0.5) == float("inf")
    assert auc_learning_curve(k, acc, 0.5) > 0


def acc_at(p, k):
    return p["a"] - p["b"] * (k + 1.0) ** (-p["c"])


def test_statistics_on_synthetic_results():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    df = make_results_table(n_datasets=4, subjects_per_dataset=8, transfer_gain=0.08, seed=1)
    wide = df[df["budget"] == 10].pivot_table(index=["dataset", "subject"], columns="method", values="accuracy")
    es = paired_effect_size(wide["ra_mdm"], wide["scratch"])
    assert es["mean_diff"] > 0 and es["p_wilcoxon"] < 0.01 and es["d_z"] > 0.5
    ntr = negative_transfer_rate(wide["ra_mdm"], wide["scratch"])
    assert 0 <= ntr["rate"] <= 0.3 and ntr["ci_low"] <= ntr["rate"] <= ntr["ci_high"]
    res = fit_transfer_mixed_model(df)
    coef = res.params
    name = [c for c in coef.index if "ra_mdm" in c and ":" not in c][0]
    assert coef[name] > 0

    # meta-regression: gains rise with a moderator
    rng = np.random.default_rng(0)
    mod = np.linspace(0, 1, 12)
    effects = 0.02 + 0.1 * mod + rng.normal(0, 0.01, 12)
    var = np.full(12, 0.01 ** 2)
    mr = meta_regression(effects, var, X=mod[:, None], names=["channels"])
    assert mr["coef"]["channels"] > 0.05 and mr["p"]["channels"] < 0.01
    assert 0 <= mr["I2"] <= 1
