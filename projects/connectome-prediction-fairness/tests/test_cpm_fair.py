"""Synthetic-data tests for cpm_fair core functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpm_fair import fc_loader, mediation, predictors, reweighting, subgroup_metrics  # noqa: E402


def _synthetic_cohort(n=400, n_nodes=20, n_signal=15, minority_frac=0.15, seed=0):
    """FC-like features with a planted linear signal, families of size 1-3,
    a majority/minority label and group-dependent motion."""
    rng = np.random.default_rng(seed)
    n_edges = n_nodes * (n_nodes - 1) // 2
    X = rng.standard_normal((n, n_edges))
    w = np.zeros(n_edges)
    w[:n_signal] = 1.0
    y = X @ w + rng.standard_normal(n) * 2.0
    groups = np.where(rng.random(n) < minority_frac, "B", "A")
    # families: consecutive subjects share a family with prob 0.5
    fam = np.zeros(n, dtype=int)
    f = 0
    for i in range(n):
        if i > 0 and rng.random() < 0.5 and fam[i - 1] == fam[i - 2 if i > 1 else i - 1]:
            fam[i] = fam[i - 1]
        else:
            f += 1
            fam[i] = f
    # families share group label
    fam_label = {}
    for i in range(n):
        fam_label.setdefault(fam[i], groups[i])
        groups[i] = fam_label[fam[i]]
    motion = 0.1 + 0.05 * (groups == "B") + 0.02 * rng.standard_normal(n)
    return X, y, groups, fam, motion


def test_fc_roundtrip():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((100, 8))
    fc = fc_loader.timeseries_to_fc(ts)
    assert fc.shape == (8, 8)
    assert np.allclose(np.diag(fc), 0)
    v = fc_loader.vectorize_upper(fc)
    assert v.shape[0] == 28
    back = fc_loader.unvectorize_upper(v, 8)
    assert np.allclose(back, fc)


def test_define_subgroups_collapses_rare():
    df = pd.DataFrame({"Race": ["White"] * 50 + ["Black or African Am."] * 40 + ["Asian"] * 3,
                       "Ethnicity": ["Not Hispanic/Latino"] * 90 + ["Hispanic/Latino"] * 3})
    g = fc_loader.define_subgroups(df, min_n=10)
    assert set(g.cat.categories) == {"White", "Black or African Am.", "Other/Multiple"}
    # Hispanic overrides race then gets collapsed as rare
    assert (g == "Other/Multiple").sum() == 3


def test_family_kfold_never_splits_families():
    X, y, groups, fam, _ = _synthetic_cohort()
    cv = predictors.FamilyKFold(n_splits=5, random_state=1)
    seen = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y, families=fam, stratify=groups):
        assert predictors.check_no_family_leak(tr, te, fam)
        seen[te] += 1
        # stratification keeps minority share roughly balanced
        assert (groups[te] == "B").mean() > 0.02
    assert np.all(seen == 1)


def test_cpm_and_ridge_recover_signal():
    X, y, groups, fam, _ = _synthetic_cohort(n=500)
    for est in (predictors.CPMRegressor(p_threshold=0.01), predictors.RidgeFC()):
        preds = predictors.cross_val_predict_family(est, X, y, fam, n_splits=5, stratify=groups)
        r = np.corrcoef(preds["y"], preds["y_hat"])[0, 1]
        assert r > 0.5, f"{type(est).__name__} r={r:.2f}"


def test_residualize_removes_confound():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((200, 1))
    X = 3 * C + rng.standard_normal((200, 4)) * 0.1
    Xr, Xt = predictors.residualize(X[:150], C[:150], X[150:], C[150:])
    assert abs(np.corrcoef(Xr[:, 0], C[:150, 0])[0, 1]) < 0.05
    assert Xt.shape == (50, 4)


def test_subgroup_metrics_and_gap_tests():
    X, y, groups, fam, motion = _synthetic_cohort(n=500)
    preds = predictors.cross_val_predict_family(predictors.RidgeFC(), X, y, fam, stratify=groups)
    perf = subgroup_metrics.subgroup_performance(preds["y"], preds["y_hat"], groups)
    assert {"A", "B", "ALL"} <= set(perf.index)
    assert perf.loc["ALL", "n"] == 500
    es = subgroup_metrics.error_structure(preds["y"], preds["y_hat"], groups, reference="A")
    assert "levene_p" in es.columns
    boot = subgroup_metrics.cluster_bootstrap_gap(preds["y"], preds["y_hat"], groups, fam, "A", n_boot=100)
    assert boot.loc["A", "gap"] == 0
    assert boot.loc["B", "ci_low"] <= boot.loc["B", "gap"] <= boot.loc["B", "ci_high"]
    perm = subgroup_metrics.permutation_gap_test(preds["y"], preds["y_hat"], groups, fam, "A", n_perm=100)
    assert 0 < perm.loc["B", "p_perm"] <= 1
    matched = subgroup_metrics.accuracy_at_matched_n(preds["y"], preds["y_hat"], groups, n=40, n_draws=20)
    assert np.isfinite(matched.loc["B", "mean"])


def test_group_dro_reduces_worst_group_loss():
    rng = np.random.default_rng(1)
    p = 10
    nA, nB = 400, 60
    XA = rng.standard_normal((nA, p))
    XB = rng.standard_normal((nB, p))
    # opposite brain-behaviour mapping in the minority group
    yA = XA[:, 0] + 0.3 * rng.standard_normal(nA)
    yB = -XB[:, 0] + 0.3 * rng.standard_normal(nB)
    X = np.vstack([XA, XB])
    y = np.concatenate([yA, yB])
    groups = np.array(["A"] * nA + ["B"] * nB)
    erm = reweighting.ERMRidge(alpha=1.0).fit(X, y, groups=groups)
    dro = reweighting.GroupDRORidge(alpha=1.0, n_iter=100, eta=0.5).fit(X, y, groups=groups)

    def worst(est):
        r = y - est.predict(X)
        return max(np.mean(r[groups == g] ** 2) for g in ("A", "B"))

    assert worst(dro) < worst(erm)
    assert dro.group_weights_["B"] > dro.group_weights_["A"]
    w = reweighting.inverse_frequency_weights(groups)
    assert np.isclose(w.mean(), 1.0)
    assert w[groups == "B"][0] > w[groups == "A"][0]
    idx = reweighting.balanced_subsample(groups, seed=0)
    assert (groups[idx] == "A").sum() == (groups[idx] == "B").sum() == nB


def test_evaluate_training_schemes_runs():
    X, y, groups, fam, _ = _synthetic_cohort(n=300, n_nodes=10)
    res = reweighting.evaluate_training_schemes(X, y, groups, fam, alpha=10.0, n_splits=3)
    assert set(res) == {"erm", "inverse_frequency", "balanced_subsample", "group_dro"}
    for tab in res.values():
        assert "B" in tab.index


def test_mediation_recovers_indirect_effect():
    rng = np.random.default_rng(0)
    n = 2000
    g = (rng.random(n) < 0.3).astype(float)
    motion = 1.0 * g + rng.standard_normal(n)
    err = 0.5 * motion + 0.2 * g + rng.standard_normal(n)
    res = mediation.mediation_by_motion(g, motion, err, n_boot=300)
    assert abs(res["indirect"] - 0.5) < 0.15
    assert res["ci_low"] < 0.5 < res["ci_high"]
    assert 0 < res["prop_mediated"] < 1


def test_motion_matching_and_decomposition():
    X, y, groups, fam, motion = _synthetic_cohort(n=500)
    t_idx, r_idx = mediation.motion_matched_subsample(groups, motion, "A", "B", caliper=0.05)
    assert len(t_idx) == len(r_idx) > 0
    assert len(set(r_idx)) == len(r_idx)
    preds = predictors.cross_val_predict_family(predictors.RidgeFC(), X, y, fam, stratify=groups)
    tab = mediation.decompose_gap(preds["y"], preds["y_hat"], groups, motion, "A", "B", n_draws=10)
    assert set(tab["component"]) == {"total_gap", "sample_size", "motion", "label_reliability", "residual"}
    comps = tab.set_index("component")["value"]
    assert np.isclose(comps["total_gap"], comps[["sample_size", "motion", "label_reliability", "residual"]].sum())
    assert mediation.attenuation_ceiling(0.8, 0.5) == pytest.approx(np.sqrt(0.4))
