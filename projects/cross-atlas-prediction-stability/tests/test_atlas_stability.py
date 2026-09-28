"""Synthetic-data tests for atlas_stability (run with PYTHONPATH=src)."""

import numpy as np
import pytest

from atlas_stability import (
    accuracy_dispersion,
    aggregate_nodes_to_networks,
    atlas_robustness_index,
    cpm_cv_predict,
    edge_weights_to_node_strength,
    finding_concordance,
    fisher_z_fc,
    majority_network_assignment,
    node_to_space,
    parcellate_timeseries,
    posthoc_selection_inflation,
    prediction_accuracy,
    prediction_concordance,
    random_contiguous_parcellation,
    ridge_cv_predict,
    simulate_vertex_dataset,
    vectorize_upper,
)


def _features(ds, labels):
    n = ds.ts.shape[0]
    X = []
    for i in range(n):
        pts = parcellate_timeseries(ds.ts[i], labels)
        X.append(vectorize_upper(fisher_z_fc(pts)))
    return np.array(X)


def test_parcellate_and_projection():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(50, 10))
    labels = np.array([1, 1, 2, 2, 2, 0, 3, 3, 3, 3])
    pts = parcellate_timeseries(ts, labels)
    assert pts.shape == (50, 3)
    assert np.allclose(pts[:, 0], ts[:, :2].mean(1))
    fc = fisher_z_fc(pts)
    assert fc.shape == (3, 3) and np.allclose(np.diag(fc), 0)
    v = vectorize_upper(fc)
    assert v.shape == (3,)
    strength = edge_weights_to_node_strength(np.array([1.0, -2.0, 3.0]), 3)
    assert np.allclose(strength, [3.0, 4.0, 5.0])
    m = node_to_space(np.array([10.0, 20.0, 30.0]), labels)
    assert np.isnan(m[5]) and m[0] == 10.0 and m[9] == 30.0
    nets = majority_network_assignment(labels, np.array([1, 1, 2, 2, 1, 0, 0, 3, 3, 3]))
    assert list(nets) == [1, 2, 3]
    assert np.allclose(aggregate_nodes_to_networks(np.array([1.0, 2.0, 3.0]), nets, 3), [1.0, 2.0, 3.0])
    lab = random_contiguous_parcellation(100, 7, rng)
    assert lab.min() == 1 and lab.max() == 7 and np.all(np.diff(lab) >= 0)


def test_true_atlas_beats_coarse_random_atlas():
    ds = simulate_vertex_dataset(n_subjects=120, n_vertices=200, n_regions=10, seed=1)
    X_true = _features(ds, ds.true_labels)
    res_true = ridge_cv_predict(X_true, ds.y, groups=ds.families, n_splits=5, alphas=(1.0, 10.0, 100.0))
    acc_true = prediction_accuracy(ds.y, res_true.yhat)["r"]
    # a coarse atlas that merges the two signal regions with others
    coarse = np.where(ds.true_labels <= 5, 1, 2)
    X_coarse = _features(ds, coarse)
    res_coarse = ridge_cv_predict(X_coarse, ds.y, groups=ds.families, n_splits=5, alphas=(1.0, 10.0, 100.0))
    acc_coarse = prediction_accuracy(ds.y, res_coarse.yhat)["r"]
    assert acc_true > 0.5
    assert acc_true > acc_coarse + 0.2
    cpm = cpm_cv_predict(X_true, ds.y, groups=ds.families, n_splits=5, p_threshold=0.05)
    assert prediction_accuracy(ds.y, cpm.yhat)["r"] > 0.3
    # the predictive pattern loads on the signal edge (regions 1-2 -> first edge of the upper triangle)
    pat = np.abs(res_true.patterns.mean(0))
    assert np.argmax(pat) == 0


def test_robustness_metrics_and_ari():
    ds = simulate_vertex_dataset(n_subjects=120, n_vertices=200, n_regions=10, seed=2)
    rng = np.random.default_rng(3)
    atlases = {"true": ds.true_labels}
    # fine atlases that split true regions (keep boundaries) vs a random atlas ignoring them
    fine = np.repeat(np.arange(1, 21), 10)
    atlases["fine"] = fine
    atlases["random"] = random_contiguous_parcellation(200, 10, rng)
    accs, preds, node_maps, labels = {}, {}, {}, {}
    for name, lab in atlases.items():
        X = _features(ds, lab)
        res = ridge_cv_predict(X, ds.y, groups=ds.families, n_splits=5, alphas=(1.0, 10.0, 100.0))
        accs[name] = prediction_accuracy(ds.y, res.yhat)["r"]
        preds[name] = res.yhat
        n_nodes = int(lab.max())
        node_maps[name] = edge_weights_to_node_strength(res.patterns.mean(0), n_nodes)
        labels[name] = lab
    disp = accuracy_dispersion(accs)
    assert 0 <= disp["normalised_range"] <= 1
    pc, pairs = prediction_concordance(preds)
    assert pairs[("fine", "true")] > pairs[("random", "true")] - 1e-9 or accs["random"] > 0.5
    fc = finding_concordance(node_maps, labels, n_perm=20, rng=rng)
    assert np.isfinite(fc["mean_r"]) and 0 <= fc["score"] <= 1
    ari = atlas_robustness_index(accs, preds, fc["score"])
    assert 0 <= ari["ari"] <= 1
    # robust subset (true + fine) should score higher than the full set including the random atlas
    ari_sub = atlas_robustness_index({k: accs[k] for k in ("true", "fine")}, {k: preds[k] for k in ("true", "fine")}, fc["score"])
    assert ari_sub["ari"] >= ari["ari"] - 0.05


def test_posthoc_selection_inflation():
    rng = np.random.default_rng(4)
    acc = 0.3 + 0.1 * rng.normal(size=(10, 8))  # no atlas is truly better
    out = posthoc_selection_inflation(acc, ensemble_acc=np.full(10, 0.32))
    assert out["best_of"] > out["nested"]
    assert out["inflation"] > 0.05
    assert "ensemble_minus_nested" in out
