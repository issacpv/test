"""Synthetic-data tests for ecg_probe."""
from __future__ import annotations

import numpy as np

from ecg_probe import embeddings, probing, erasure, fairness, stats


def _synthetic_ecgs(n=60, seed=0):
    """Generate ECG-like arrays whose amplitude depends on a binary attribute."""
    rng = np.random.default_rng(seed)
    X, attr = [], []
    t = np.linspace(0, 10, embeddings.N_SAMPLES)
    for i in range(n):
        a = i % 2
        amp = 1.0 + 0.6 * a  # attribute shifts amplitude -> decodable
        sig = amp * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.1, t.size)
        X.append(np.stack([sig * (1 + 0.02 * k) for k in range(embeddings.N_LEADS)]))
        attr.append(a)
    return np.array(X), np.array(attr)


def test_mock_encoder_shapes():
    enc = embeddings.get_encoder("mock", n_layers=3, dim=32)
    x = np.random.default_rng(0).normal(size=(12, embeddings.N_SAMPLES))
    layers = enc.encode(x)
    assert "final" in layers and "layer0" in layers
    assert layers["final"].shape == (32,)


def test_extract_and_probe_decodes_attribute():
    X, attr = _synthetic_ecgs(80)
    enc = embeddings.get_encoder("mock", n_layers=3, dim=64)
    Z = embeddings.extract_matrix(enc, X, layer="final")
    groups = np.arange(len(attr))  # each sample its own group here
    auc = probing.linear_probe_classification(Z, attr, groups=groups)
    assert auc > 0.75  # amplitude-linked attribute is decodable


def test_selectivity_positive_for_real_task():
    X, attr = _synthetic_ecgs(80)
    enc = embeddings.get_encoder("mock", dim=64)
    Z = embeddings.extract_matrix(enc, X)
    res = probing.probe_with_selectivity(Z, attr)
    assert res.selectivity > 0.1  # real task decodes better than control


def test_mdl_lower_for_decodable():
    X, attr = _synthetic_ecgs(120)
    enc = embeddings.get_encoder("mock", dim=64)
    Z = embeddings.extract_matrix(enc, X)
    rng = np.random.default_rng(1)
    code_real = probing.mdl_online_code(Z, attr, seed=0)
    code_rand = probing.mdl_online_code(Z, rng.permutation(attr), seed=0)
    assert code_real < code_rand  # decodable attribute has shorter codelength


def test_leace_erases_linear_predictability():
    X, attr = _synthetic_ecgs(120)
    enc = embeddings.get_encoder("mock", dim=64)
    Z = embeddings.extract_matrix(enc, X)
    auc_before = probing.linear_probe_classification(Z, attr, groups=np.arange(len(attr)))
    erase = erasure.leace_fit(Z, attr)
    Ze = erase(Z)
    auc_after = probing.linear_probe_classification(Ze, attr, groups=np.arange(len(attr)))
    assert auc_before > 0.75
    assert auc_after < 0.65  # erasure collapses decodability toward chance


def test_inlp_reduces_predictability():
    X, attr = _synthetic_ecgs(120)
    enc = embeddings.get_encoder("mock", dim=48)
    Z = embeddings.extract_matrix(enc, X)
    erase = erasure.inlp_fit(Z, attr, n_iter=8)
    Ze = erase(Z)
    auc_after = probing.linear_probe_classification(Ze, attr, groups=np.arange(len(attr)))
    assert auc_after < 0.7


def test_subgroup_gap_and_link():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = y * 0.5 + rng.normal(0, 1, 200)
    group = rng.integers(0, 2, 200)
    out = fairness.subgroup_auroc_gap(y, scores, group)
    assert 0 <= out["gap"] <= 1
    leak = {"A": 0.9, "B": 0.6, "C": 0.7, "D": 0.55}
    gaps = {"A": 0.20, "B": 0.05, "C": 0.10, "D": 0.03}
    corr = fairness.leakage_gap_correlation(leak, gaps)
    assert corr["rho"] > 0


def test_delong_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y * 1.5 + rng.normal(0, 1, 300)
    bad = rng.normal(0, 1, 300)
    d = stats.delong_test(y, good, bad)
    assert d["auc_a"] > d["auc_b"]
    assert d["p"] < 0.05
    mean, lo, hi = stats.bootstrap_ci(np.array([0.8, 0.82, 0.79, 0.85, 0.81]))
    assert lo <= mean <= hi
