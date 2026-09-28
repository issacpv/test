"""Synthetic-data tests for conn_h2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conn_h2 import features, gradients, heritability, spatial_nulls  # noqa: E402


def _block_fc(n_per_block=20, n_blocks=3, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    labels = np.repeat(np.arange(n_blocks), n_per_block)
    fc = np.where(labels[:, None] == labels[None, :], 0.8, 0.1) + noise * rng.standard_normal((n, n))
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 0)
    return fc, labels


def test_diffusion_map_recovers_block_structure():
    fc, labels = _block_fc()
    grads, lambdas = gradients.compute_gradients(fc, n_components=5, sparsity=0.5)
    assert grads.shape == (60, 5)
    assert np.all(np.diff(lambdas) <= 1e-10)
    # first two gradients separate the three blocks: between-block variance dominates
    g = grads[:, :2]
    within = sum(g[labels == k].var(axis=0).sum() for k in range(3)) / 3
    between = g.var(axis=0).sum()
    assert between > 5 * within


def test_procrustes_alignment_recovers_rotation():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((40, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    src = ref @ Q
    aligned = gradients.procrustes_align(src, ref)
    assert np.allclose(aligned, ref, atol=1e-8)
    al = gradients.align_group([ref, src, -src], n_iter=5)
    assert np.allclose(al[1], ref, atol=1e-6)


def test_subject_gradient_features_and_dispersion():
    fcs = [_block_fc(seed=s, noise=0.15)[0] for s in range(4)]
    loads, ecc = gradients.subject_gradient_features(fcs, n_components=3, sparsity=0.5)
    assert loads.shape == (4, 60, 3)
    assert ecc.shape == (4, 60)
    assert np.all(ecc >= 0)
    # aligned subjects should correlate strongly on gradient 1
    assert abs(np.corrcoef(loads[0, :, 0], loads[1, :, 0])[0, 1]) > 0.8


def test_sc_fc_coupling_and_multilinear():
    rng = np.random.default_rng(0)
    n = 30
    sc = rng.gamma(2.0, 2.0, size=(n, n))
    sc = (sc + sc.T) / 2
    sc[rng.random((n, n)) < 0.3] = 0
    sc = np.triu(sc, 1)
    sc = sc + sc.T
    fc = np.log1p(sc) + 0.3 * rng.standard_normal((n, n))
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 0)
    coup = features.sc_fc_coupling(sc, fc)
    assert coup.shape == (n,)
    assert np.nanmean(coup) > 0.5
    r2 = features.multilinear_coupling(sc, fc)
    assert np.nanmean(r2) > 0.3
    assert np.nanmax(r2) <= 1.0


def test_dynamic_states():
    rng = np.random.default_rng(0)
    ts = [rng.standard_normal((300, 8)) for _ in range(3)]
    fo, dwell, trans = features.dynamic_state_features(ts, window=50, step=25, n_states=3)
    assert fo.shape == (3, 3) and dwell.shape == (3, 3) and trans.shape == (3, 9)
    assert np.allclose(fo.sum(axis=1), 1.0)


def _simulate_twins(n_mz=300, n_dz=300, h2=0.6, c2=0.1, seed=0):
    rng = np.random.default_rng(seed)
    e2 = 1 - h2 - c2
    A_mz = rng.standard_normal(n_mz)
    C_mz = rng.standard_normal(n_mz)
    mz = np.column_stack([np.sqrt(h2) * A_mz + np.sqrt(c2) * C_mz + np.sqrt(e2) * rng.standard_normal(n_mz) for _ in range(2)])
    A_shared = rng.standard_normal(n_dz)
    C_dz = rng.standard_normal(n_dz)
    dz = np.column_stack([
        np.sqrt(h2) * (np.sqrt(0.5) * A_shared + np.sqrt(0.5) * rng.standard_normal(n_dz)) + np.sqrt(c2) * C_dz + np.sqrt(e2) * rng.standard_normal(n_dz)
        for _ in range(2)
    ])
    return mz, dz


def test_heritability_estimators_recover_simulated_h2():
    mz, dz = _simulate_twins(n_mz=600, n_dz=600, h2=0.6, c2=0.1, seed=3)
    fal = heritability.falconer_h2(mz, dz)
    assert abs(fal["h2"] - 0.6) < 0.15
    df = heritability.defries_fulker(mz, dz)
    assert abs(df["h2"] - 0.6) < 0.15
    ace = heritability.ace_ml(mz, dz)
    assert abs(ace.loc["ACE", "a2"] - 0.6) < 0.15
    assert ace.loc["E", "lrt_p"] < 1e-6  # genetic effects clearly needed
    ci = heritability.bootstrap_ci(mz, dz, heritability.falconer_h2, "h2", n_boot=100)
    assert ci["ci_low"] <= ci["h2"] <= ci["ci_high"]


def test_build_twin_pairs_and_map():
    rng = np.random.default_rng(0)
    rows = []
    fam = 0
    for i in range(40):  # 40 MZ pairs
        fam += 1
        rows += [dict(Subject=f"s{fam}a", Family_ID=fam, ZygosityGT="MZ", Mother_ID=f"m{fam}", Father_ID=f"f{fam}"),
                 dict(Subject=f"s{fam}b", Family_ID=fam, ZygosityGT="MZ", Mother_ID=f"m{fam}", Father_ID=f"f{fam}")]
    for i in range(40):  # 40 DZ pairs with an extra non-twin sibling
        fam += 1
        rows += [dict(Subject=f"s{fam}a", Family_ID=fam, ZygosityGT="DZ", Mother_ID=f"m{fam}", Father_ID=f"f{fam}"),
                 dict(Subject=f"s{fam}b", Family_ID=fam, ZygosityGT="DZ", Mother_ID=f"m{fam}", Father_ID=f"f{fam}"),
                 dict(Subject=f"s{fam}c", Family_ID=fam, ZygosityGT=" ", Mother_ID=f"m{fam}", Father_ID=f"f{fam}")]
    df = pd.DataFrame(rows)
    pairs = heritability.build_twin_pairs(df)
    s = pairs.summary()
    assert s["n_mz_pairs"] == 40 and s["n_dz_pairs"] == 40
    assert s["n_sib_pairs"] == 0  # third sibling has no non-twin partner (twins already used)
    feats = rng.standard_normal((len(df), 5))
    cov = rng.standard_normal((len(df), 2))
    tab = heritability.heritability_map(feats, pairs, method="falconer", covariates=cov, n_boot=20)
    assert tab.shape[0] == 5 and {"h2", "ci_low", "ci_high"} <= set(tab.columns)
    tab_ace = heritability.heritability_map(feats[:, :2], pairs, method="ace")
    assert "lrt_p_CE" in tab_ace.columns
    rg = heritability.genetic_correlation_falconer(feats[:, 0], feats[:, 1], pairs)
    assert set(rg) >= {"rG", "h2_x", "h2_y"}


def test_spin_permutations_and_tests():
    rng = np.random.default_rng(0)
    n = 50
    def sphere(k):
        v = rng.standard_normal((k, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True) * 100
    lh = sphere(n)
    lh[:, 0] = -np.abs(lh[:, 0])
    rh = lh * np.array([-1, 1, 1])
    spins = spatial_nulls.spin_permutations(lh, rh, n_perm=200, seed=1)
    assert spins.shape == (200, 2 * n)
    assert spins[:, :n].max() < n and spins[:, n:].min() >= n
    x = rng.standard_normal(2 * n)
    y = x + 0.5 * rng.standard_normal(2 * n)
    res = spatial_nulls.spatial_correlation_test(x, y, spins)
    assert res["p_spin"] < 0.05 and res["r"] > 0.5
    z = rng.standard_normal(2 * n)
    res0 = spatial_nulls.spatial_correlation_test(x, z, spins)
    assert 0 < res0["p_spin"] <= 1
    expr = pd.DataFrame(rng.standard_normal((2 * n, 200)), columns=[f"g{i}" for i in range(200)])
    expr["g0"] = x + 0.3 * rng.standard_normal(2 * n)
    gs = spatial_nulls.random_gene_set_null(x, expr, ["g0", "g1"], n_perm=100, spins=spins)
    assert gs["n_genes_used"] == 2 and "p_spin" in gs
    adj = spatial_nulls.fdr_bh(np.array([0.01, 0.04, 0.5, np.nan]))
    assert np.isnan(adj[3]) and adj[0] <= adj[1] <= adj[2]
    assert heritability.disattenuated_h2(0.3, 0.6) == pytest.approx(0.5)
