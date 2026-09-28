"""Synthetic tests for celltype_mri (no downloads)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celltype_mri.abc_atlas import (assign_structures, clr, coverage_report, densities_by_structure,
                                    sampled_volume_mm3, voxel_density_map)
from celltype_mri.mri import align_tables, contrast_table, regional_stats, structure_centroids, t1w_t2w_ratio
from celltype_mri.regress import (cross_atlas_concordance, fit_cv, nested_comparison, relative_importance)
from celltype_mri.spatial_nulls import (distance_weights, morans_i, msr_surrogates, naive_pvalue,
                                        spatial_pvalue)


def _annotation(n=6, res_um=100.0):
    """6 cubic structures in a 12x6x6 grid (ids 1..6 along x)."""
    ann = np.zeros((12, 6, 6), dtype=int)
    for k in range(n):
        ann[2 * k:2 * k + 2] = k + 1
    return ann


def test_abc_atlas_densities():
    rng = np.random.default_rng(0)
    ann = _annotation()
    res = 100.0
    # 300 cells per structure, mixed classes with structure-dependent oligodendrocyte fraction
    rows = []
    for sid in range(1, 7):
        for _ in range(300):
            x = (2 * (sid - 1) + rng.random() * 2) * res / 1000.0
            y, z = rng.random() * 6 * res / 1000.0, rng.random() * 6 * res / 1000.0
            cls = "Oligo" if rng.random() < 0.1 * sid else "Neuron"
            rows.append({"x": x, "y": y, "z": z, "class": cls})
    cells = pd.DataFrame(rows)
    cells["structure_id"] = assign_structures(cells[["x", "y", "z"]].to_numpy(), ann, res)
    assert set(cells["structure_id"]) == set(range(1, 7))
    vol = sampled_volume_mm3(ann, res)
    assert vol[1] == pytest.approx(2 * 6 * 6 * (0.1 ** 3))
    dens = densities_by_structure(cells, "structure_id", "class", vol, min_cells=100)
    assert dens.shape == (6, 2)
    assert dens["Oligo"].iloc[-1] > dens["Oligo"].iloc[0]
    assert np.allclose(clr(dens).sum(axis=1), 0.0)
    vm = voxel_density_map(cells[["x", "y", "z"]].to_numpy(), ann.shape, res)
    assert vm.sum() == len(cells)
    assert coverage_report(cells).loc[1, "n_cells"] == 300


def test_mri_regional_stats_and_ratio():
    ann = _annotation()
    vol = np.zeros(ann.shape)
    for sid in range(1, 7):
        vol[ann == sid] = sid * 10.0
    s = regional_stats(vol, ann, range(1, 7), min_voxels=10)
    assert list(s.values) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    tab = contrast_table({"t1w": vol, "t2w": vol * 2}, ann, range(1, 7), min_voxels=10)
    assert tab.shape == (6, 2) and np.allclose(t1w_t2w_ratio(vol, vol * 2)[ann > 0], 0.5, atol=1e-5)
    cent = structure_centroids(ann, range(1, 7), 100.0)
    assert cent.shape == (6, 3) and cent["x_mm"].is_monotonic_increasing
    d = pd.DataFrame(np.ones((6, 2)), index=range(1, 7))
    a, b = align_tables(d, tab.loc[[2, 3, 4]])
    assert list(a.index) == [2, 3, 4]


def test_regression_recovers_planted_predictor():
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["Oligo", "Neuron", "Astro", "Micro"])
    y = pd.Series(2.0 * X["Oligo"] + 0.3 * X["Neuron"] + rng.normal(0, 0.5, n))
    res = fit_cv(X, y, "ridge", cv="loo")
    assert res.r2_cv > 0.8 and res.coef.idxmax() == "Oligo"
    imp = relative_importance(X, y)
    assert imp.idxmax() == "Oligo" and imp.sum() == pytest.approx(res.r2_in, abs=0.05)
    comp = nested_comparison(X[["Neuron", "Astro"]], X[["Oligo"]], y, cv="kfold")
    assert comp["delta_r2"] > 0.5
    groups = pd.Series(np.repeat(np.arange(4), n // 4))
    assert np.isfinite(fit_cv(X, y, "pls", cv="group", groups=groups).r2_cv)
    conc = cross_atlas_concordance(imp, imp * 1.1)
    assert conc["kendall_tau"] == pytest.approx(1.0) and conc["lin_ccc"] > 0.9


def test_spatial_nulls_preserve_moran_and_inflate_naive_p():
    rng = np.random.default_rng(2)
    n = 60
    coords = rng.random((n, 3)) * 10
    W = distance_weights(coords, k=8)
    # smooth field: value = smooth function of position + noise -> strong autocorrelation
    x = np.sin(coords[:, 0] / 2) + 0.2 * rng.normal(size=n)
    y = np.cos(coords[:, 0] / 2 + 0.3) + 0.2 * rng.normal(size=n)  # correlated with x only via space
    I_obs = morans_i(x, W)
    S = msr_surrogates(x, W, n_surr=200, seed=0)
    I_surr = np.array([morans_i(s, W) for s in S])
    assert S.shape == (200, n)
    assert np.allclose(S.mean(axis=1), x.mean(), atol=1e-6) and np.allclose(S.std(axis=1), x.std(), atol=1e-6)
    assert abs(I_surr.mean() - I_obs) < 0.5 * abs(I_obs) + 0.05
    stat = lambda v: np.corrcoef(v, y)[0, 1]  # noqa: E731
    p_naive = naive_pvalue(stat, x, n_perm=300, seed=0)
    res = spatial_pvalue(stat, x, W, n_surr=300, seed=0)
    assert res["p_spatial"] >= p_naive  # spatial null is at least as conservative here
    assert 0 < res["p_spatial"] <= 1
