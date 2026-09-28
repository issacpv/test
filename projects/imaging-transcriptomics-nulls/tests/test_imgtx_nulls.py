"""Synthetic-data tests for imgtx_nulls (no atlas downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imgtx_nulls.maps import (  # noqa: E402
    parcellate_vector, load_parcellated_csv, split_hemispheres, geodesic_distance_matrix, align_map_to_expression,
)
from imgtx_nulls.expression import simulate_expression, differential_stability, zscore_genes  # noqa: E402
from imgtx_nulls.nulls import (  # noqa: E402
    spin_permutations, naive_permutations, variogram_surrogates, moran_spectral_randomization, moran_i,
    knn_weights, null_pvalue, compare_nulls,
)
from imgtx_nulls.enrichment import gene_scores, enrichment_with_nulls, bh_fdr, make_gene_sets  # noqa: E402


@pytest.fixture(scope="module")
def geometry():
    lh, rh = split_hemispheres(60)
    coords = np.vstack([lh, rh])
    D = geodesic_distance_matrix(coords, radius=100.0)
    return lh, rh, coords, D


def _smooth_map(D, length_scale, seed):
    rng = np.random.default_rng(seed)
    K = np.exp(-D / length_scale) + 1e-6 * np.eye(D.shape[0])
    return np.linalg.cholesky(K) @ rng.normal(size=D.shape[0])


def test_maps_helpers(tmp_path):
    vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    labels = np.array([1, 1, 2, 2, 0])
    s = parcellate_vector(vals, labels, {1: "A", 2: "B"})
    assert s["A"] == 1.5 and s["B"] == 3.5 and len(s) == 2
    p = tmp_path / "map.csv"
    pd.DataFrame({"region": ["A", "B", "C"], "value": [0.1, 0.2, np.nan]}).to_csv(p, index=False)
    m = load_parcellated_csv(p)
    expr = pd.DataFrame(np.ones((3, 2)), index=["A", "B", "C"], columns=["g1", "g2"])
    mm, ee = align_map_to_expression(m, expr)
    assert list(mm.index) == ["A", "B"] and ee.shape == (2, 2)


def test_spin_permutations_properties(geometry):
    lh, rh, coords, _ = geometry
    n = coords.shape[0]
    perms = spin_permutations(lh, rh, n_perm=20, method="vasa", seed=1)
    assert perms.shape == (20, n)
    for row in perms:
        assert sorted(row.tolist()) == list(range(n))          # true permutation
        assert (row[:60] < 60).all() and (row[60:] >= 60).all()  # hemispheres stay separate
    near = spin_permutations(lh, rh, n_perm=20, method="nearest", seed=1)
    assert near.shape == (20, n) and (near[:, :60] < 60).all()


def test_spatial_nulls_are_more_conservative(geometry):
    lh, rh, coords, D = geometry
    x = _smooth_map(D, 40.0, seed=1)
    y = _smooth_map(D, 40.0, seed=2)  # independent smooth map
    res = compare_nulls(x, y, lh, rh, D=D, n_perm=200, seed=0)
    assert set(res) == {"naive", "spin_nearest", "spin_vasa", "variogram", "moran"}
    # spatially-aware nulls have wider null distributions than the naive shuffle
    for m in ("spin_vasa", "variogram", "moran"):
        assert res[m]["null_sd"] > res["naive"]["null_sd"]
    assert 0 < res["naive"]["p"] <= 1


def test_variogram_surrogates_preserve_autocorrelation(geometry):
    _, _, coords, D = geometry
    x = _smooth_map(D, 40.0, seed=3)
    W = knn_weights(D, k=6)
    surr = variogram_surrogates(x, D, n_surr=30, seed=0)
    assert surr.shape == (30, x.size)
    assert np.allclose(np.sort(surr[0]), np.sort(x))  # rank-matched to original values
    i_obs = moran_i(x, W)
    i_surr = np.array([moran_i(s, W) for s in surr])
    i_shuf = np.array([moran_i(x[p], W) for p in naive_permutations(x.size, 30)])
    assert abs(i_surr.mean() - i_obs) < abs(i_shuf.mean() - i_obs)
    assert i_surr.mean() > i_shuf.mean() + 0.1


def test_moran_spectral_randomization(geometry):
    _, _, coords, D = geometry
    x = _smooth_map(D, 40.0, seed=4)
    W = knn_weights(D, k=6)
    i_obs = moran_i(x, W)
    surr = moran_spectral_randomization(x, W, n_surr=25, procedure="singleton", seed=0)
    assert surr.shape == (25, x.size)
    assert np.allclose([moran_i(s, W) for s in surr], i_obs, atol=1e-8)   # exact preservation
    assert np.allclose(surr.mean(axis=1), x.mean()) and np.allclose(surr.var(axis=1), x.var(), rtol=1e-6)
    pair = moran_spectral_randomization(x, W, n_surr=25, procedure="pair", seed=0)
    assert abs(np.mean([moran_i(s, W) for s in pair]) - i_obs) < 0.15
    assert null_pvalue(0.9, np.array([0.1, -0.2, 0.3])) == pytest.approx(1 / 4)


def test_expression_and_enrichment(geometry):
    _, _, coords, D = geometry
    donors = simulate_expression(coords, n_genes=200, n_spatial_genes=50, length_scale=30.0, noise=0.3, n_donors=3, seed=0)
    ds = differential_stability(donors)
    assert ds.shape == (200,)
    assert ds.iloc[:50].mean() > ds.iloc[50:].mean()  # spatial genes are more stable across donors
    expr = donors["donor0"]
    z = zscore_genes(expr)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-10)
    # map built from a planted gene set
    planted = list(expr.columns[:10])
    y = expr[planted].mean(axis=1).to_numpy() + np.random.default_rng(1).normal(0, 0.2, len(expr))
    r = gene_scores(expr, y)
    assert r.loc[planted].mean() > r.drop(planted).mean()
    sets = make_gene_sets(expr.columns, n_sets=10, size=20, seed=0, planted={"PLANTED": planted})
    surr = variogram_surrogates(y, D, n_surr=100, seed=0)
    res = enrichment_with_nulls(expr, y, sets, null_maps=surr, n_gene_null=500, seed=0)
    assert {"category", "size", "score", "p_gene", "q_gene", "p_ensemble", "q_ensemble", "z_ensemble"} <= set(res.columns)
    top = res.iloc[0]
    assert top["category"] == "PLANTED" and top["p_gene"] < 0.01 and top["p_ensemble"] < 0.05
    assert res["p_ensemble"].between(0, 1).all()
    q = bh_fdr([0.01, 0.02, 0.5, 0.04])
    assert q[2] == 0.5 and q[0] <= q[1] <= q[3] <= q[2]
