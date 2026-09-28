"""Synthetic-data tests for xspecies_it (no network, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xspecies_it import cross_species as xs  # noqa: E402
from xspecies_it import nulls  # noqa: E402
from xspecies_it import regional_expression as re_  # noqa: E402


def _smooth_map(coords: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    D = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    K = np.exp(-0.5 * (D / scale) ** 2)
    return K @ rng.standard_normal(len(coords))


@pytest.fixture
def toy_regions():
    rng = np.random.default_rng(0)
    n_reg, n_types, n_genes = 60, 5, 40
    coords = rng.uniform(0, 8, size=(n_reg, 3))
    comp = rng.dirichlet(np.ones(n_types), size=n_reg)
    signatures = rng.gamma(2.0, 1.0, size=(n_types, n_genes))
    expr = comp @ signatures + 0.05 * rng.standard_normal((n_reg, n_genes))
    regions = [f"R{i}" for i in range(n_reg)]
    genes = [f"G{i}" for i in range(n_genes)]
    return (pd.DataFrame(expr, index=regions, columns=genes),
            pd.DataFrame(comp, index=regions, columns=[f"T{i}" for i in range(n_types)]),
            pd.DataFrame(coords, index=regions, columns=["x_ccf", "y_ccf", "z_ccf"]), rng)


# ------------------------------------------------------------ regional_expression
def test_normalize_and_aggregate_cells():
    rng = np.random.default_rng(1)
    n_cells, n_genes = 500, 6
    counts = rng.poisson(3.0, size=(n_cells, n_genes))
    meta = pd.DataFrame({"parcellation_structure": rng.choice(["A", "B", "C"], n_cells),
                         "subclass": rng.choice(["s1", "s2"], n_cells)})
    x = re_.normalize_counts(counts)
    assert x.shape == counts.shape and (x >= 0).all()
    agg, n = re_.aggregate_cells_to_regions(x, meta, [f"g{i}" for i in range(n_genes)], min_cells=50)
    assert set(agg.index) == {"A", "B", "C"}
    assert (n >= 50).all()
    comp = re_.composition_matrix(meta, min_cells=50)
    assert np.allclose(comp.sum(axis=1), 1.0)


def test_decompose_composition_recovers_composition_driven_genes(toy_regions):
    expr, comp, coords, rng = toy_regions
    # add a gene that is pure within-type noise (not explained by composition)
    expr = expr.copy()
    expr["noise_gene"] = rng.standard_normal(len(expr))
    dec = re_.decompose_composition(expr, comp)
    assert dec.fitted.shape == expr.shape and dec.residual.shape == expr.shape
    assert np.allclose(dec.fitted + dec.residual, expr.loc[dec.fitted.index])
    assert dec.r2.drop("noise_gene").median() > 0.8
    assert dec.r2["noise_gene"] < 0.3


# ------------------------------------------------------------ nulls
def test_variogram_surrogates_preserve_autocorrelation(toy_regions):
    _, _, coords, rng = toy_regions
    c = coords.to_numpy()
    x = _smooth_map(c, rng, scale=1.5)
    surr, fit = nulls.variogram_surrogates(x, c, n_surr=30, seed=1, return_fit=True)
    assert surr.shape == (30, len(x))
    I_obs = nulls.morans_i(x, c)
    I_surr = np.array([nulls.morans_i(s, c) for s in surr])
    I_perm = np.array([nulls.morans_i(rng.permutation(x), c) for _ in range(30)])
    # surrogates keep far more spatial autocorrelation than plain permutations
    assert I_surr.mean() > I_perm.mean() + 0.1
    assert I_surr.mean() > 0.5 * I_obs
    assert np.nanmedian(fit) > 0.7


def test_spatial_null_is_calibrated_for_independent_maps(toy_regions):
    _, _, coords, rng = toy_regions
    c = coords.to_numpy()
    x = _smooth_map(c, rng, scale=1.5)
    y = _smooth_map(c, rng, scale=1.5)
    res = nulls.spatial_null_corr(x, y, c, n_surr=200, seed=2)
    assert 0.0 < res["p_spatial"] <= 1.0
    assert abs(res["null_mean"]) < 0.2


def test_gene_ensemble_null_detects_enriched_set():
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 0.1, 500)
    mask = np.zeros(500, bool)
    mask[:30] = True
    scores[mask] += 0.3
    level = rng.normal(size=500)
    res = nulls.gene_ensemble_null(scores, mask, match_on=level, n_perm=500)
    assert res["p_ensemble"] < 0.01 and res["z"] > 3
    null_res = nulls.gene_ensemble_null(rng.normal(size=500), mask, n_perm=500)
    assert null_res["p_ensemble"] > 0.01


def test_panel_aware_enrichment_returns_both_backgrounds():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=2000)
    panel = np.zeros(2000, bool)
    panel[:500] = True
    scores[panel] += 0.5  # panel genes systematically higher (panel bias)
    gene_set = np.zeros(2000, bool)
    gene_set[:40] = True  # gene set inside the panel
    res = nulls.panel_aware_enrichment(scores, gene_set, panel, n_perm=500)
    assert res["genome_background"]["p_ensemble"] < res["panel_background"]["p_ensemble"]


def test_bh_fdr_monotone():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    q = nulls.bh_fdr(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12) and q.max() <= 1


# ------------------------------------------------------------ cross_species
def test_align_orthologs_and_profiles(toy_regions):
    expr, comp, coords, rng = toy_regions
    mouse = expr
    human = expr.copy() + 0.2 * rng.standard_normal(expr.shape)
    human.columns = [g.replace("G", "H") for g in expr.columns]
    orth = pd.DataFrame({"Gene name": list(expr.columns), "Human gene name": list(human.columns)})
    m, h, o = xs.align_orthologs(mouse, human, orth)
    assert list(m.columns) == list(h.columns) == list(o["Gene name"])
    bmap = mouse["G0"] * 2 + 0.1 * rng.standard_normal(len(mouse))
    prof_m = xs.association_profile(bmap, m)
    prof_h = xs.association_profile(bmap, h)
    assert prof_m["G0"] > 0.9
    agree = xs.profile_agreement(prof_m, prof_h, n_boot=200)
    assert agree["rho"] > 0.8 and agree["ci_low"] <= agree["rho"] <= agree["ci_high"]


def test_conjunction_and_replication_table(toy_regions):
    expr, comp, coords, rng = toy_regions
    bmap = expr[["G0", "G1", "G2", "G3"]].mean(axis=1) + 0.05 * rng.standard_normal(len(expr))
    res_m = xs.test_association_in_species(bmap, expr, coords, ["G0", "G1", "G2", "G3"], n_surr=100, n_perm=300)
    assert res_m.r_set > 0.3 and res_m.p_spatial < 0.05
    weak = xs.SpeciesAssociation(r_set=-0.1, p_spatial=0.6, p_ensemble=0.7, z_ensemble=0.2)
    strong = xs.SpeciesAssociation(r_set=0.5, p_spatial=0.001, p_ensemble=0.002, z_ensemble=4.0)
    assert xs.conjunction_p(strong, weak) == 1.0  # sign disagreement
    assert xs.conjunction_p(strong, strong) == 0.002
    table = xs.replication_table({"a": (strong, strong), "b": (strong, weak), "c": (strong, res_m)})
    assert table.loc[table.association == "a", "replicates"].item()
    assert not table.loc[table.association == "b", "replicates"].item()
    rate = xs.replication_rate(table, n_boot=100)
    assert 0 <= rate["ci_low"] <= rate["rate"] <= rate["ci_high"] <= 1
    assert xs.attenuation_corrected(0.3, 0.5, 0.5) == pytest.approx(0.6)
