"""Synthetic-data tests for gene_timescales (no network, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gene_timescales import gene_maps, nulls, timescales as ts  # noqa: E402


# ------------------------------------------------------------ timescales
def test_intrinsic_timescale_recovers_ou_timescale():
    tau_true = 0.08
    spikes = ts.simulate_ou_spike_train(tau=tau_true, rate_hz=20.0, duration=600.0, seed=0)
    epochs = [(i * 10.0, (i + 1) * 10.0) for i in range(60)]
    fit = ts.intrinsic_timescale(spikes, epochs, bin_size=0.005, max_lag=0.5, min_fit_lag=0.01)
    assert fit is not None
    assert 0.5 * tau_true < fit.tau < 2.0 * tau_true
    assert fit.r2 > 0.5


def test_sparse_unit_returns_none():
    spikes = np.array([0.5, 3.0, 7.0])
    assert ts.intrinsic_timescale(spikes, [(0.0, 10.0)], min_rate_hz=0.5) is None


def test_two_timescale_fit_orders_by_amplitude():
    lags = np.arange(1, 100) * 0.005
    acf = 0.6 * np.exp(-lags / 0.02) + 0.3 * np.exp(-lags / 0.2) + 0.02
    fit = ts.fit_exponential_timescale(lags, acf, two_timescales=True)
    assert fit.amplitude >= (fit.amplitude_secondary or 0)
    assert fit.r2 > 0.99


def test_tuning_metrics():
    oris = np.arange(0, 360, 45)
    resp = np.where(oris == 90, 10.0, 1.0)
    m = ts.orientation_selectivity(resp, oris)
    assert m["osi"] > 0.5 and m["pref_ori_deg"] == pytest.approx(90, abs=1)
    psth = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9, 10, 6, 4])
    assert ts.response_latency(psth, 0.01, baseline_bins=10, n_sd=3) == pytest.approx(0.0)
    assert ts.adaptation_index(psth, onset_bin=10, early_bins=2, late_bins=2) < 0
    assert ts.preferred_temporal_frequency([0, 0, 5, 0], [1, 2, 4, 8]) == pytest.approx(4.0)


# ------------------------------------------------------------ gene maps / prediction
@pytest.fixture
def toy_areas():
    rng = np.random.default_rng(0)
    n_areas, n_genes, n_types = 30, 60, 4
    comp = rng.dirichlet(np.ones(n_types), size=n_areas)
    sig = rng.gamma(2, 1, size=(n_types, n_genes))
    genes = comp @ sig + 0.1 * rng.standard_normal((n_areas, n_genes))
    areas = [f"A{i}" for i in range(n_areas)]
    G = pd.DataFrame(genes, index=areas, columns=[f"Kcn{i}" if i < 10 else f"G{i}" for i in range(n_genes)])
    C = pd.DataFrame(comp, index=areas, columns=[f"T{i}" for i in range(n_types)])
    coords = pd.DataFrame(rng.uniform(0, 5, (n_areas, 3)), index=areas)
    return G, C, coords, rng


def test_ish_area_matrix_prefers_coronal():
    long = pd.DataFrame({"gene": ["Pvalb"] * 4, "section_data_set_id": [1, 1, 2, 2],
                         "plane": ["coronal", "coronal", "sagittal", "sagittal"],
                         "structure_id": [10, 20, 10, 20], "expression_energy": [1.0, 2.0, 100.0, 200.0]})
    M = gene_maps.ish_area_matrix(long, {10: "VISp", 20: "VISl"})
    assert M.loc["VISp", "Pvalb"] == 1.0 and M.loc["VISl", "Pvalb"] == 2.0


def test_pls_and_ridge_predict_composition_driven_target(toy_areas):
    G, C, coords, rng = toy_areas
    y = pd.Series(C["T0"] * 2.0 + 0.05 * rng.standard_normal(len(C)), index=C.index)
    pls = gene_maps.pls_predict(G, y, max_components=3)
    rid = gene_maps.ridge_predict(G, y)
    assert pls.r2_oos > 0.6 and rid.r2_oos > 0.6
    assert len(pls.loadings) == G.shape[1]
    part = gene_maps.composition_partial(G, C, y)
    assert part["r2_composition"] > 0.8
    assert part["r2_genes_given_composition"] < 0.5
    score, mask = gene_maps.gene_family_score(pls.loadings, ["KCN"])
    assert mask.sum() == 10 and np.isfinite(score)


def test_prediction_fails_for_random_target(toy_areas):
    G, C, coords, rng = toy_areas
    y = pd.Series(rng.standard_normal(len(G)), index=G.index)
    assert gene_maps.ridge_predict(G, y).r2_oos < 0.3


# ------------------------------------------------------------ nulls
def test_variogram_surrogates_and_spatial_null(toy_areas):
    G, C, coords, rng = toy_areas
    c = coords.to_numpy()
    D = np.linalg.norm(c[:, None] - c[None], axis=-1)
    y = np.exp(-0.5 * (D / 1.0) ** 2) @ rng.standard_normal(len(c))
    surr = nulls.variogram_surrogates(y, c, n_surr=40, seed=1)
    assert surr.shape == (40, len(y))
    I_s = np.mean([nulls.morans_i(s, c) for s in surr])
    I_p = np.mean([nulls.morans_i(rng.permutation(y), c) for _ in range(40)])
    assert I_s > I_p
    res = nulls.spatial_null_r2(lambda v: float(np.corrcoef(v, y)[0, 1] ** 2), y, c, n_surr=50)
    assert res["r2"] == pytest.approx(1.0) and res["p_spatial"] < 0.05


def test_hierarchy_and_gene_ensemble_nulls():
    rng = np.random.default_rng(5)
    h = np.linspace(0, 1, 30)
    y = h + 0.1 * rng.standard_normal(30)
    res = nulls.hierarchy_null_r2(lambda v: float(np.corrcoef(v, h)[0, 1] ** 2), y, h, n_perm=200)
    assert res["p_hierarchy"] < 0.05
    scores = rng.normal(0, 0.1, 300)
    mask = np.zeros(300, bool)
    mask[:20] = True
    scores[mask] += 0.5
    ens = nulls.gene_ensemble_null(scores, mask, match_on=rng.normal(size=300), n_perm=300)
    assert ens["p_ensemble"] < 0.01
    q = nulls.bh_fdr([0.01, 0.04, 0.5])
    assert q[0] <= q[1] <= q[2]
