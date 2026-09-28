"""Synthetic tests for meso_vs_axon: CCF assignment, concordance metrics, heterogeneity index and nulls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meso_vs_axon import ccf_assign as ca  # noqa: E402
from meso_vs_axon import concordance as cc  # noqa: E402
from meso_vs_axon import heterogeneity as het  # noqa: E402
from meso_vs_axon.allen_connectivity import normalize_vector, region_projection_vector  # noqa: E402

RES = 25.0


def toy_annotation():
    """Small CCF-like volume: shape (8, 8, 8) at 25 um; fine ids 11/12 (left), 21/22 (right), 0 outside."""
    ann = np.zeros((8, 8, 8), dtype=np.int64)
    ann[:4, :, :4] = 11   # anterior-left  -> summary A (id 1)
    ann[4:, :, :4] = 12   # posterior-left -> summary B (id 2)
    ann[:4, :, 4:] = 21   # anterior-right -> summary A
    ann[4:, :, 4:] = 22   # posterior-right -> summary B
    ancestors = {11: [11, 1, 997], 12: [12, 2, 997], 21: [21, 1, 997], 22: [22, 2, 997]}
    summary_ids = [1, 2]
    acronyms = {1: "A", 2: "B"}
    return ann, ancestors, summary_ids, acronyms


def toy_neuron(midline=100.0):
    # soma in right-anterior (A), axon: one branch stays ipsi in B, one crosses to contra A
    text = "\n".join([
        "1 1 30 30 150 5 -1",
        "2 2 40 30 150 1 1",
        "3 2 60 30 150 1 2",
        "4 2 150 30 150 1 3",   # posterior-right -> B ipsi (long segment)
        "5 2 170 30 150 1 4",   # tip in B ipsi
        "6 2 60 30 100 1 3",
        "7 2 60 30 40 1 6",     # anterior-left -> A contra
        "8 2 60 30 20 1 7",     # tip in A contra
    ])
    return ca.parse_swc_text(text)


# ---------------------------------------------------------------------- CCF assignment
def test_assignment_and_target_vector():
    ann, anc, summ, acr = toy_annotation()
    tree = toy_neuron()
    tips = ca.axon_terminals(tree)
    assert tips.shape == (2, 3)
    fine = ca.assign_points_to_structures(tips, ann, RES)
    assert fine.tolist() == [22, 11]
    assert ca.collapse_to_summary(fine, anc, summ).tolist() == [2, 1]
    vec_t = ca.per_neuron_target_vector(tree, ann, RES, anc, summ, acr, weight="terminals")
    assert set(vec_t.index) == {"B_ipsi", "A_contra"} and vec_t.sum() == pytest.approx(1.0)
    vec_l = ca.per_neuron_target_vector(tree, ann, RES, anc, summ, acr, weight="length")
    assert vec_l.sum() == pytest.approx(1.0) and vec_l["B_ipsi"] > vec_l["A_contra"]
    assert ca.soma_summary_structure(tree, ann, RES, anc, summ) == 1
    # outside-volume points map to 0
    assert ca.assign_points_to_structures(np.array([[-10.0, 0, 0], [1e5, 0, 0]]), ann, RES).tolist() == [0, 0]


def test_mouselight_json_roundtrip(tmp_path):
    nrn = {"idString": "AA0001", "soma": {"x": 30, "y": 30, "z": 150, "radius": 5, "allenId": 21},
           "axon": [{"sampleNumber": 1, "structureIdentifier": 2, "x": 40, "y": 30, "z": 150, "radius": 1, "parentNumber": -1, "allenId": 21},
                    {"sampleNumber": 2, "structureIdentifier": 2, "x": 170, "y": 30, "z": 150, "radius": 1, "parentNumber": 1, "allenId": 22}],
           "dendrite": [{"sampleNumber": 1, "structureIdentifier": 3, "x": 20, "y": 30, "z": 150, "radius": 1, "parentNumber": -1, "allenId": 21}]}
    p = tmp_path / "ml.json"
    p.write_text(json.dumps({"neurons": [nrn]}))
    trees = ca.mouselight_json_neurons(p)
    assert len(trees) == 1 and len(trees[0]) == 4
    assert (trees[0].types == 2).sum() == 2 and trees[0].parent_idx[1] == 0  # axon root attached to soma
    ann, *_ = toy_annotation()
    assert ca.allen_id_agreement(trees[0], ann, RES) == pytest.approx(1.0)
    n = ca.mouselight_json_to_swc_files(p, tmp_path / "out")
    assert n == 1 and (tmp_path / "out" / "AA0001.swc").exists()
    re = ca.read_swc(tmp_path / "out" / "AA0001.swc")
    assert len(re) == 4


# ---------------------------------------------------------------------- concordance
def _synthetic_population(n_neurons=60, n_targets=30, n_channels=3, seed=0, structured=True, channel_seed=0):
    """Neurons drawn from distinct projection 'channels' (structured) or from the pooled bulk (null).

    ``channel_seed`` fixes the channels (hence the bulk vector) so that structured and null
    populations can be compared against the same bulk map.
    """
    crng = np.random.default_rng(channel_seed)
    rng = np.random.default_rng(seed)
    channels = np.zeros((n_channels, n_targets))
    for c in range(n_channels):
        idx = crng.choice(n_targets, size=6, replace=False)
        channels[c, idx] = crng.dirichlet(np.ones(6))
    bulk = channels.mean(axis=0)
    M = np.zeros((n_neurons, n_targets))
    for i in range(n_neurons):
        if structured:
            base = channels[rng.integers(n_channels)]
            v = base * rng.uniform(0.5, 1.5, size=n_targets)
        else:
            k = 6
            idx = rng.choice(n_targets, size=k, replace=False, p=bulk / bulk.sum())
            v = np.zeros(n_targets)
            v[idx] = rng.dirichlet(np.ones(k))
        M[i] = v / v.sum()
    cols = [f"T{j}_ipsi" for j in range(n_targets)]
    return pd.Series(bulk, index=cols), pd.DataFrame(M, columns=cols)


def test_concordance_metrics_basic():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert cc.jaccard(a, b) == pytest.approx(1 / 3)
    assert cc.auroc(np.array([0.9, 0.8, 0.1, 0.0]), np.array([1, 1, 0, 0], bool)) == pytest.approx(1.0)
    prec, rec = cc.precision_recall_at_k(np.array([0.9, 0.8, 0.1, 0.0]), np.array([1, 0, 1, 0], bool), k=2)
    assert prec == pytest.approx(0.5) and rec == pytest.approx(0.5)
    bulk, M = _synthetic_population()
    tab = cc.concordance_table(bulk, M)
    assert tab["auroc"].mean() > 0.8 and tab["jaccard"].between(0, 1).all()
    pooled = cc.pooled_concordance(bulk, M)
    assert pooled["spearman"] > 0.7
    curve = cc.subsampling_curve(bulk, M, n_values=(1, 5, 20, 50), n_rep=10)
    means = curve.groupby("n")["spearman"].mean()
    assert means.loc[50] > means.loc[1]
    assert cc.n_for_recovery(curve, rho_target=0.5) is not None
    obs, null, p = cc.label_permutation_null(bulk, M, n_perm=20)
    assert obs > null.mean() and p < 0.1


# ---------------------------------------------------------------------- heterogeneity
def test_heterogeneity_and_independent_sampling_null():
    bulk, M_struct = _synthetic_population(structured=True, seed=1)
    _, M_null = _synthetic_population(structured=False, seed=2)
    h = het.heterogeneity_index(M_struct, bulk.to_numpy())
    assert 0 <= h["phi_jaccard"] <= 1 and 0 <= h["motif_entropy"] <= 1
    assert h["targets_per_neuron"] == pytest.approx(6, abs=0.5)
    # parallel channels -> every neuron has a near-twin -> nearest-neighbour distance far below the null
    nn = het.mean_nearest_neighbour_distance
    dev_struct = het.null_deviation(M_struct, bulk.to_numpy(), n_sim=100, stat=nn)
    dev_null = het.null_deviation(M_null, bulk.to_numpy(), n_sim=100, stat=nn)
    assert dev_struct["z"] < -2 and dev_struct["p"] < 0.05
    assert abs(dev_null["z"]) < 3.5  # neurons sampled independently from the bulk are consistent with the null
    assert dev_struct["z"] < dev_null["z"]
    # PHI (mean pairwise distance) is consistent with the null for the null population
    assert abs(het.null_deviation(M_null, bulk.to_numpy(), n_sim=100)["z"]) < 3.5
    assert h["motif_entropy"] < het.heterogeneity_index(M_null)["motif_entropy"]
    est, lo, hi = het.bootstrap_ci(M_struct, n_boot=100)
    assert lo <= est <= hi
    r = het.rarefied_statistic(M_struct, n_common=20, n_rep=20)
    assert 0 <= r <= 1
    table = het.region_table({"R1": M_struct, "R2": M_null}, {"R1": bulk, "R2": bulk}, n_common=20, n_sim=50, n_boot=50)
    assert set(table.index) == {"R1", "R2"} and {"dev_z", "devnn_z", "phi_rarefied"} <= set(table.columns)
    assert table.loc["R1", "devnn_z"] < table.loc["R2", "devnn_z"]


def test_identical_neurons_have_zero_heterogeneity():
    M = pd.DataFrame(np.tile([0.5, 0.5, 0, 0], (5, 1)), columns=list("abcd"))
    h = het.heterogeneity_index(M)
    assert h["phi_jaccard"] == 0 and h["motif_entropy"] == 0 and h["multi_target_fraction"] == 1


# ---------------------------------------------------------------------- allen helpers
def test_region_projection_vector_and_normalize():
    mat = pd.DataFrame([[0.5, 0.1, 0.0], [0.3, 0.3, 0.1], [0.0, 0.9, 0.2]], index=[1, 2, 3],
                       columns=["A_ipsi", "B_ipsi", "B_contra"])
    exps = pd.DataFrame({"id": [1, 2, 3], "structure_abbrev": ["A", "A", "B"]})
    v = region_projection_vector(mat, exps, "A")
    assert v["A_ipsi"] == pytest.approx(0.4) and v.name == "A"
    nv = normalize_vector(v, drop_self="A")
    assert "A_ipsi" not in nv.index and nv.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        region_projection_vector(mat, exps, "Z")
