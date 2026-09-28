"""Synthetic tests for morph2ephys: SWC -> graph, ephys extractor on simulated sweeps, ridge baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morph2ephys import swc_graph as sg  # noqa: E402
from morph2ephys.baseline import (  # noqa: E402
    permutation_null_r2,
    residual_type_anova,
    ridge_grouped_cv,
    transfer_evaluation,
    within_type_partial_r2,
)
from morph2ephys.ephys_features import Sweep, extract_long_square_features, sag_ratio, simulate_sweep  # noqa: E402


def synthetic_swc(n_bif: int = 20, seed: int = 0, apical: bool = True) -> str:
    rng = np.random.default_rng(seed)
    rows = [(1, 1, 0.0, 0.0, 0.0, 6.0, -1), (2, 2, 0.0, 5.0, 0.0, 0.5, 1), (3, 2, 0.0, 40.0, 0.0, 0.5, 2)]
    nid = 4
    pos = {1: np.zeros(3)}
    tips = []
    for stem_type, start in ((3, np.array([12.0, 0.0, 0.0])), (4 if apical else 3, np.array([0.0, -12.0, 0.0]))):
        rows.append((nid, stem_type, *start, 1.0, 1))
        pos[nid] = start
        tips.append((nid, stem_type))
        nid += 1
    for _ in range(n_bif):
        parent, ty = tips.pop(rng.integers(len(tips)))
        for _ in range(2):
            p = pos[parent] + rng.normal(size=3) * 12.0
            rows.append((nid, ty, *p, 0.7, parent))
            pos[nid] = p
            tips.append((nid, ty))
            nid += 1
    return "\n".join(" ".join(str(v) for v in r) for r in rows)


# ---------------------------------------------------------------------- graph
def test_swc_to_graph_shapes_and_features():
    tree = sg.parse_swc_text(synthetic_swc())
    g = sg.swc_to_graph(tree)  # axon dropped
    assert g.n_nodes == len(tree) - 2
    assert g.edge_index.shape == (2, 2 * (g.n_nodes - 1))
    assert g.x.shape[1] == len(sg.NODE_FEATURE_NAMES)
    assert int(g.x[:, 0].sum()) == 1  # one soma node
    assert (g.x[:, 1] == 0).all()  # no axon nodes
    g2 = sg.swc_to_graph(tree, include_axon=True)
    assert g2.n_nodes == len(tree)
    rot = sg.rotate_about_y(g, np.pi / 2)
    assert np.allclose(rot.pos[:, 1], g.pos[:, 1])
    assert not np.allclose(rot.pos[:, 0], g.pos[:, 0])


def test_morphometric_vector_consistency():
    tree = sg.parse_swc_text(synthetic_swc(n_bif=20, seed=3))
    m = sg.morphometric_vector(tree)
    assert m["dend_n_bif"] == 20
    assert m["dend_n_tips"] == 22  # two stems, each binary: tips = bifs + stems
    assert m["dend_length"] == pytest.approx(m["basal_length"] + m["apical_length"])
    assert 0 < m["apical_fraction"] < 1
    assert m["n_stems"] == 2


def test_subtree_dropout_reduces_nodes():
    tree = sg.parse_swc_text(synthetic_swc(n_bif=30, seed=5))
    g = sg.swc_to_graph(tree)
    g_drop = sg.subtree_dropout(g, tree, p_drop=0.5, rng=np.random.default_rng(0))
    assert 0 < g_drop.n_nodes < g.n_nodes


# ---------------------------------------------------------------------- ephys
def test_extractor_recovers_simulated_parameters():
    r_in, tau, sag = 200.0, 25.0, 0.25
    sweeps = [simulate_sweep(a, r_in_MOhm=r_in, tau_ms=tau, sag_frac=sag, seed=1) for a in (-100, -50, 50, 100, 150, 200)]
    feats = extract_long_square_features(sweeps)
    # steady state = R I (1 - sag)  -> extractor sees an effective Rin of R(1-sag)
    assert feats["rin"] == pytest.approx(r_in * (1 - sag), rel=0.1)
    # the slow sag current already builds during the rise, so the measured peak deflection is
    # below R*I and the sag ratio is below sag_frac; it must still be clearly > 0 and monotone
    assert 0.05 < feats["sag"] < sag + 0.02
    more_sag = extract_long_square_features(
        [simulate_sweep(a, r_in_MOhm=r_in, tau_ms=tau, sag_frac=0.5, seed=1) for a in (-100, -50)])
    assert more_sag["sag"] > feats["sag"] + 0.05
    assert feats["tau"] == pytest.approx(tau, rel=0.35)
    assert feats["rheobase"] in (100.0, 150.0, 200.0)
    assert 0.3 < feats["ap_width"] < 3.0
    assert np.isfinite(feats["fi_slope"]) and feats["fi_slope"] > 0


def test_adaptation_sign():
    sw = simulate_sweep(250.0, adapt_pA_per_spike=40.0, tau_adapt_ms=300.0, seed=2)
    feats = extract_long_square_features([simulate_sweep(-50, seed=0), sw])
    assert feats["adaptation"] > 0  # ISIs lengthen with adaptation


def test_sag_ratio_zero_without_sag():
    sw = simulate_sweep(-80.0, sag_frac=0.0, seed=0)
    assert abs(sag_ratio(sw)) < 0.05


# ---------------------------------------------------------------------- baseline
def _synthetic_regression(n=240, p=12, n_groups=24, seed=0, shift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * scale + shift
    w = np.zeros(p)
    w[:4] = [1.0, -0.8, 0.5, 0.3]
    y = X @ w + rng.normal(scale=0.5, size=n)
    groups = rng.integers(0, n_groups, size=n)
    return X, y, groups


def test_ridge_grouped_cv_and_null():
    X, y, groups = _synthetic_regression()
    res = ridge_grouped_cv(X, y, groups)
    assert res.r2 > 0.6 and res.spearman > 0.7
    obs, null, p = permutation_null_r2(X, y, groups, n_perm=20)
    assert obs > null.max() and p < 0.1


def test_transfer_and_within_type():
    Xs, ys, gs = _synthetic_regression(seed=1)
    Xt, yt, gt = _synthetic_regression(n=120, seed=2)
    tr = transfer_evaluation(Xs, ys, gs, Xt, yt, gt)
    assert tr.r2_transfer > 0.5
    assert abs(tr.transfer_gap) < 0.3
    types = np.where(Xt[:, 0] > 0, "A", "B")
    out = within_type_partial_r2(Xt, yt, types, gt)
    assert out["r2_type_only"] > 0 and out["partial_r2_within_type"] > 0.2
    an = residual_type_anova(yt - yt.mean(), types)
    assert an["p"] < 0.05


def test_gnn_module_imports_without_torch():
    from morph2ephys import gnn

    assert isinstance(gnn.HAVE_TORCH, bool)
    if gnn.HAVE_TORCH:  # pragma: no cover - only when torch is installed
        tree = sg.parse_swc_text(synthetic_swc())
        graphs = [sg.swc_to_graph(tree) for _ in range(6)]
        Y = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
        model, hist = gnn.train_regressor(graphs, Y, epochs=2, hidden=16, n_layers=2, batch_size=3)
        assert gnn.predict(model, graphs).shape == (6, 2)
