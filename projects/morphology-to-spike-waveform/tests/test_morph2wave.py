"""Synthetic-data tests for morph2wave (no network, no NEURON, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morph2wave import eap_forward as fw  # noqa: E402
from morph2wave import regression, swc, waveform_features as wff  # noqa: E402


def ball_and_stick_swc(apical_len: float = 300.0, n_basal: int = 4, basal_len: float = 120.0, soma_r: float = 8.0,
                       axon: bool = True, seed: int = 0) -> str:
    """Soma + apical trunk (+y) + basal dendrites (-y, spread in x/z) + axon (-y)."""
    rng = np.random.default_rng(seed)
    rows = [(1, 1, 0.0, 0.0, 0.0, soma_r, -1)]
    nid = 2
    parent = 1
    for k in range(int(apical_len // 20)):
        rows.append((nid, 4, 0.0, 20.0 * (k + 1), 0.0, 1.5 - k * 0.02, parent))
        parent = nid
        nid += 1
    for b in range(n_basal):
        ang = 2 * np.pi * b / n_basal
        parent = 1
        for k in range(int(basal_len // 15)):
            r_ = 15.0 * (k + 1)
            rows.append((nid, 3, r_ * np.cos(ang) * 0.8, -r_ * 0.6, r_ * np.sin(ang) * 0.8, 0.8, parent))
            parent = nid
            nid += 1
    if axon:
        parent = 1
        for k in range(10):
            rows.append((nid, 2, 5.0, -10.0 * (k + 1) - soma_r, 0.0, 0.5, parent))
            parent = nid
            nid += 1
    return "# ball and stick\n" + "\n".join(" ".join(str(v) for v in r) for r in rows)


# ------------------------------------------------------------ swc
def test_parse_and_features():
    tree = swc.parse_swc_text(ball_and_stick_swc())
    f = swc.morphology_features(tree)
    assert f["soma_radius"] == pytest.approx(8.0)
    assert f["n_stems"] == 6  # apical + 4 basal + axon
    assert f["has_apical"] == 1.0 and f["apical_trunk_radius"] > 1.0
    assert f["dend_area_0_50"] > 0 and f["dend_area_total"] > f["dend_area_0_50"]
    assert f["dend_vertical_asymmetry"] > 0  # apical dominates upward
    assert 0 < f["dend_polarity"] <= 1
    assert np.isfinite(f["axon_origin_dist"])
    s, e, r, ty, d = swc.compartmentalize(tree, max_len=5.0)
    assert np.all(np.linalg.norm(e - s, axis=1) <= 5.0 + 1e-9)
    assert len(s) > len(tree)


def test_rotation_preserves_lengths():
    tree = swc.parse_swc_text(ball_and_stick_swc())
    rot = swc.rotate_about_y(tree, np.pi / 3)
    f0, f1 = swc.morphology_features(tree), swc.morphology_features(rot)
    assert f0["dend_area_total"] == pytest.approx(f1["dend_area_total"], rel=1e-6)


# ------------------------------------------------------------ forward model
def test_line_source_matches_point_source_far_away():
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[0.0, 5.0, 0.0]])
    far = np.array([[0.0, 2.5, 500.0], [0.0, 2.5, 1000.0]])
    M = fw.line_source_matrix(starts, ends, far)
    P = fw.point_source_matrix(np.array([[0.0, 2.5, 0.0]]), far)
    assert np.allclose(M, P, rtol=1e-3)
    assert M[1, 0] == pytest.approx(M[0, 0] / 2, rel=1e-3)  # 1/r decay


def test_probe_geometries():
    for name, ctor in fw.PROBES.items():
        p = ctor()
        assert p.xyz.shape[1] == 3 and p.name == name
    assert fw.neuropixels_ultra().pitch_y < fw.neuropixels_2().pitch_y < fw.neuropixels_1().pitch_y


def test_simulated_eap_has_negative_trough_at_soma_and_is_charge_balanced():
    tree = swc.parse_swc_text(ball_and_stick_swc())
    s, e, r, ty, d = swc.compartmentalize(tree, max_len=10.0)
    probe = fw.neuropixels_1(n_rows=30)
    y_mid = probe.xyz[:, 1].mean()
    wf, t = fw.simulate_eap(s, e, r, ty, d, probe, soma_offset=(0.0, y_mid, 30.0))
    assert wf.shape == (len(probe.xyz), len(t))
    I = fw.propagating_spike_currents(s, e, r, ty, d, t)
    assert np.allclose(I.sum(0), 0.0, atol=1e-9)
    ipk = wff.peak_channel(wf)
    assert wf[ipk].min() < -5.0
    assert abs(probe.xyz[ipk, 1] - y_mid) < 40.0  # peak site near the soma depth


# ------------------------------------------------------------ waveform features
def test_waveform_features_on_synthetic():
    t = np.arange(0, 4, 0.02)
    v = -100 * np.exp(-0.5 * ((t - 1.0) / 0.15) ** 2) + 30 * np.exp(-0.5 * ((t - 1.5) / 0.3) ** 2)
    f = wff.single_channel_features(v, t)
    assert f["trough_to_peak_ms"] == pytest.approx(0.5, abs=0.06)
    assert 0.2 < f["half_width_ms"] < 0.5
    assert f["peak_trough_ratio"] == pytest.approx(0.3, abs=0.05)
    tree = swc.parse_swc_text(ball_and_stick_swc())
    s, e, r, ty, d = swc.compartmentalize(tree)
    probe = fw.neuropixels_ultra()
    wf, t = fw.simulate_eap(s, e, r, ty, d, probe, soma_offset=(0.0, probe.xyz[:, 1].mean(), 20.0))
    sp = wff.spatial_features(wf, probe.xyz, t)
    assert sp["n_sites_footprint"] >= 4 and sp["footprint_um"] > 0
    assert sp["decay_exponent"] < 0
    assert -1 <= sp["asymmetry_above_below"] <= 1
    full = wff.waveform_feature_vector(wf, probe.xyz, t)
    assert "trough_to_peak_ms" in full and "spread_um" in full


def test_energy_permutation_test():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(2, 1, 60)
    res = wff.permutation_energy_test(a, b, n_perm=200)
    assert res["p"] < 0.05


# ------------------------------------------------------------ regression / partition
def test_regression_and_variance_partition():
    rng = np.random.default_rng(1)
    n_cells, n_place = 30, 4
    rows = []
    for c in range(n_cells):
        area = rng.uniform(1000, 8000)
        stems = rng.integers(2, 8)
        for p in range(n_place):
            rows.append({"cell": c, "dend_area_0_100": area, "n_stems": stems, "noise": rng.normal(),
                         "footprint_um": 0.01 * area + 3 * stems + 5 * rng.normal()})
    df = pd.DataFrame(rows)
    X = df[["dend_area_0_100", "n_stems", "noise"]]
    res = regression.cross_validated_regression(X, df["footprint_um"], df["cell"], kind="ridge")
    assert res.r2 > 0.7
    assert res.importance.index[0] == "dend_area_0_100"
    assert regression.shuffled_negative_control(X, df["footprint_um"], df["cell"]) < 0.3

    # factorial: value = morphology effect (large) + channel effect (small) + placement noise
    morph_eff = rng.normal(0, 3, 12)
    chan_eff = rng.normal(0, 1, 5)
    rows = []
    for i in range(12):
        for j in range(5):
            for k in range(3):
                rows.append({"morphology": i, "channel_set": j, "placement": k,
                             "feat": morph_eff[i] + chan_eff[j] + 0.3 * rng.normal()})
    part = regression.factorial_variance_partition(pd.DataFrame(rows), "feat")
    assert part["eta2_morphology"] > part["eta2_channel_set"] > part["eta2_interaction"]
    assert abs(sum(v for k, v in part.items() if k.startswith("eta2_")) - 1.0) < 1e-9
