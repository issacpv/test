"""Synthetic tests for dbs_popvta (no network, no NEURON)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbs_popvta import axon_model as am  # noqa: E402
from dbs_popvta import lead_field as lf  # noqa: E402
from dbs_popvta import morph_paths as mp  # noqa: E402
from dbs_popvta import population as pop  # noqa: E402


# ------------------------------------------------------------------ fields
def test_point_source_formula_and_lead_field_symmetry():
    v = lf.point_source_potential(np.array([[1.0, 0, 0]]), np.array([[0, 0, 0]]), np.array([1.0]), sigma=0.2)
    assert v[0] == pytest.approx(1e3 / (4 * np.pi * 0.2 * 1.0))
    field = lf.LeadField(lf.LEADS["medtronic_3389"], active=[(1, None, 1.0)])
    zc = field.contact_centre()[2]
    a = field.potential(np.array([[2.0, 0, zc]]))[0]
    b = field.potential(np.array([[0, 2.0, zc]]))[0]
    assert a == pytest.approx(b, rel=1e-6)
    assert field.potential(np.array([[1.0, 0, zc]]))[0] > a > field.potential(np.array([[4.0, 0, zc]]))[0]
    # encapsulation lowers conductivity -> higher potentials for the same current
    enc = lf.LeadField(lf.LEADS["medtronic_3389"], active=[(1, None, 1.0)], encapsulation_factor=0.7)
    assert enc.potential(np.array([[2.0, 0, zc]]))[0] > a
    # voltage mode: mean surface potential equals the set voltage
    vf = lf.LeadField(lf.LEADS["medtronic_3389"], active=[(1, None, 1.0)], mode="voltage")
    surf = vf.potential(vf._src, amplitude=2.0)
    assert np.mean(surf) == pytest.approx(2000.0, rel=1e-6)


def test_segmented_contact_is_directional():
    spec = lf.LEADS["bsci_vercise_cartesia"]
    field = lf.LeadField(spec, active=[(1, 0, 1.0)])  # segment 0 faces +x
    zc = field.contact_centre()[2]
    towards = field.potential(np.array([[2.0, 0, zc]]))[0]
    away = field.potential(np.array([[-2.0, 0, zc]]))[0]
    assert towards > away * 1.3
    assert field.efield_magnitude(np.array([[2.0, 0, zc]]), 3.0)[0] > 0


# ------------------------------------------------------------------ axon model
def _straight_axon(D, r, field, length=15.0):
    path = mp.straight_fiber(length, r, field.contact_centre()[2], orientation="parallel")
    axon = am.build_axon_on_path(path, D)
    return axon, field.unit_potential(axon.nodes)


def test_crrss_axon_strength_distance_and_diameter():
    field = lf.LeadField(lf.LEADS["medtronic_3389"], active=[(1, None, 1.0)])
    pulse = am.Pulse(width_ms=0.06, shape="cathodic")
    th = {}
    for D, r in [(5.7, 2.0), (5.7, 4.0), (10.0, 2.0)]:
        axon, ve = _straight_axon(D, r, field)
        th[(D, r)] = am.threshold_current(axon, ve, pulse, hi=30.0)
    assert 0.05 < th[(5.7, 2.0)] < 20.0
    assert th[(5.7, 4.0)] > th[(5.7, 2.0)]        # strength-distance
    assert th[(10.0, 2.0)] < th[(5.7, 2.0)]       # larger fibres are easier to excite
    # a propagated spike: the far node depolarises above -20 mV at threshold, not below
    axon, ve = _straight_axon(5.7, 2.0, field)
    res_hi = axon.simulate(ve, th[(5.7, 2.0)] * 1.05, pulse, record=True)
    res_lo = axon.simulate(ve, th[(5.7, 2.0)] * 0.8, pulse)
    assert res_hi["spiked"] and not res_lo["spiked"]
    assert res_hi["v"].max() > 0.0
    af = am.activating_function(-ve, axon.L)      # cathodic: negative potentials -> positive AF at nearest node
    assert af.argmax() == int(np.argmin(np.linalg.norm(axon.nodes - field.contact_centre(), axis=1)))
    stats = am.af_statistics(ve, axon)
    assert set(stats) >= {"af_max", "n_pos_nodes", "n_local_max"}


def test_resample_path_and_no_spike_without_stimulus():
    p = np.array([[0, 0, 0], [0, 0, 1.0], [0, 1.0, 1.0]])
    rs = am.resample_path(p, 0.25)
    assert len(rs) == 9
    assert np.allclose(np.linalg.norm(np.diff(rs, axis=0), axis=1), 0.25, atol=1e-9)
    axon = am.MyelinatedAxon(rs, 5.7)
    res = axon.simulate(np.zeros(axon.n), 0.0, am.Pulse(), t_end=0.5)
    assert not res["spiked"] and np.all(np.abs(res["v_max"] - am.V_REST) < 1.0)


# ------------------------------------------------------------------ morphology
SWC = """# synthetic
1 1 0 0 0 5 -1
2 2 0 0 -10 0.5 1
3 2 0 0 -300 0.5 2
4 2 100 0 -600 0.5 3
5 2 -100 0 -600 0.5 3
6 2 -100 0 -1200 0.5 5
7 3 10 0 0 1 1
8 3 200 0 0 1 7
"""


def test_axon_paths_descriptors_and_placement():
    swc = mp.load_swc(SWC)
    paths = mp.axon_paths(swc, min_length_mm=0.2)
    assert len(paths) == 2
    lengths = sorted(p.shape[0] for p in paths)
    assert lengths == [4, 5]  # soma attachment + axon nodes on each root-to-terminal path
    bp = mp.branch_points(swc)
    assert bp.shape == (1, 3) and bp[0, 2] == pytest.approx(-0.3)
    d = mp.path_descriptors(paths[0])
    assert d["length_mm"] > d["chord_mm"] and d["tortuosity"] >= 1.0 and d["n_bends"] >= 0
    rng = np.random.default_rng(0)
    placed = mp.place_path(paths[1], distance_mm=2.5, z_mm=3.0, rng=rng)
    c = placed.mean(axis=0)
    assert np.hypot(c[0], c[1]) == pytest.approx(2.5) and c[2] == pytest.approx(3.0)
    assert mp.path_descriptors(placed)["length_mm"] == pytest.approx(mp.path_descriptors(paths[1])["length_mm"])


# ------------------------------------------------------------------ population
def test_sweep_probabilistic_vta_and_variance_decomposition():
    field = lf.LeadField(lf.LEADS["medtronic_3389"], active=[(1, None, 1.0)])
    paths = {"straight": mp.straight_fiber(12.0, 0.0, 0.0, "parallel") - np.array([0, 0, 0.0])}
    sweep = pop.threshold_sweep(field, paths, diameters_um=[5.7, 10.0], distances_mm=[1.0, 2.0, 3.0, 4.5],
                                pulse=am.Pulse(0.06), rotate=False, hi=30.0)
    assert len(sweep) == 8 and np.isfinite(sweep["threshold"]).all()
    curve = pop.activation_probability_curve(sweep, amplitude=3.0)
    assert (np.diff(curve["p_active"].values) <= 1e-12).all()  # non-increasing with distance
    r_lo = pop.probabilistic_vta_radius(sweep, amplitude=1.0)["r_p50"]
    r_hi = pop.probabilistic_vta_radius(sweep, amplitude=5.0)["r_p50"]
    assert r_hi >= r_lo
    iso = pop.efield_isoline_radius(field, amplitude=3.0)
    assert 0.5 < iso < 10.0
    # variance decomposition on synthetic factorial data with a dominant factor
    rng = np.random.default_rng(0)
    rows = []
    for m in range(20):
        for D in (2, 5.7, 10):
            for r in (1, 2, 3):
                rows.append({"morphology": m, "diameter_um": D, "distance_mm": r,
                             "threshold": np.exp(0.05 * m + 1.5 * np.log(r) - 0.3 * D + rng.normal(0, 0.05))})
    vd = pop.variance_decomposition(pd.DataFrame(rows), "threshold", ["morphology", "diameter_um", "distance_mm"], n_boot=20)
    frac = dict(zip(vd["term"], vd["fraction"]))
    assert frac["diameter_um"] > frac["morphology"] and frac["distance_mm"] > frac["morphology"]
    assert abs(sum(frac.values()) - 1.0) < 1e-6
    ln = pop.lognormal_vs_normal(np.exp(rng.normal(0, 0.5, 300)))
    assert ln["aic_lognormal"] < ln["aic_normal"] and ln["cv"] > 0.3
