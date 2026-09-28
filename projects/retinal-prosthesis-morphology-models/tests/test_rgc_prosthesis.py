"""Synthetic tests for rgc_prosthesis (no network, no NEURON)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rgc_prosthesis import electrode as el  # noqa: E402
from rgc_prosthesis import population_stats as ps  # noqa: E402
from rgc_prosthesis import rgc_model as rm  # noqa: E402
from rgc_prosthesis import swc_morph as sm  # noqa: E402


# ------------------------------------------------------------------ electrodes
def test_disk_and_point_potentials():
    centre = np.array([0.0, 0.0, -30.0])
    v_centre = el.disk_electrode_potential(np.array([[0.0, 0.0, -30.0]]), centre, 50.0, 10.0, sigma=0.3)[0]
    assert v_centre == pytest.approx(1e3 * 10.0 / (4 * 0.3 * 50.0), rel=1e-6)  # V0 on the disk
    v_near = el.disk_electrode_potential(np.array([[0.0, 0.0, 0.0]]), centre, 50.0, 10.0)[0]
    v_far = el.disk_electrode_potential(np.array([[0.0, 0.0, 200.0]]), centre, 50.0, 10.0)[0]
    assert v_centre > v_near > v_far > 0
    # far field of a disk approaches the point source with the same current
    far = np.array([[0.0, 0.0, 5000.0]])
    assert el.disk_electrode_potential(far, centre, 50.0, 10.0)[0] == pytest.approx(
        el.point_source_potential(far, centre, 10.0)[0], rel=0.01)
    epi = el.Electrode.epiretinal(height_um=30, radius_um=50)
    sub = el.Electrode.subretinal(depth_um=150, radius_um=50)
    assert epi.centre_um[2] < 0 < sub.centre_um[2]
    w = el.Pulse(0.1, "biphasic").waveform(np.array([0.0, 0.06, 0.16, 0.3]))
    assert list(w) == [0.0, -1.0, 1.0, 0.0]


# ------------------------------------------------------------------ morphology
def test_synthetic_rgc_morphometrics_and_axon_synthesis():
    swc = sm.synthetic_rgc(n_dendrites=4, with_axon=False)
    m = sm.morphometrics(swc)
    assert m["soma_diam_um"] == pytest.approx(15.0)
    assert m["dend_field_diam_um"] > 50 and m["dend_length_um"] > 100 and m["n_branch_points"] == 4
    assert m["strat_depth_um"] == pytest.approx(25.0, abs=3)
    assert not sm.has_axon(swc)
    ax = sm.synthesize_axon(swc, ais_start_um=30, ais_len_um=35, axon_len_um=500)
    assert sm.has_axon(ax)
    reg = ax["type"].value_counts()
    assert reg[sm.HILLOCK] == 6 and reg[sm.AIS] == 7 and reg[sm.AXON] == 100
    assert (ax.loc[ax["type"] == sm.AXON, "z"] < 0).all()  # nerve-fibre layer on the vitreal side
    # AIS geometry is a design factor: replacing the axon changes AIS placement
    ax2 = sm.synthesize_axon(ax, ais_start_um=60, ais_len_um=35, axon_len_um=500)
    assert ax2["type"].value_counts()[sm.HILLOCK] == 12
    comp = sm.compartmentalize(ax, max_len_um=10.0)
    assert comp.n > 100
    assert comp.length.max() <= 10.0 + 1e-9
    assert comp.parent[0] == -1 and (comp.parent[1:] >= 0).all() and (comp.parent[1:] < np.arange(1, comp.n)).all()
    assert set(np.unique(comp.region)) == {sm.SOMA, sm.BASAL, sm.HILLOCK, sm.AIS, sm.AXON}
    # orientation helper flips a cell whose dendrites point to negative z
    flipped = swc.copy()
    flipped["z"] = -flipped["z"]
    assert sm.morphometrics(sm.orient_z(flipped))["strat_depth_um"] > 0


# ------------------------------------------------------------------ biophysics
@pytest.fixture(scope="module")
def cell():
    swc = sm.synthetic_rgc(n_dendrites=3, dend_len_um=80.0, axon_len_um=300.0)
    comp = sm.compartmentalize(swc, max_len_um=15.0)
    c = rm.RGCCable(comp)
    eq = c.equilibrate(t_ms=30.0)
    assert eq["max_drift_mV"] < 3.0 and not eq["spontaneous_spike"]
    return c


def test_threshold_epiretinal_and_distance_dependence(cell):
    pulse = el.Pulse(width_ms=0.1, shape="biphasic")
    near = el.Electrode.epiretinal(height_um=30, radius_um=50)
    far = el.Electrode.epiretinal(height_um=120, radius_um=50)
    th_near = rm.threshold_current(cell, near.unit_potential(cell.comp.xyz), pulse, hi=2000.0)
    th_far = rm.threshold_current(cell, far.unit_potential(cell.comp.xyz), pulse, hi=2000.0)
    assert np.isfinite(th_near) and 0.5 < th_near < 2000.0
    assert th_far > th_near
    # supra-threshold: spike propagates to the distal axon and invades the soma; sub-threshold: none
    ve = near.unit_potential(cell.comp.xyz)
    hi = cell.run(ve, th_near * 1.1, pulse, t_end=pulse.duration_ms + 2.5)
    lo = cell.run(ve, th_near * 0.7, pulse, t_end=pulse.duration_ms + 2.5)
    assert hi["spiked"] and hi["soma_spiked"] and not lo["spiked"]
    site = rm.first_spiking_region(cell, ve, th_near * 1.1, pulse)
    assert site["region"] in (sm.AIS, sm.HILLOCK, sm.AXON, sm.SOMA)
    assert ps.activation_site(site["region"]) in ("ais", "hillock", "distal_axon", "soma")
    af = rm.activating_function(cell, -ve)
    assert af.max() > 0


def test_longer_pulse_lowers_threshold(cell):
    ve = el.Electrode.epiretinal(height_um=50, radius_um=50).unit_potential(cell.comp.xyz)
    th_short = rm.threshold_current(cell, ve, el.Pulse(0.1, "cathodic"), hi=2000.0)
    th_long = rm.threshold_current(cell, ve, el.Pulse(0.5, "cathodic"), hi=2000.0)
    assert th_long < th_short  # strength-duration


# ------------------------------------------------------------------ population statistics
def test_population_statistics_recover_planted_structure():
    rng = np.random.default_rng(0)
    rows = []
    for t, (soma, off) in {"alpha": (22.0, 0.0), "small": (12.0, 0.3), "mid": (16.0, 0.1)}.items():
        for i in range(40):
            sd = soma + rng.normal(0, 2)
            ais = rng.choice([20.0, 40.0, 60.0])
            thr = np.exp(3.0 - 0.06 * sd + 0.01 * ais + off + rng.normal(0, 0.15))
            rows.append({"cell_id": f"{t}{i}", "cell_type": t, "soma_diam_um": sd, "ais_start_um": ais,
                         "dend_field_diam_um": rng.normal(200, 30), "threshold_uA": thr})
    df = pd.DataFrame(rows)
    summ = ps.type_threshold_summary(df)
    assert set(summ["cell_type"]) == {"alpha", "small", "mid"}
    assert summ.set_index("cell_type").loc["alpha", "median"] < summ.set_index("cell_type").loc["small", "median"]
    det = ps.morphological_determinants(df, ["soma_diam_um", "ais_start_um", "dend_field_diam_um"])
    tab = det["table"].set_index("feature")
    assert tab.loc["soma_diam_um", "coef"] < 0 and tab.loc["ais_start_um", "coef"] > 0
    assert abs(tab.loc["dend_field_diam_um", "coef"]) < 0.1
    assert det["r2_marginal"] > 0.3
    vf = ps.variance_fractions(df, "threshold_uA", ["cell_type", "ais_start_um"])
    assert vf["cell_type"] > vf["ais_start_um"] > 0 and abs(sum(vf.values()) - 1) < 1e-6
    sel = ps.selectivity_auc(df, "alpha", "small")
    assert sel["auc"] < 0.5 and sel["p_mannwhitney"] < 0.01
    pw = ps.pairwise_selectivity(df)
    assert len(pw) == 3 and (pw["auc_max"] >= 0.5).all()
    emp = df.loc[df["cell_type"] == "alpha", "threshold_uA"].values * 3.0
    cmp = ps.compare_with_empirical(df.loc[df["cell_type"] == "alpha", "threshold_uA"].values, emp)
    assert cmp["scale"] == pytest.approx(3.0) and cmp["ks_p"] > 0.5 and cmp["qq_slope_log"] == pytest.approx(1.0, abs=1e-6)
