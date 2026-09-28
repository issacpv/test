"""Synthetic tests for ais_plasticity (no NEURON, no network)."""
from __future__ import annotations

import numpy as np
import pytest

from ais_plasticity.ais_theory import (AISParams, calibrate_offset, demand_distance, demand_length,
                                       plasticity_demand, somatic_ap_amplitude_proxy, threshold_mv)
from ais_plasticity.dendritic_load import Passive, load_metrics, somatic_impedance
from ais_plasticity.neuron_builder import spike_features
from ais_plasticity.swc import ball_and_stick, from_array


def _rall_input_resistance(r_soma, d, L, p: Passive):
    a = 4 * np.pi * r_soma ** 2 * 1e-8
    g_soma = a / p.rm_ohm_cm2
    d_cm = d * 1e-4
    lam = np.sqrt((d_cm / 4) * p.rm_ohm_cm2 / p.ra_ohm_cm)
    r_inf = (2 / np.pi) * np.sqrt(p.rm_ohm_cm2 * p.ra_ohm_cm) * d_cm ** -1.5
    g_cyl = np.tanh(L * 1e-4 / lam) / r_inf
    return 1 / (g_soma + g_cyl)


def test_load_metrics_match_rall_and_capacitance_limits():
    p = Passive()
    t = ball_and_stick(10.0, 2.0, 400.0, n_seg=100)
    m = load_metrics(t, p, freqs_hz=(100.0, 1000.0, 20000.0))
    assert m["input_resistance_mohm"] == pytest.approx(_rall_input_resistance(10, 2, 400, p) * 1e-6, rel=0.01)
    # a lone sphere: effective capacitance equals total capacitance at any frequency
    sphere = from_array(np.array([[1, 1, 0, 0, 0, 10.0, -1]]))
    ms = load_metrics(sphere, p, freqs_hz=(1000.0,))
    assert ms["c_eff_1000hz_pf"] == pytest.approx(ms["total_capacitance_pf"], rel=1e-6)
    # with a dendrite, the AIS 'sees' less of the tree at higher frequency
    assert m["c_eff_20000hz_pf"] < m["c_eff_100hz_pf"] <= m["total_capacitance_pf"] * 1.001
    z = somatic_impedance(t, (0.0, 5000.0), p)
    assert abs(z[1]) < abs(z[0])


def test_threshold_monotonic_and_calibration():
    p = AISParams()
    thr = [threshold_mv(d, 40.0, p) for d in (5, 10, 20, 40, 80)]
    assert all(np.diff(thr) < 0)  # threshold falls with distance (resistive coupling)
    thr_len = [threshold_mv(20.0, L, p) for L in (10, 20, 40, 80)]
    assert all(np.diff(thr_len) < 0)
    cal = calibrate_offset(30.0, 40.0, -45.0, p)
    assert threshold_mv(30.0, 40.0, cal) == pytest.approx(-45.0)
    d = demand_distance(threshold_mv(30.0, 40.0, p), 40.0, p)
    assert d == pytest.approx(30.0, rel=1e-4)
    L = demand_length(threshold_mv(30.0, 40.0, p), 30.0, p)
    assert L == pytest.approx(40.0, rel=1e-4)
    assert demand_distance(-200.0, 40.0, p) is None


def test_amplitude_proxy_and_plasticity_demand():
    p = AISParams()
    small = load_metrics(ball_and_stick(6.0, 1.0, 200.0, 40))
    large = load_metrics(ball_and_stick(12.0, 3.0, 800.0, 80))
    a_small = somatic_ap_amplitude_proxy(30.0, 40.0, small["c_eff_1000hz_pf"], small["input_conductance_ns"], p)
    a_large = somatic_ap_amplitude_proxy(30.0, 40.0, large["c_eff_1000hz_pf"], large["input_conductance_ns"], p)
    assert a_large < a_small  # bigger load -> smaller somatic AP for the same AIS
    dem = plasticity_demand(large, reference=small, p=p)
    assert dem["load_ratio"] > 1
    if dem["distance_for_amplitude_um"] is not None:
        assert dem["delta_distance_um"] < 0  # larger tree needs a closer AIS to keep amplitude
    same = plasticity_demand(small, reference=small, p=p)
    assert same["delta_distance_um"] == pytest.approx(0.0, abs=1e-3)


def test_spike_features_on_synthetic_trace():
    t = np.linspace(0, 10, 4001)
    v = -70 + 100 * np.exp(-((t - 5) / 0.3) ** 2)
    f = spike_features(t, v, dvdt_threshold=20.0)
    assert f["spiked"] == 1.0 and -70 < f["threshold_mv"] < 30 and f["peak_mv"] == pytest.approx(30, abs=0.1)
    flat = spike_features(t, np.full_like(t, -70.0))
    assert flat["spiked"] == 0.0 and np.isnan(flat["threshold_mv"])
