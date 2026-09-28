"""Synthetic-data tests for resp_mech."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resp_mech import charting, single_compartment as sc, state_space


def test_algebra_and_breath_fit():
    assert sc.driving_pressure(24, 8) == 16
    assert sc.static_compliance(480, 24, 8) == pytest.approx(30.0)
    assert sc.airway_resistance(30, 24, 0.5) == pytest.approx(12.0)
    mp = sc.mechanical_power(20, 480, 30, 24, 8)
    assert mp == pytest.approx(0.098 * 20 * 0.48 * (30 - 8), rel=1e-6)
    assert sc.ti_from_rr_ie(20, 0.5) == pytest.approx(1.0)
    assert sc.inspiratory_flow_l_s(500, 1.0) == pytest.approx(0.5)
    b = sc.simulate_vcv_breath(r=10.0, c_ml=40.0, peep=6.0, vt_ml=450.0, flow_l_s=0.6, fs=200, noise_cmh2o=0.2)
    fit = sc.fit_breath_least_squares(b.paw, b.flow, b.volume)
    assert fit["R"] == pytest.approx(10.0, abs=1.0)
    assert fit["C"] == pytest.approx(40.0, rel=0.1)
    assert fit["PEEP"] == pytest.approx(6.0, abs=0.5)
    est = sc.charted_point_estimates(30, 24, 8, 480, 0.5, 20)
    assert est["dp"] == 16 and est["R"] == pytest.approx(12.0)
    assert np.isnan(sc.charted_point_estimates(np.nan, 24, 8, 480, 0.5, 20)["R"])


def test_kalman_recovers_piecewise_mechanics():
    times = np.arange(0, 48 * 3600, 120.0)  # 2-min charting for 48 h
    e_true = np.where(times < 24 * 3600, 25.0, 45.0)  # elastance rises after 24 h
    r_true = np.full(len(times), 12.0)
    obs = state_space.simulate_charting(e_true, r_true, times, pplat_fraction=0.2, noise_sd=1.5, seed=0)
    kf = state_space.KalmanMechanics()
    out = kf.filter(obs)
    first = out[out["time_s"] < 20 * 3600]
    last = out[out["time_s"] > 30 * 3600]
    assert first["E_mean"].median() == pytest.approx(25.0, abs=4.0)
    assert last["E_mean"].median() == pytest.approx(45.0, abs=5.0)
    assert out["R_mean"].median() == pytest.approx(12.0, abs=3.0)
    assert (out["E_sd"] > 0).all() and (out["dp_inferred"] > 0).all()
    # rows without any plateau still get an inferred driving pressure
    assert out.loc[out["pplat"].isna(), "dp_inferred"].notna().all()
    with pytest.raises(KeyError):
        kf.filter(obs.drop(columns=["peep"]))


def test_kalman_uncertainty_grows_over_gaps():
    times = np.array([0, 3600, 7200, 7200 + 12 * 3600, 7200 + 12 * 3600 + 3600], float)
    obs = state_space.simulate_charting(np.full(5, 30.0), np.full(5, 10.0), times, pplat_fraction=1.0, seed=1)
    out = state_space.KalmanMechanics().filter(obs, smooth=False)
    # predicted variance right after a 12 h gap is larger than after a 1 h gap
    assert out.loc[3, "E_sd"] >= out.loc[2, "E_sd"] * 0.9


def test_label_mapping_pivot_flow_and_episodes():
    labels = ["Peak Insp. Pressure", "Plateau Pressure", "PEEP set", "Tidal Volume (observed)", "Respiratory Rate (Total)",
              "Inspiratory Time", "Ventilator Mode", "Heart Rate", "Compliance", "Resistance"]
    m = charting.map_labels(labels)
    assert m["Peak Insp. Pressure"] == "ppeak" and m["Plateau Pressure"] == "pplat"
    assert m["PEEP set"] == "peep" and m["Tidal Volume (observed)"] == "vt_obs"
    assert m["Respiratory Rate (Total)"] == "rr" and m["Inspiratory Time"] == "ti"
    assert m["Ventilator Mode"] == "mode" and "Heart Rate" not in m
    assert m["Compliance"] == "compliance_charted" and m["Resistance"] == "resistance_charted"
    rows = pd.DataFrame({
        "stay_id": [1] * 7 + [2],
        "time": [0, 0, 0, 0, 0, 0, 0, 0],
        "label": ["Peak Insp. Pressure", "Plateau Pressure", "PEEP set", "Tidal Volume (observed)", "Respiratory Rate (Total)", "Inspiratory Time", "Ventilator Mode", "PEEP set"],
        "value": ["30", "24", "8", "0.45", "20", "1.0", "CMV/ASSIST", "5"],
    })
    wide = charting.pivot_charting(rows, m)
    assert len(wide) == 2 and wide.loc[0, "mode"] == "CMV/ASSIST"
    wide = charting.harmonize_units(wide)
    assert wide.loc[0, "vt_obs"] == pytest.approx(450.0)
    wide = charting.derive_flow_proxy(wide)
    assert wide.loc[0, "flow_l_s"] == pytest.approx(0.45)
    assert charting.classify_mode("CMV/ASSIST") == "controlled"
    assert charting.classify_mode("Pressure Support") == "spontaneous"
    assert charting.classify_mode(np.nan) == "unknown"
    eps = charting.ventilation_episodes(np.concatenate([np.arange(0, 10, 0.5), np.arange(30, 33, 0.5)]))
    assert eps == [(0.0, 9.5)]
