"""Synthetic-data tests for bayes_pk (no network, no PhysioNet data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_pk import bayes_forecast as bf  # noqa: E402
from bayes_pk import dosing_records as dr  # noqa: E402
from bayes_pk import evaluation as ev  # noqa: E402
from bayes_pk import pk_models as pk  # noqa: E402

COV = {"crcl_ml_min": 80.0, "weight_kg": 80.0}


def test_single_infusion_auc_equals_dose_over_cl_1cmt_and_2cmt():
    doses = pd.DataFrame({"start": [0.0], "end": [1.0], "amount_mg": [1000.0]})
    s1 = pk.illustrative_1cmt_prior()
    p1 = s1.individual_params(COV)
    auc1 = pk.auc_interval(s1, p1, doses, 0.0, 400.0, n_grid=20001)
    assert auc1 == pytest.approx(1000.0 / p1["CL"], rel=1e-3)
    s2 = pk.illustrative_icu_prior()
    p2 = s2.individual_params(COV)
    auc2 = pk.auc_interval(s2, p2, doses, 0.0, 600.0, n_grid=40001)
    assert auc2 == pytest.approx(1000.0 / p2["CL"], rel=2e-3)


def test_2cmt_concentration_continuous_and_peaks_at_end_of_infusion():
    s2 = pk.illustrative_icu_prior()
    p2 = s2.individual_params(COV)
    doses = pd.DataFrame({"start": [0.0], "end": [2.0], "amount_mg": [1500.0]})
    t = np.linspace(0, 12, 1201)
    c = pk.concentration(s2, p2, doses, t)
    assert np.all(c >= 0)
    assert abs(t[np.argmax(c)] - 2.0) < 0.02
    # continuity at the infusion end
    assert abs(pk.concentration(s2, p2, doses, [1.999])[0] - pk.concentration(s2, p2, doses, [2.001])[0]) < 0.05


def test_map_estimate_moves_towards_true_clearance():
    rng = np.random.default_rng(0)
    spec = pk.illustrative_icu_prior()
    eta_true = np.array([0.45, -0.1])  # eta order = sorted(omega) -> ("CL", "V1")
    sim = dr.simulate_course(spec, COV, eta_true, n_doses=8, interval_h=12, level_times_h=[23.5, 47.5, 71.5, 95.5], rng=rng)
    prior_cl = spec.individual_params(COV)["CL"]
    true_cl = sim["params"]["CL"]
    fit = bf.map_estimate(spec, COV, sim["doses"], sim["levels"]["time_h"], sim["levels"]["value"])
    assert fit.success
    assert abs(fit.params["CL"] - true_cl) < abs(prior_cl - true_cl)
    assert np.all(np.isfinite(fit.se_eta)) and fit.se_eta[0] < np.sqrt(spec.omega["CL"])
    # a priori fit is the prior mode
    fit0 = bf.map_estimate(spec, COV, sim["doses"], [], [])
    assert np.allclose(fit0.eta, 0.0)


def test_forecast_and_model_average_and_auc_uncertainty():
    rng = np.random.default_rng(1)
    spec = pk.illustrative_icu_prior()
    sim = dr.simulate_course(spec, COV, [0.2, 0.0], n_doses=8, level_times_h=[23.5, 47.5, 71.5], rng=rng)
    fc = bf.forecast_next_level(spec, COV, sim["doses"], sim["levels"], n_used=2)
    assert fc["time_h"] == pytest.approx(71.5)
    assert fc["predicted"] > 0
    ma = bf.model_average([spec, pk.illustrative_1cmt_prior()], COV, sim["doses"], sim["levels"]["time_h"][:2], sim["levels"]["value"][:2], [71.5])
    w = np.array(list(ma["weights"].values()))
    assert w.sum() == pytest.approx(1.0) and ma["prediction"].shape == (1,)
    fit = bf.map_estimate(spec, COV, sim["doses"], sim["levels"]["time_h"], sim["levels"]["value"])
    auc = bf.auc_with_uncertainty(spec, COV, fit, sim["doses"], 72.0, 96.0, n_draws=50, rng=rng)
    assert auc["q05"] <= auc["auc"] <= auc["q95"] or auc["sd"] > 0


def test_course_assembly_and_trough_inference():
    t0 = pd.Timestamp("2150-01-01 08:00")
    starts = [t0 + pd.Timedelta(hours=h) for h in (0, 12, 24, 24 * 7, 24 * 7 + 12)]
    doses = pd.DataFrame(
        {
            "subject_id": 1,
            "hadm_id": 10,
            "stay_id": 100,
            "start": starts,
            "end": [s + pd.Timedelta(hours=1) for s in starts],
            "amount_mg": 1000.0,
            "weight_kg": 80.0,
            "source": "inputevents",
        }
    )
    dc = dr.assemble_courses(doses, gap_hours=72)
    assert dc["course_id"].nunique() == 2
    assert dc["time_h"].iloc[0] == 0.0 and dc["time_h"].iloc[3] == 0.0
    levels = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 10, 10],
            "charttime": [t0 + pd.Timedelta(hours=23.5), t0 + pd.Timedelta(hours=12.5), t0 + pd.Timedelta(days=30)],
            "value_mg_L": [14.0, 30.0, 9.0],
            "itemid": [1, 1, 1],
        }
    )
    lv = dr.attach_levels(dc, levels)
    assert len(lv) == 2  # the day-30 level is outside any course tail
    trough = lv[lv["time_h"] == 23.5].iloc[0]
    assert bool(trough["trough_inferred"]) and not bool(trough["during_infusion"])
    infusion = lv[lv["time_h"] == 12.5].iloc[0]
    assert bool(infusion["during_infusion"])


def test_itemid_resolution_and_inputevents_unit_conversion():
    d_items = pd.DataFrame({"itemid": [1, 2, 3], "label": ["Vancomycin", "Vancomycin (Oral)", "Propofol"]})
    hits = dr.resolve_itemids(d_items, r"^vancomycin$")
    assert hits["itemid"].tolist() == [1]
    ie = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [1, 1],
            "stay_id": [1, 1],
            "itemid": [1, 1],
            "starttime": ["2150-01-01 00:00", "2150-01-01 12:00"],
            "endtime": ["2150-01-01 01:00", "2150-01-01 13:00"],
            "amount": [1.0, 1000.0],
            "amountuom": ["g", "mg"],
            "patientweight": [80.0, 80.0],
        }
    )
    out = dr.doses_from_inputevents(ie, [1])
    assert out["amount_mg"].tolist() == [1000.0, 1000.0]


def test_metrics_target_attainment_and_renal_equations():
    m = ev.prediction_errors([10, 20, 30], [11, 18, 33])
    assert m["n"] == 3 and abs(m["rbias_pct"]) < 10 and m["rrmse_pct"] > 0
    ci = ev.cluster_bootstrap([10, 20, 30, 40], [11, 18, 33, 39], [1, 1, 2, 2], n_boot=50, rng=np.random.default_rng(0))
    assert (ci["ci_lo"] <= ci["ci_hi"]).all()
    ta = ev.target_attainment([350, 450, 550, 700])
    assert ta["in_range"] == pytest.approx(0.5) and ta["below"] == pytest.approx(0.25)
    assert ev.ckd_epi_2021(1.0, 50, female=False) == pytest.approx(91.7, abs=1.0)
    assert ev.cockcroft_gault(1.0, 40, female=True, weight_kg=72) == pytest.approx(85.0, abs=0.5)
    assert ev.kdigo_stage_from_creatinine(1.0, 2.1) == 2 and ev.kdigo_stage_from_creatinine(1.0, 1.1, 0.35) == 1


def test_timing_perturbation_experiment_shape():
    rng = np.random.default_rng(2)
    spec = pk.illustrative_1cmt_prior()
    sim = dr.simulate_course(spec, COV, [0.1, 0.0], n_doses=6, level_times_h=[23.5, 47.5], rng=rng)
    res = ev.timing_perturbation_experiment(spec, COV, sim["doses"], sim["levels"], sd_minutes=30, n_rep=5, rng=rng)
    assert len(res) == 6 and res.loc[0, "rel_change"] == 0.0
    assert np.all(np.isfinite(res["auc24"]))
