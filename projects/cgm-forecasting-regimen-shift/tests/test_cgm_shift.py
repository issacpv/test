"""Synthetic-data tests for cgm_shift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cgm_shift import loaders, regimen, models, clinical, shift, audit


def test_synthetic_and_grid():
    df = loaders.synthetic_stream(n_hours=24, regimen="open-loop", seed=0)
    grid = loaders.to_grid(df)
    assert {"glucose", "imputed", "basal", "bolus", "carbs", "device_mode"} <= set(grid.columns)
    assert grid["glucose"].notna().all()


def test_iob_decays():
    bolus = np.zeros(20)
    bolus[0] = 5.0
    iob = loaders.insulin_on_board(bolus)
    assert iob[0] == 5.0
    assert iob[5] < iob[0] and iob[-1] < iob[5]  # monotone decay


def test_make_windows_shapes():
    df = loaders.synthetic_stream(n_hours=24, seed=1)
    grid = loaders.to_grid(df)
    w = loaders.make_windows(grid, history_len=12, horizon_min=30)
    assert w.X.shape[1] == 12
    assert w.X_exo.shape[1] == 12 and w.X_exo.shape[2] == 2
    assert len(w.y) == len(w.X) == len(w.valid)


def test_regimen_labeling():
    assert regimen.label_dataset("ohiot1dm") == "open-loop"
    assert regimen.label_dataset("azt1d") == "hybrid-closed-loop"
    assert regimen.label_from_metadata("Tandem t:slim", "Control-IQ") == "hybrid-closed-loop"
    assert regimen.label_from_metadata("DIY rig", "OpenAPS") == "DIY-loop"
    active = regimen.aid_active(np.array(["sleep", "unknown", "exercise", "manual"]))
    assert active.tolist() == [True, False, True, False]


def test_ar_ridge_beats_persistence_in_sample():
    df = loaders.synthetic_stream(n_hours=96, seed=2)
    grid = loaders.to_grid(df)
    w = loaders.make_windows(grid, history_len=12, horizon_min=30)
    n = len(w.y)
    tr, te = slice(0, n * 3 // 4), slice(n * 3 // 4, n)
    pers = models.PersistenceModel().fit(w.X[tr], w.y[tr])
    ar = models.ARRidgeModel().fit(w.X[tr], w.y[tr], w.X_exo[tr])
    rmse_p = models.rmse(w.y[te], pers.predict(w.X[te]))
    rmse_a = models.rmse(w.y[te], ar.predict(w.X[te], w.X_exo[te]))
    assert rmse_a <= rmse_p + 1e-6  # AR should not be worse than persistence


def test_hypo_warning_and_leadtime():
    y_true = np.array([65, 80, 60, 120, 68])
    y_pred = np.array([66, 82, 90, 118, 60])  # misses one true hypo, catches two
    w = clinical.hypo_warning(y_true, y_pred)
    assert w["n_events"] == 3
    assert 0 <= w["sensitivity"] <= 1
    lt = clinical.detection_lead_time(np.arange(5) * 5.0, y_true, y_pred, horizon_min=30)
    assert lt == 30.0 or np.isnan(lt)


def test_clarke_zones_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 300, 200)
    p = y + rng.normal(0, 10, 200)
    z = clinical.clarke_zones(y, p)
    assert abs(sum(z.values()) - 1.0) < 1e-9
    assert z["A"] + z["B"] > 0.5  # good predictions land in A/B


def test_marginal_shift_and_feedback():
    ol = loaders.synthetic_stream(n_hours=96, regimen="open-loop", seed=3)["glucose"].to_numpy()
    cl = loaders.synthetic_stream(n_hours=96, regimen="closed-loop", seed=3)["glucose"].to_numpy()
    ms = shift.marginal_shift(ol, cl)
    assert ms["var_ratio"] < 1.0  # closed-loop has lower variance
    # controller feedback: closed-loop glucose change more predictable than open-loop
    ins = np.zeros_like(cl)
    r2_cl = shift.controller_feedback_r2(cl, ins)
    r2_ol = shift.controller_feedback_r2(ol, np.zeros_like(ol))
    assert r2_cl >= r2_ol - 0.05


def test_variance_matched_surrogate():
    rng = np.random.default_rng(0)
    src = rng.normal(150, 40, 500)
    tgt = rng.normal(120, 15, 500)
    sur = shift.variance_matched_surrogate(src, tgt)
    assert abs(sur.std() - tgt.std()) < 1e-6
    assert abs(sur.mean() - tgt.mean()) < 1e-6


def test_audit_detects_artefacts():
    df = loaders.synthetic_stream(n_hours=24, seed=5)
    grid = loaders.to_grid(df)
    # inject a calibration jump and a flatline
    g = grid["glucose"].to_numpy().copy()
    g[50] = g[49] + 120  # sudden jump
    g[100:110] = 111.0    # flatline
    grid["glucose"] = g
    rep = audit.audit_stream(grid)
    assert rep["n_calibration_jumps"] >= 1
    assert rep["n_flatline_segments"] >= 1
    assert "imputed_fraction" in rep
