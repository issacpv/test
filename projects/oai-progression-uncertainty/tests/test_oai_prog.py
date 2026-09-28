"""Synthetic-data tests for oai_prog."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oai_prog import cohort, survival, uncertainty


def test_cohort_events_and_folds():
    readings, tkr, fu = cohort.simulate_oai_like(n_participants=200, seed=0)
    long = cohort.knee_long_table(readings)
    assert set(long["VISIT"]).issubset(cohort.IMAGING_VISITS)
    prog = cohort.progression_time(long)
    death = fu[["ID", "death_months"]]
    cr = cohort.competing_risk_table(prog, tkr=tkr, death=death)
    assert set(cr["event"]).issubset({0, 1, 2, 3})
    assert (cr["kl0"] < 4).all()
    assert (cr["time_months"] > 0).all()
    # a knee with TKR before progression must be coded as TKR
    if len(tkr):
        k = cohort.knee_id(tkr).iloc[0]
        row = cr[cr["knee"] == k]
        if len(row) and np.isnan(row["prog_months"].iloc[0]):
            assert row["event"].iloc[0] == cohort.EVENT_TKR
    dis = cohort.reader_disagreement(readings)
    assert dis["n_readers"].max() >= 2 and dis["disagree"].any()
    folds = cohort.participant_grouped_folds(cr["ID"].to_numpy(), n_folds=5)
    for pid in cr["ID"].unique()[:20]:
        assert len(set(folds[cr["ID"].to_numpy() == pid])) == 1
    with pytest.raises(ValueError):
        cohort.knee_long_table(pd.DataFrame({"ID": [1], "SIDE": [1], "VISIT": ["V99"], "KL": [1]}))


def test_explicit_event_priority():
    prog = pd.DataFrame({"knee": ["1_1", "2_1", "3_1"], "ID": [1, 2, 3], "SIDE": [1, 1, 1], "kl0": [2, 2, 2],
                         "prog_months": [48.0, np.nan, 24.0], "last_visit_months": [96.0, 72.0, 96.0]})
    tkr = pd.DataFrame({"ID": [1, 3], "SIDE": [1, 1], "tkr_months": [48.0, 12.0]})
    death = pd.DataFrame({"ID": [2], "death_months": [60.0]})
    cr = cohort.competing_risk_table(prog, tkr, death)
    assert list(cr["event"]) == [cohort.EVENT_TKR, cohort.EVENT_DEATH, cohort.EVENT_TKR]
    assert list(cr["time_months"]) == [48.0, 60.0, 12.0]


def test_uncertainty_decomposition_and_conformal():
    rng = np.random.default_rng(1)
    n, k, m = 600, 3, 5
    logits = rng.normal(size=(n, k)) * 2
    y = np.array([rng.choice(k, p=np.exp(l) / np.exp(l).sum()) for l in logits])
    members = logits[None] + rng.normal(0, 0.3, size=(m, n, k))
    probs = np.exp(members) / np.exp(members).sum(-1, keepdims=True)
    dec = uncertainty.ensemble_decomposition(probs)
    assert dec["mean"].shape == (n, k)
    assert np.all(dec["epistemic"] >= -1e-9) and np.all(dec["total"] >= dec["aleatoric"] - 1e-9)
    ece = uncertainty.expected_calibration_error(dec["mean"], y)
    assert 0 <= ece < 0.15  # probabilities were generated from the same logits, so roughly calibrated
    cal, test = np.arange(300), np.arange(300, n)
    thr = uncertainty.mondrian_conformal_thresholds(dec["mean"][cal], y[cal], alpha=0.1)
    sets = uncertainty.conformal_prediction_sets(dec["mean"][test], thr)
    cov = uncertainty.coverage_and_size(sets, y[test])
    assert cov["coverage"] >= 0.85 and 1 <= cov["mean_size"] <= 3
    # readers disagree more often where epistemic uncertainty is high (noise scaled to the MI spread)
    noisy = dec["epistemic"] + rng.normal(0, 0.3 * dec["epistemic"].std(), n)
    disagree = (noisy > np.percentile(noisy, 70)).astype(int)
    assert uncertainty.uncertainty_vs_disagreement_auroc(dec["epistemic"], disagree) > 0.8
    groups = rng.choice(["F", "M"], n)
    df = uncertainty.subgroup_metrics(dec["mean"], y, groups, sets=uncertainty.conformal_prediction_sets(dec["mean"], thr))
    assert df["group"].iloc[-1] == "MAX_GAP" and len(df) == 3
    with pytest.raises(ValueError):
        uncertainty.ensemble_decomposition(dec["mean"])


def test_survival_metrics():
    rng = np.random.default_rng(2)
    n = 800
    risk = rng.normal(size=n)
    times, events = survival.simulate_competing_risks(n, risk, seed=3)
    grid = np.array([12, 24, 48, 96], float)
    cif1 = survival.aalen_johansen(times, events, 1, grid)
    cif2 = survival.aalen_johansen(times, events, 2, grid)
    cif3 = survival.aalen_johansen(times, events, 3, grid)
    assert np.all(np.diff(cif1) >= 0) and cif1[-1] + cif2[-1] + cif3[-1] <= 1.0 + 1e-9
    km = survival.kaplan_meier(times, events > 0, grid)
    assert np.all(np.diff(km) <= 0) and km[-1] == pytest.approx(1 - (cif1[-1] + cif2[-1] + cif3[-1]), abs=1e-6)
    c = survival.cause_specific_cindex(risk, times, events, cause=1)
    c_null = survival.cause_specific_cindex(rng.permutation(risk), times, events, cause=1)
    assert c > 0.6 > c_null - 0.05
    # informative CIF prediction should beat a constant at the marginal rate
    pred_good = 1 - np.exp(-0.01 * np.exp(risk) * 48)
    pred_const = np.full(n, cif1[2])
    b_good = survival.ipcw_brier(pred_good, times, events, 1, 48)
    b_const = survival.ipcw_brier(pred_const, times, events, 1, 48)
    assert b_good < b_const
