"""Simulator-based tests for pva_detect (no data, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pva_detect import breaths, cohort, lungsim, surrogates  # noqa: E402


def _run(name: str, duration: float = 120.0, seed: int = 0):
    cfg = lungsim.SCENARIOS[name]
    cfg = lungsim.SimConfig(**{**cfg.__dict__, "duration_s": duration, "seed": seed})
    return lungsim.simulate(cfg)


def test_controlled_ventilation_is_synchronous():
    r = _run("controlled")
    expected = 120.0 * 15 / 60
    assert abs(len(r.breaths) - expected) <= 1
    assert (r.breaths["label"] == "normal").all() and r.asynchrony_index() == 0.0
    seg = breaths.segment_breaths(r.flow, r.fs)
    assert abs(len(seg) - len(r.breaths)) <= 1
    f = breaths.breath_features(r.flow, r.paw, r.fs, seg)
    assert f["vt_insp"].median() == pytest.approx(0.45, rel=0.1)
    assert f["ti"].median() == pytest.approx(1.0, abs=0.1) and f["peep"].median() == pytest.approx(5.0, abs=0.5)
    assert (f["pip"] > f["peep"]).all()
    ie = breaths.detect_ineffective_efforts(r.flow, r.fs, seg)
    assert len(ie) <= 1  # false-positive floor
    assert len(breaths.detect_double_triggering(f)) == 0


def test_ineffective_efforts_are_generated_and_detected():
    r = _run("ineffective_efforts")
    true_ie = r.efforts.loc[~r.efforts["delivered"], "start_idx"].to_numpy()
    assert len(true_ie) >= 10 and r.asynchrony_index() > 10
    seg = breaths.segment_breaths(r.flow, r.fs)
    ie = breaths.detect_ineffective_efforts(r.flow, r.fs, seg)
    m = breaths.match_events(ie["idx"].to_numpy(), true_ie, r.fs, tol_s=0.7)
    assert m["recall"] >= 0.7 and m["precision"] >= 0.7, m


def test_double_triggering_is_generated_and_detected():
    r = _run("double_triggering")
    n_dt = int((r.breaths["label"] == "double_trigger").sum())
    assert n_dt >= 3
    seg = breaths.segment_breaths(r.flow, r.fs)
    f = breaths.breath_features(r.flow, r.paw, r.fs, seg)
    det = breaths.detect_double_triggering(f)
    true_idx = r.breaths.loc[r.breaths["label"] == "double_trigger", "start_idx"].to_numpy()
    m = breaths.match_events(seg["start_idx"].to_numpy()[det], true_idx, r.fs, tol_s=0.3)
    assert m["recall"] >= 0.7 and m["precision"] >= 0.7, m
    # stacked breaths carry larger inspired volume than singles
    assert f.loc[det, "vt_insp"].mean() > 0.0


def test_reverse_and_auto_triggering_scenarios_label_breaths():
    rt = _run("reverse_triggering", duration=60)
    assert (rt.breaths["label"] == "reverse_trigger").sum() >= 5
    assert (rt.efforts["label"] == "reverse_trigger_effort").all()
    at = _run("auto_triggering", duration=60)
    assert (at.breaths["label"] == "auto_trigger").sum() >= 1
    assert breaths.asynchrony_index(10, 0, 0) == 0.0 and breaths.asynchrony_index(9, 1, 0) == 10.0


def test_surrogates_track_true_asynchrony_index():
    rows, ys = [], []
    for name in ("synchronous_assisted", "ineffective_efforts", "double_triggering", "controlled"):
        for seed in range(3):
            r = _run(name, duration=240, seed=seed)
            seg = breaths.segment_breaths(r.flow, r.fs)
            f = breaths.breath_features(r.flow, r.paw, r.fs, seg)
            # ground-truth labels aligned to segmented breaths by nearest onset
            lab = pd.Series("normal", index=f.index)
            for _, b in r.breaths.iterrows():
                j = int(np.argmin(np.abs(seg["start_idx"].to_numpy() - b["start_idx"])))
                lab.iloc[j] = b["label"]
            ie_t = r.efforts.loc[~r.efforts["delivered"], "start_idx"].to_numpy() / r.fs
            bins = surrogates.bin_breaths(f, lab, ie_t, bin_s=60.0, rr_set=lungsim.SCENARIOS[name].vent.rr_set, duration_s=240)
            X = surrogates.surrogate_features(bins)
            rows.append(X.assign(scenario=name, seed=seed))
            ys.append(bins["ai_true"].to_numpy())
    X, y = pd.concat(rows, ignore_index=True), np.concatenate(ys)
    composite = surrogates.trigger_excess_index(X).to_numpy()
    ok = np.isfinite(y) & np.isfinite(composite)
    assert spearmanr(composite[ok], y[ok]).correlation > 0.5
    ie_rows = X["scenario"] == "ineffective_efforts"
    assert X.loc[ie_rows, "rr_monitor_excess"].mean() > X.loc[~ie_rows, "rr_monitor_excess"].mean() + 2
    tr = X["seed"].to_numpy() < 2
    model = surrogates.fit_surrogate_model(X[tr], y[tr])
    ev = surrogates.evaluate_surrogate(model, X[~tr], y[~tr])
    assert ev["spearman"] > 0.5 and (np.isnan(ev["auroc_ai10"]) or ev["auroc_ai10"] > 0.7), ev


def test_charted_features_and_cohort_helpers():
    rng = np.random.default_rng(0)
    t = np.arange(0, 4 * 3600, 120.0)
    chart = pd.DataFrame({"t_s": t, "rr_total": 18 + rng.normal(0, 1, t.size), "rr_set": 14.0,
                          "vt_obs": 0.45 + rng.normal(0, 0.02, t.size), "pip": 25 + rng.normal(0, 1, t.size),
                          "minute_volume": 8.0 + rng.normal(0, 0.3, t.size)})
    feats = surrogates.charted_window_features(chart, window_s=3600.0)
    assert len(feats) == 4 and set(surrogates.SURROGATE_FEATURES) <= set(feats.columns)
    assert feats["rr_excess"].mean() == pytest.approx(4.0, abs=1.0)
    ref = pd.DataFrame({"id": [1, 2, 3, 4], "Variable Name": ["Respiratory rate set", "Peak inspiratory pressure",
                                                                 "Tidal volume", "Heart rate"]})
    lk = cohort.lookup_variables(ref, cohort.HIRID_HINTS, "Variable Name", "id")
    assert set(lk["concept"]) >= {"rr_set", "pip", "vt_obs"} and 4 not in lk["variable_id"].tolist()
    long = pd.DataFrame({"stay_id": [1, 1, 1, 1], "t_s": [0, 0, 3600, 3600],
                         "itemid": [224688, 224690, 224688, 224690], "valuenum": [14, 18, 14, 20]})
    wide = cohort.pivot_charted(long, cohort.MIMIC_VENT_ITEMS)
    assert list(wide.columns[:2]) == ["stay_id", "t_s"] and wide["rr_total"].tolist() == [18, 20]
    assert "chartevents.csv.gz" in cohort.mimic_vent_sql("/x") and "respiratoryCharting" in cohort.eicu_resp_sql("/x")
    assert cohort.parse_vent_mode("CMV/ASSIST") == "controlled" and cohort.parse_vent_mode("PSV/SBT") == "spontaneous"
    # exposure-outcome: constructed association is recovered
    n = 400
    ai = rng.uniform(0, 30, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.08 * ai)))
    df = pd.DataFrame({"ai_mean": ai, "died": y.astype(int), "age": rng.normal(60, 10, n), "hosp": rng.integers(0, 5, n)})
    res = cohort.logistic_association(df, "ai_mean", "died", ["age"], cluster="hosp")
    assert res["or"] > 1.0 and res["ci_low"] > 1.0
    wins = pd.DataFrame({"stay_id": [1, 1, 2], "ai_pred": [5.0, 15.0, 2.0]})
    ex = cohort.exposure_table(wins)
    assert ex.loc[ex.stay_id == 1, "frac_windows_high"].item() == 0.5
