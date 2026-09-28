"""Synthetic-data tests for shift decomposition, calibration and fairness utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icu_transport import calibration, cohorts, fairness, shift  # noqa: E402


def _sim(n: int, rng: np.random.Generator, beta: np.ndarray, shift_x: float = 0.0, prior_logit: float = 0.0):
    X = rng.normal(shift_x, 1.0, size=(n, len(beta)))
    logits = X @ beta + prior_logit
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def test_weighted_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500) + 0.3 * y
    assert abs(shift.weighted_auroc(y, p) - roc_auc_score(y, p)) < 1e-9
    w = rng.uniform(0.5, 2.0, 500)
    assert abs(shift.weighted_auroc(y, p, w) - roc_auc_score(y, p, sample_weight=w)) < 1e-9


def test_bbse_recovers_induced_label_shift():
    rng = np.random.default_rng(1)
    beta = np.array([1.5, -1.0, 0.5])
    Xs, ys = _sim(6000, rng, beta)
    # induce pure label shift on the target by subsampling negatives
    Xt, yt = _sim(6000, rng, beta)
    keep = (yt == 1) | (rng.uniform(size=len(yt)) < 0.4)
    Xt, yt = Xt[keep], yt[keep]
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression().fit(Xs[:3000], ys[:3000])
    ps = clf.predict_proba(Xs[3000:])[:, 1]
    pt = clf.predict_proba(Xt)[:, 1]
    ls = shift.bbse_label_shift(ys[3000:], ps >= 0.5, pt >= 0.5)
    assert abs(ls.target_prior[1] - yt.mean()) < 0.05


def test_domain_classifier_detects_covariate_shift_and_not_null():
    rng = np.random.default_rng(2)
    Xs = rng.normal(0, 1, (1500, 4))
    Xt = rng.normal(1.0, 1, (1500, 4))
    cs = shift.covariate_shift(Xs, Xt)
    assert cs.domain_auroc > 0.8
    assert cs.effective_sample_size < 1500
    X0 = rng.normal(0, 1, (1500, 4))
    cs0 = shift.covariate_shift(Xs, X0)
    assert 0.4 < cs0.domain_auroc < 0.6


def test_decomposition_negative_control_is_near_zero():
    rng = np.random.default_rng(3)
    beta = np.array([1.0, -1.0])
    X, y = _sim(8000, rng, beta)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression().fit(X[:2000], y[:2000])
    Xa, ya, Xb, yb = X[2000:5000], y[2000:5000], X[5000:], y[5000:]
    res = shift.full_decomposition(lambda Z: clf.predict_proba(Z)[:, 1], Xa, ya, Xb, yb, metrics=("auroc", "brier"))
    for m in res.values():
        assert abs(m.total_gap) < 0.03
        assert abs(m.covariate) < 0.03 and abs(m.label) < 0.03 and abs(m.concept) < 0.03


def test_decomposition_attributes_pure_label_shift_to_label_component():
    rng = np.random.default_rng(4)
    beta = np.array([1.2, -0.8])
    Xs, ys = _sim(8000, rng, beta, prior_logit=-1.0)
    Xt, yt = _sim(8000, rng, beta, prior_logit=1.0)  # same P(x); P(y|x) differs only by prior... not pure
    # construct pure label shift instead: resample target rows by class
    keep = (yt == 1) | (rng.uniform(size=len(yt)) < 0.3)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression().fit(Xs[:3000], ys[:3000])
    Xt2, yt2 = Xs[3000:], ys[3000:]
    keep2 = (yt2 == 1) | (rng.uniform(size=len(yt2)) < 0.3)
    res = shift.full_decomposition(lambda Z: clf.predict_proba(Z)[:, 1], Xs[3000:6000], ys[3000:6000],
                                   Xt2[keep2], yt2[keep2], metrics=("citl",))["citl"]
    # calibration-in-the-large gap should be mostly explained by the label component
    assert abs(res.total_gap) > 0.5
    assert abs(res.label) > 0.5 * abs(res.total_gap)


def test_calibration_slope_intercept_on_calibrated_and_miscalibrated():
    rng = np.random.default_rng(5)
    p = rng.beta(2, 5, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    r = calibration.calibration_slope_intercept(y, p)
    assert abs(r["intercept"]) < 0.1 and abs(r["slope"] - 1) < 0.1
    lp = np.log(p / (1 - p))
    p_over = 1 / (1 + np.exp(-(2 * lp + 1.0)))  # over-confident + shifted
    r2 = calibration.calibration_slope_intercept(y, p_over)
    assert r2["slope"] < 0.7 and r2["intercept"] < -0.3
    assert calibration.expected_calibration_error(y, p) < calibration.expected_calibration_error(y, p_over)


def test_recalibration_methods_restore_calibration():
    rng = np.random.default_rng(6)
    p = rng.beta(2, 5, 6000)
    y = (rng.uniform(size=6000) < p).astype(int)
    lp = np.log(p / (1 - p))
    bad = 1 / (1 + np.exp(-(1.5 * lp + 0.8)))
    base = calibration.calibration_report(y[3000:], bad[3000:])
    for m in ("intercept", "platt", "temperature", "isotonic"):
        rc = calibration.fit_recalibrator(bad[:3000], y[:3000], m)
        q = rc(bad[3000:])
        rep = calibration.calibration_report(y[3000:], q)
        # each method is asserted on the quantity it is designed to fix
        if m in ("intercept", "platt", "isotonic"):
            assert abs(rep["intercept"]) < abs(base["intercept"])
        if m in ("temperature", "platt", "isotonic"):
            assert abs(rep["slope"] - 1) < abs(base["slope"] - 1)
        if m in ("platt", "isotonic"):
            assert rep["ece"] < base["ece"]
    curve = pd.DataFrame(calibration.few_shot_learning_curve(bad, y, n_grid=(50, 200), n_repeats=3))
    assert set(curve["method"]) >= {"intercept", "platt"}
    assert curve.loc[curve.n == 200, "ece"].mean() <= curve.loc[curve.n == 50, "ece"].mean() + 0.02


def test_decision_curve_beats_treat_all_for_informative_model():
    rng = np.random.default_rng(7)
    p = rng.beta(2, 8, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    dc = calibration.decision_curve(y, p, thresholds=np.array([0.1, 0.2, 0.3]))
    assert np.all(dc["net_benefit"] >= dc["treat_all"] - 1e-9)


def test_subgroup_metrics_and_permutation_null():
    rng = np.random.default_rng(8)
    n = 3000
    g = rng.choice(["F", "M"], n)
    p = rng.beta(2, 6, n)
    # miscalibrate one group
    lp = np.log(p / (1 - p))
    p_obs = np.where(g == "F", 1 / (1 + np.exp(-(lp + 1.0))), p)
    y = (rng.uniform(size=n) < p).astype(int)
    thr = fairness.operating_threshold(p_obs, 0.1)
    t = fairness.subgroup_metrics(y, p_obs, g, thr, n_boot=20)
    assert set(t.index) == {"F", "M"}
    gaps = fairness.parity_gaps(t)
    assert gaps["intercept"] > 0.5
    res = fairness.permutation_gap_test(y, p_obs, g, thr, metric="intercept", n_perm=50)
    assert res["p_value"] < 0.1


def test_cohort_sql_renders_and_labels_work():
    for db in cohorts.DBS:
        sql = cohorts.stay_sql(db, "/data/root")
        assert "stay_id" in sql and "death_hours" in sql
    sql = cohorts.concept_sql("mimiciv", "creatinine", "/data/root", max_hours=48)
    assert "labevents" in sql and "50912" in sql
    with pytest.raises(RuntimeError):
        cohorts.concept_sql("hirid", "creatinine", "/data/root")  # unverified ids must be explicit
    assert "20000600" in cohorts.concept_sql("hirid", "creatinine", "/data/root", allow_unverified=True)

    stays = pd.DataFrame({"stay_id": [1, 2, 3], "death_hours": [30.0, 100.0, np.nan]})
    y = cohorts.label_mortality(stays, 24, 48)
    assert y.tolist() == [1, 0, 0]

    creat = pd.DataFrame({"stay_id": [1] * 4 + [2] * 3, "hours": [1, 20, 40, 60, 1, 30, 50],
                          "value": [1.0, 1.0, 1.6, 1.7, 0.8, 0.9, 0.85]})
    st = cohorts.kdigo_creatinine_stage(creat)
    assert st.loc[(st.stay_id == 1) & (st.hours == 40), "stage"].item() == 1
    aki = cohorts.label_aki(creat, None, None, pred_hour=24, horizon_h=48)
    assert aki.loc[1] == 1 and aki.loc[2] == 0

    abx = pd.DataFrame({"stay_id": [1, 2], "hours": [10.0, 10.0]})
    cul = pd.DataFrame({"stay_id": [1, 2], "hours": [12.0, 200.0]})
    ts = cohorts.suspected_infection_time(abx, cul)
    assert ts.loc[1] == 10.0 and 2 not in ts.index

    idx = pd.MultiIndex.from_product([[1], range(48)], names=["stay_id", "hour"])
    sofa = pd.Series(np.r_[np.zeros(30), np.full(18, 3.0)], index=idx)
    onset = cohorts.sepsis3_onset(sofa, ts)
    assert onset.loc[1] == 30.0
    lab = cohorts.label_sepsis(onset, [1, 2], pred_hour=24, horizon_h=12)
    assert lab.loc[1] == 1 and lab.loc[2] == 0


def test_hourly_grid_and_sofa():
    stays = pd.DataFrame({"stay_id": [1]})
    long = pd.DataFrame({
        "stay_id": [1, 1, 1, 1, 1],
        "hours": [0.5, 2.2, 1.0, 3.0, 0.0],
        "value": [80.0, 90.0, 60.0, 250.0, 0.2],
        "end_hours": [np.nan, np.nan, np.nan, np.nan, 5.0],
        "concept": ["hr", "hr", "map", "platelets", "norepinephrine"],
    })
    grid = cohorts.hourly_grid(long, stays, n_hours=6)
    assert grid.loc[(1, 0), "hr"] == 80.0 and grid.loc[(1, 1), "hr"] == 80.0  # carried forward
    assert grid.loc[(1, 2), "hr"] == 90.0 and grid.loc[(1, 0), "hr__n"] == 1
    assert grid.loc[(1, 4), "norepinephrine"] == 0.2 and grid.loc[(1, 5), "norepinephrine"] == 0.0
    sofa = cohorts.sofa_hourly(grid)
    assert sofa.loc[(1, 1)] == 4  # MAP<70 but pressor at 0.2 -> cardio 4
    feats = cohorts.window_features(grid, 0, 6)
    assert "hr__mean" in feats and "hr__count" in feats and feats.loc[1, "hr__count"] == 2
