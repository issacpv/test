"""Synthetic-data tests for stagebias."""
import numpy as np
import pandas as pd
import pytest

from stagebias import error_structure as es
from stagebias import features as ft
from stagebias import hypnogram as hy
from stagebias import measurement_error as me

PROFUSION = """<?xml version="1.0" encoding="utf-8"?>
<CMPStudyConfig>
  <EpochLength>30</EpochLength>
  <ScoredEvents>
    <ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept>
      <Start>65.0</Start><Duration>20.0</Duration><SignalLocation>AIRFLOW</SignalLocation></ScoredEvent>
    <ScoredEvent><EventType>Arousals|Arousals</EventType><EventConcept>Arousal|Arousal ()</EventConcept>
      <Start>84.0</Start><Duration>5.0</Duration><SignalLocation>EEG</SignalLocation></ScoredEvent>
  </ScoredEvents>
  <SleepStages>
    <SleepStage>0</SleepStage><SleepStage>1</SleepStage><SleepStage>2</SleepStage><SleepStage>3</SleepStage>
    <SleepStage>4</SleepStage><SleepStage>5</SleepStage><SleepStage>9</SleepStage>
  </SleepStages>
</CMPStudyConfig>
"""


def test_parse_profusion_xml_stages_and_events():
    stages, events = hy.parse_profusion_xml(PROFUSION)
    assert stages.tolist() == [0, 1, 2, 3, 3, 4, -1]
    assert len(events.table) == 2
    assert len(events.respiratory()) == 1
    assert len(events.arousals()) == 1
    mask = hy.events_to_epoch_mask(events.respiratory(), n_epochs=7, pad_epochs=0)
    # hypopnoea 65-85 s overlaps epochs 2 (60-90 s) only
    assert mask.tolist() == [False, False, True, False, False, False, False]
    mask1 = hy.events_to_epoch_mask(events.respiratory(), n_epochs=7, pad_epochs=1)
    assert mask1.tolist() == [False, True, True, True, False, False, False]


def test_annotations_to_epochs_and_fractions():
    stages = hy.annotations_to_epochs([0, 60, 120, 180], [60, 60, 60, 30],
                                      ["Sleep stage W", "Sleep stage 2", "Sleep stage 4", "Sleep stage R"])
    assert stages.tolist() == [0, 0, 2, 2, 3, 3, 4]
    fr = hy.stage_fractions(stages)
    assert fr["N3"] == pytest.approx(2 / 5)
    assert hy.n3_percent(stages) == pytest.approx(40.0)
    P = np.zeros((7, 5))
    P[np.arange(7), stages] = 1.0
    assert hy.expected_stage_fractions(P)["N3"] == pytest.approx(0.4)


def test_features_and_stager_learn_delta_vs_sigma():
    rng = np.random.default_rng(1)
    fs, n_epochs = 100.0, 120
    t = np.arange(int(fs * 30)) / fs
    labels = rng.choice([2, 3], size=n_epochs)
    sig = []
    for lab in labels:
        base = rng.normal(0, 1, len(t))
        if lab == 3:
            base += 4.0 * np.sin(2 * np.pi * 1.0 * t + rng.uniform(0, 2 * np.pi))
        else:
            base += 2.0 * np.sin(2 * np.pi * 13.0 * t + rng.uniform(0, 2 * np.pi))
        sig.append(base)
    eeg = np.concatenate(sig)
    X, names = ft.features_from_record(eeg, fs)
    assert X.shape == (n_epochs, len(names))
    assert "log_sigma_delta_ratio" in names
    model = ft.StagingBaseline(max_iter=50).fit(X[:80], labels[:80])
    P = model.predict_proba(X[80:])
    assert P.shape == (40, 5)
    assert np.allclose(P.sum(axis=1), 1.0)
    acc = (model.predict(X[80:]) == labels[80:]).mean()
    assert acc > 0.9


def test_confusion_kappa_and_stage_metrics():
    y = np.array([0, 1, 2, 3, 4] * 20)
    yp = y.copy()
    yp[3::5] = 2  # every N3 -> N2
    cm = es.confusion_matrix(y, yp)
    assert cm[3, 2] == pytest.approx(1.0)
    m = es.stage_metrics(y, yp)
    assert m.loc[m.stage == "N3", "sensitivity"].item() == pytest.approx(0.0)
    assert 0 < m.attrs["kappa"] < 1
    assert es.cohens_kappa(y, y) == pytest.approx(1.0)
    assert es.ahi_stratum(np.array([1, 7, 20, 40])).tolist() == [0, 1, 2, 3]


def test_event_locked_error_and_differential_test():
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.choice([1, 2, 3], size=n)
    ev = rng.random(n) < 0.2
    err = np.where(ev, rng.random(n) < 0.4, rng.random(n) < 0.1)
    yp = np.where(err, (y + 1) % 5, y)
    res = es.event_locked_error(y, yp, ev)
    assert res["odds_ratio"] > 3.0 and res["ci_low"] > 1.0
    df = pd.DataFrame({"y_true": y, "y_pred": yp, "record": rng.integers(0, 40, n),
                       "ahi": rng.normal(15, 5, n), "hf": ev.astype(float)})
    out = es.differential_misclassification_test(df, ["ahi", "hf"], true_stage=3)
    assert out.loc["hf", "coef"] > 0 and out.loc["hf", "p"] < 0.01
    stat, lo, hi = es.bootstrap_records(es.cohens_kappa, y, yp, df["record"].to_numpy(), n_boot=50)
    assert lo <= stat <= hi


def test_n3_bias_and_bland_altman():
    rng = np.random.default_rng(3)
    stages = np.concatenate([hy.synthetic_hypnogram(200, rng) for _ in range(3)])
    rec = np.repeat([0, 1, 2], 200)
    pred = stages.copy()
    pred[(stages == 3) & (rng.random(len(stages)) < 0.5)] = 2
    tab = es.n3_bias_per_record(stages, pred, rec)
    assert (tab["diff"] <= 0).all()
    ba = es.bland_altman(tab["n3_true"], tab["n3_pred"])
    assert ba["bias"] < 0 and ba["n"] == 3


def test_regression_calibration_and_simex_recover_attenuated_effect():
    rng = np.random.default_rng(4)
    beta = 0.8
    df = me.simulate_cohort(4000, beta=beta, sigma_u=1.0, rng=rng)
    fit = lambda x, d: me.logistic_log_or(x, d)  # noqa: E731
    truth = fit(df["x_true"].to_numpy(), df)
    naive = fit(df["x_err"].to_numpy(), df)
    assert naive < 0.8 * truth  # attenuation
    rc = me.regression_calibration(fit, df, "x_err", "x_gold")
    assert abs(rc["corrected"] - truth) < abs(rc["naive"] - truth)
    s2 = me.estimate_error_variance(df["x_err"], df["x_gold"])
    assert 0.6 < s2 < 1.5
    sx = me.simex(fit, df, "x_err", sigma_u2=s2, n_sim=10, rng=rng)
    assert abs(sx["corrected"] - truth) < abs(sx["naive"] - truth)
    swap = me.swap_in_comparison(fit, df, {"true": "x_true", "err": "x_err"}, n_boot=20, rng=rng)
    assert swap.loc[swap.exposure == "err", "diff_vs_ref"].item() < 0
    null = me.classical_error_null(fit, df, "x_true", s2, n_draws=5, rng=rng)
    assert null.shape == (5,)


def test_cox_log_hr_runs_on_synthetic_survival():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.5 * x))
    c = rng.exponential(2.0, n)
    df = pd.DataFrame({"time": np.minimum(t, c), "event": (t <= c).astype(int), "age": rng.normal(60, 10, n)})
    b = me.cox_log_hr(x, df, covariates=["age"])
    assert 0.2 < b < 0.9
