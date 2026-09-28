"""Synthetic-data tests for cardiac_twin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiac_twin import data, ecg_features, evaluation, twin_model


def test_synthetic_ecg_features_track_parameters():
    fs = 500
    sig, leads = ecg_features.synthesize_12lead(fs=fs, hr_bpm=72, qrs_ms=90, r_amp_mv=1.0, s_amp_mv=1.0, seed=0)
    assert sig.shape == (fs * 10, 12)
    f = ecg_features.extract_features(sig, leads, fs)
    assert 65 < f.hr_bpm < 80
    assert 60 < f.qrs_ms < 130
    assert f.n_beats >= 10
    # wider QRS and larger voltages should be recovered monotonically
    sig2, _ = ecg_features.synthesize_12lead(fs=fs, hr_bpm=72, qrs_ms=150, r_amp_mv=1.8, s_amp_mv=1.8, seed=0)
    f2 = ecg_features.extract_features(sig2, leads, fs)
    assert f2.qrs_ms > f.qrs_ms
    assert f2.sokolow_lyon_mv > f.sokolow_lyon_mv
    # axis shift rightward increases the measured axis
    sig3, _ = ecg_features.synthesize_12lead(fs=fs, axis_shift_deg=40, seed=0)
    f3 = ecg_features.extract_features(sig3, leads, fs)
    assert f3.axis_deg > f.axis_deg


def test_pairing_and_patient_split():
    ecg = pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "ecg_time": ["2020-01-01", "2020-01-20", "2020-02-01", "2020-03-01"],
    })
    echo = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "echo_time": ["2020-01-03", "2020-02-20", "2020-03-02"],
    })
    pairs = data.pair_ecg_echo(ecg, echo, max_days=7)
    assert len(pairs) == 2  # subject 2's ECG is 19 days away
    assert set(pairs["subject_id"]) == {1, 3}
    row = pairs[pairs["subject_id"] == 1].iloc[0]
    assert row["ecg_index"] == 0 and row["delay_days"] == pytest.approx(-2.0)
    split = data.patient_level_split(np.array([1, 1, 2, 2, 3, 4, 5, 6, 7, 8]), seed=1)
    for s in np.unique([1, 2]):
        assert len(set(split[np.array([1, 1, 2, 2, 3, 4, 5, 6, 7, 8]) == s])) == 1


def test_unpaired_aligner_recovers_ef_and_counterfactuals():
    rng = np.random.default_rng(0)
    echo_cohort = twin_model.sample_prior(3000, rng)
    echo_obs = twin_model.echo_forward(echo_cohort, rng)
    ecg_cohort = twin_model.sample_prior(1500, rng)
    ecg_obs = twin_model.ecg_forward(ecg_cohort, rng)
    aligner = twin_model.UnpairedLatentAligner(n_sim=8000, seed=1).fit(echo_obs["ef"].to_numpy())
    pred = aligner.predict_state(ecg_obs)
    r = np.corrcoef(pred["lvef_calibrated"], ecg_cohort["lvef"])[0, 1]
    assert r > 0.45  # EF is only indirectly observed by ECG, so moderate correlation is expected
    # calibrated EF should match the echo EF distribution
    assert abs(pred["lvef_calibrated"].mean() - echo_obs["ef"].mean()) < 1.0
    # counterfactual: lowering EF should lengthen QRS; more mass should raise voltage
    base = twin_model.sample_prior(200, rng)
    f0 = twin_model.ecg_forward(base, np.random.default_rng(5))
    _, f1 = twin_model.counterfactual(base, np.random.default_rng(5), lvef=-20)
    assert (f1["qrs_ms"] - f0["qrs_ms"]).mean() > 3
    _, f2 = twin_model.counterfactual(base, np.random.default_rng(5), lv_mass_g=100)
    assert (f2["sokolow_lyon_mv"] - f0["sokolow_lyon_mv"]).mean() > 0.5
    with pytest.raises(KeyError):
        twin_model.counterfactual(base, rng, not_a_state=1.0)


def test_evaluation_utils():
    rng = np.random.default_rng(2)
    y = rng.normal(55, 12, 400)
    yhat = y + rng.normal(0, 6, 400)
    m = evaluation.regression_metrics(y, yhat)
    assert 4 < m["mae"] < 8 and m["r"] > 0.8
    c = evaluation.classification_metrics((y < 40).astype(int), -yhat)
    assert c["auroc"] > 0.85
    groups = rng.integers(0, 80, 400)
    est, lo, hi = evaluation.cluster_bootstrap(lambda a, b: evaluation.regression_metrics(a, b)["mae"], y, yhat, groups, n_boot=100)
    assert lo <= est <= hi
    unc = np.abs(yhat - y) + rng.normal(0, 2, 400)
    assert evaluation.discordance_auroc(np.abs(yhat - y), unc, threshold=8) > 0.7
    assert evaluation.interval_coverage(y, yhat - 20, yhat + 20) > 0.95
    assert evaluation.interval_score(y, yhat - 20, yhat + 20) >= 40
    deltas = np.column_stack([rng.normal(5, 1, 50), rng.normal(-3, 1, 50), rng.normal(0, 1, 50)])
    obs, p = evaluation.permutation_null_consistency(deltas, np.array([1, -1, 0]), n_perm=200)
    assert obs > 0.95 and p < 0.05
