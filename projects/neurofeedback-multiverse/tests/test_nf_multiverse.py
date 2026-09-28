"""Tests on synthetic EEG: feedback recomputation, learning indices, artifact specificity, spec-curve test."""
from __future__ import annotations

import numpy as np
import pytest

from nf_multiverse import (
    FeedbackSpec,
    artifact_regressors,
    build_specification_space,
    classify_learner,
    cohen_kappa,
    compute_feedback,
    contingency,
    decision_agreement,
    feedback_specificity,
    individual_alpha_frequency,
    learner_agreement,
    reward_decisions,
    simulate_session,
    spec_table,
    specification_curve,
    specification_curve_test,
    within_session_slope,
)
from nf_multiverse.artifacts import target_reference_signal
from nf_multiverse.feedback import agreement_matrix, extract_target
from nf_multiverse.learning import block_means

TARGET = ("Pz", "P3", "P4")


@pytest.fixture(scope="module")
def learning_session():
    return simulate_session(fs=250, baseline_s=30, block_s=20, n_blocks=8, learning_gain=1.0, eog_amp_uv=20, emg_amp_uv=5, seed=1)


def test_specification_space_and_table():
    specs = build_specification_space()
    assert len(specs) == 4 * 2 * 4 * 3 * 3 * 3 * 2
    names = [s.name() for s in specs]
    assert len(set(names)) == len(names)
    small = build_specification_space(max_specs=50)
    assert 0 < len(small) <= 50
    tab = spec_table(small)
    assert {"reference", "estimator", "window_s", "name", "spec_id"} <= set(tab.columns)


def test_feedback_recovers_learning(learning_session):
    ses = learning_session
    fb = compute_feedback(ses.eeg, ses.ch_names, ses.fs, FeedbackSpec(), TARGET, baseline=ses.baseline)
    assert len(fb.times) == len(fb.values) and np.isfinite(fb.values).mean() > 0.95
    bm = block_means(fb.times, fb.values, ses.block_edges)
    res = within_session_slope(bm)
    assert res["slope"] > 0 and res["p"] < 0.05
    assert classify_learner(res["slope"], res["p"])
    flat = simulate_session(fs=250, baseline_s=30, block_s=20, n_blocks=8, learning_gain=0.0, seed=2)
    fb0 = compute_feedback(flat.eeg, flat.ch_names, flat.fs, FeedbackSpec(), TARGET, baseline=flat.baseline)
    res0 = within_session_slope(block_means(fb0.times, fb0.values, flat.block_edges))
    assert abs(res0["slope"]) < abs(res["slope"])


def test_individual_alpha_frequency():
    ses = simulate_session(fs=250, baseline_s=40, block_s=10, n_blocks=2, alpha_freq=11.0, seed=3)
    x = extract_target(ses.eeg, ses.ch_names, "car", TARGET)
    iaf = individual_alpha_frequency(x[: int(40 * ses.fs)], ses.fs)
    assert abs(iaf - 11.0) < 0.75
    fb = compute_feedback(ses.eeg, ses.ch_names, ses.fs, FeedbackSpec(band="individual"), TARGET, baseline=ses.baseline)
    assert abs(np.mean(fb.band) - 11.0) < 0.75


def test_estimators_and_references_agree_partially(learning_session):
    ses = learning_session
    base = FeedbackSpec()
    fbs = {}
    for est in ("welch", "hilbert", "fft", "bandpass_rms"):
        fb = compute_feedback(ses.eeg, ses.ch_names, ses.fs, base.with_(estimator=est), TARGET, baseline=ses.baseline)
        fbs[est] = (fb.times, reward_decisions(fb.values))
    assert decision_agreement(fbs["welch"][1], fbs["welch"][1]) == 1.0
    M = agreement_matrix(fbs)
    assert M.loc["welch", "hilbert"] > 0.7 and M.shape == (4, 4)
    lap = compute_feedback(ses.eeg, ses.ch_names, ses.fs, base.with_(reference="laplacian", window_s=2.0, step_s=0.5), TARGET, baseline=ses.baseline)
    assert np.isfinite(lap.values).mean() > 0.9  # different grid, still valid
    thr = compute_feedback(ses.eeg, ses.ch_names, ses.fs, base.with_(artifact="threshold", artifact_threshold_uv=30.0), TARGET, baseline=ses.baseline)
    assert thr.bad_fraction > 0.0


def test_artifact_specificity_and_regression():
    clean = simulate_session(fs=250, baseline_s=30, block_s=20, n_blocks=6, learning_gain=0.5, eog_amp_uv=0.0, seed=4)
    dirty = simulate_session(fs=250, baseline_s=30, block_s=20, n_blocks=6, learning_gain=0.5, eog_amp_uv=150.0, seed=4)
    spec = FeedbackSpec(reference="none")
    r2 = {}
    for name, ses in (("clean", clean), ("dirty", dirty)):
        fb = compute_feedback(ses.eeg, ses.ch_names, ses.fs, spec, ("Fz",), baseline=ses.baseline)
        reg = artifact_regressors(ses.eeg, ses.ch_names, ses.fs, spec.window_s, spec.step_s)
        r2[name] = feedback_specificity(fb.values, reg)["r2"]
    assert r2["dirty"] > r2["clean"] + 0.1
    fb_reg = compute_feedback(dirty.eeg, dirty.ch_names, dirty.fs, spec.with_(artifact="regress_eog"), ("Fz",), baseline=dirty.baseline)
    reg = artifact_regressors(dirty.eeg, dirty.ch_names, dirty.fs, spec.window_s, spec.step_s)
    assert feedback_specificity(fb_reg.values, reg)["r2"] < r2["dirty"]


def test_contingency_with_ground_truth(learning_session):
    ses = learning_session
    fb = compute_feedback(ses.eeg, ses.ch_names, ses.fs, FeedbackSpec(), TARGET, baseline=ses.baseline)
    truth = target_reference_signal(ses.alpha_envelope, ses.fs, fb.times, fb.spec.window_s)
    c = contingency(fb.values, truth, reward_decisions(fb.values))
    assert c["spearman"] > 0.3 and c["mutual_information"] > 0.05
    assert 0 < c["reward_rate"] < 1


def test_learner_agreement_and_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    labels = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], bool)
    agg = learner_agreement(labels)
    assert agg["fraction_unanimous"] == 0.75 and 0 < agg["mean_pairwise_kappa"] < 1


def test_specification_curve_test_detects_learning_and_null():
    rng = np.random.default_rng(5)
    n_specs, n_subj = 20, 16
    signal = rng.normal(0.6, 0.3, (1, n_subj)) + rng.normal(0, 0.2, (n_specs, n_subj))
    res = specification_curve_test(signal, n_perm=300, seed=0)
    assert res["p_median"] < 0.05 and res["share_positive"] > 0.9
    null = rng.normal(0, 0.5, (n_specs, n_subj))
    res0 = specification_curve_test(null, n_perm=300, seed=0)
    assert res0["p_median"] > 0.05
    curve = specification_curve(signal, [f"s{i}" for i in range(n_specs)], n_boot=100)
    assert list(curve.columns) >= ["spec_id", "name", "mean", "ci_low", "ci_high", "p", "frac_positive"]
    assert (curve["mean"].diff().dropna() >= 0).all()
