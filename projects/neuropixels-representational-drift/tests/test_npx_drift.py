"""Synthetic-data tests for npx_drift (no NWB files or network needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npx_drift import drift_metrics, loaders, nulls, quality, responses  # noqa: E402


@pytest.fixture(scope="module")
def stable_session():
    return loaders.synthetic_session(n_units=45, n_conditions=6, n_repeats=36, drift_rate=0.0, seed=1)


@pytest.fixture(scope="module")
def drifting_session():
    return loaders.synthetic_session(n_units=45, n_conditions=6, n_repeats=36, drift_rate=0.12, seed=1)


def _xcb(sess, n_blocks=3):
    R, stim = responses.response_matrix(sess, table="synthetic")
    X = responses.trial_responses(R)
    cond = stim["condition"].to_numpy()
    block = responses.time_blocks(stim["start_time"].to_numpy(), n_blocks)
    return X, cond, block


def test_bin_spikes_and_response_matrix(stable_session):
    st = np.array([0.1, 0.2, 1.05, 1.5, 2.9])
    counts = responses.bin_spikes(st, np.array([0.0, 1.0, 2.0]), (0.0, 1.0), 0.5)
    assert counts.tolist() == [[2, 0], [1, 1], [0, 1]]
    R, stim = responses.response_matrix(stable_session, table="synthetic", bin_size=0.1)
    assert R.shape[0] == len(stim) and R.shape[1] == len(stable_session.unit_ids) and R.shape[2] == 5
    X = responses.trial_responses(R)
    assert X.shape == (len(stim), len(stable_session.unit_ids))
    assert (X >= 0).all()


def test_time_blocks_and_balance():
    t = np.arange(90.0)
    b = responses.time_blocks(t, 3)
    assert np.bincount(b).tolist() == [30, 30, 30] and b[0] == 0 and b[-1] == 2
    cond = np.tile([0, 1, 2], 30)
    keep = responses.balance_trials(cond, b)
    assert keep.sum() == 90  # already balanced


def test_quality_filters_and_area_groups(stable_session):
    u = stable_session.units
    kept = quality.filter_units(u)
    assert len(kept) <= len(u)
    assert ((kept["isi_violations"] < 0.5) & (kept["amplitude_cutoff"] < 0.1) & (kept["presence_ratio"] > 0.95)).all()
    ibl = u.copy(); ibl["label"] = (np.arange(len(u)) % 2)
    assert len(quality.filter_units(ibl)) == (np.arange(len(u)) % 2).sum()
    strata = quality.quality_strata(u)
    assert set(strata.unique()) <= {0, 1, 2}
    assert quality.area_group("VISp") == "cortex" and quality.area_group("LGd") == "thalamus"
    assert quality.area_group("CA1") == "hippocampal" and quality.area_group("DG-mo") == "hippocampal"
    assert quality.area_group("SCig") == "midbrain" and quality.area_group("grey") == "other"
    cov = quality.unit_stability_covariates(stable_session.spike_times, *stable_session.t_range, n_blocks=4)
    assert {"rate_cv", "presence_ratio_bins", "rate_slope"} <= set(cov.columns)
    assert cov["presence_ratio_bins"].between(0, 1).all()


def test_drift_metrics_detect_drift(stable_session, drifting_session):
    Xs, cs, bs = _xcb(stable_session)
    Xd, cd, bd = _xcb(drifting_session)
    pv_s = drift_metrics.drift_rate_from_matrix(drift_metrics.population_vector_correlation(Xs, cs, bs))
    pv_d = drift_metrics.drift_rate_from_matrix(drift_metrics.population_vector_correlation(Xd, cd, bd))
    assert pv_s["lag1"] > pv_d["lag1"]
    tc_s = drift_metrics.tuning_correlation_per_unit(Xs, cs, bs)
    tc_d = drift_metrics.tuning_correlation_per_unit(Xd, cd, bd)
    assert np.nanmedian(tc_s) > np.nanmedian(tc_d)
    acc_s = drift_metrics.decoder_cross_time(Xs, cs, bs)
    acc_d = drift_metrics.decoder_cross_time(Xd, cd, bd)
    assert acc_s.shape == (3, 3) and np.isfinite(acc_s).all()
    assert np.nanmean(np.diag(acc_s)) > 1 / 6 + 0.2  # decodable well above chance
    assert drift_metrics.decoder_drift_index(acc_d) > drift_metrics.decoder_drift_index(acc_s)
    rd = drift_metrics.rdm_stability(Xs, cs, bs)
    assert np.isfinite(rd).all() and np.allclose(np.diag(rd), 1.0)
    summ = drift_metrics.drift_summary(Xd, cd, bd)
    assert summ.n_units == Xd.shape[1] and 0 <= summ.decoder_within <= 1


def test_null_models(stable_session, drifting_session):
    Xs, cs, bs = _xcb(stable_session)
    Xd, cd, bd = _xcb(drifting_session)
    ns = nulls.time_shuffle_null(Xs, cs, bs, nulls.lag1_pv_metric, n_perm=60, alternative="less")
    nd = nulls.time_shuffle_null(Xd, cd, bd, nulls.lag1_pv_metric, n_perm=60, alternative="less")
    assert nd["p_value"] < 0.05           # drifting population is below its time-shuffle null
    assert ns["p_value"] > 0.05           # stationary population is not
    ps = nulls.poisson_rate_matched_null(Xs, cs, bs, nulls.lag1_pv_metric, n_perm=40)
    assert abs(ps["observed"] - ps["null_mean"]) < 4 * ps["null_sd"] + 0.05
    assert nulls.permutation_pvalue(1.0, np.zeros(9)) == pytest.approx(0.1)
    shifted = nulls.circular_shift_spike_trains(stable_session.spike_times, *stable_session.t_range)
    assert all(len(shifted[u]) == len(stable_session.spike_times[u]) for u in shifted)


def test_cache_roundtrip(tmp_path, stable_session):
    p = tmp_path / "sess.npz"
    loaders.cache_session(stable_session, p)
    back = loaders.load_cached_session(p)
    assert set(back.spike_times) == set(stable_session.spike_times)
    np.testing.assert_allclose(back.spike_times[3], stable_session.spike_times[3])
    assert len(back.stimulus) == len(stable_session.stimulus)


def test_assign_condition():
    stim = pd.DataFrame({"start_time": [0, 1, 2], "stop_time": [1, 2, 3], "frame": [3, None, 3]})
    out = loaders.assign_condition(stim, ("frame",))
    assert out["condition"].tolist() == [0, -1, 0]
