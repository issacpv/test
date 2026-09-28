"""Synthetic tests for art_damping (second-order catheter model; no data, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from art_damping import flush, impact, sqi, transfer  # noqa: E402

FS = 125.0


@pytest.fixture(scope="module")
def true_abp():
    return transfer.synthetic_true_abp(fs=FS, n_beats=50, hr=80, seed=0)


def _measured(true_abp, fn, zeta):
    y = transfer.apply_catheter_system(true_abp.abp, FS, fn, zeta)
    return y, sqi.beat_damping_features(y, FS, sqi.detect_onsets(y, FS))


def test_underdamping_overestimates_and_overdamping_underestimates_sbp(true_abp):
    f0 = sqi.beat_damping_features(true_abp.abp, FS, sqi.detect_onsets(true_abp.abp, FS))
    _, fu = _measured(true_abp, 8.0, 0.15)
    _, fo = _measured(true_abp, 6.0, 1.5)
    _, fa = _measured(true_abp, 30.0, 0.5)
    assert fu["sbp"].median() > f0["sbp"].median() + 3 and fu["dbp"].median() <= f0["dbp"].median() + 1
    assert fo["sbp"].median() < f0["sbp"].median() - 3 and fo["dbp"].median() >= f0["dbp"].median() - 1
    assert abs(fu["map"].median() - f0["map"].median()) < 2.5 and abs(fo["map"].median() - f0["map"].median()) < 2.5
    assert abs(fa["sbp"].median() - f0["sbp"].median()) < 2


def test_gardner_adequacy_rules():
    assert transfer.gardner_adequacy(30.0, 0.5) == "adequate"
    assert transfer.gardner_adequacy(10.0, 0.2) == "underdamped"
    assert transfer.gardner_adequacy(15.0, 1.2) == "overdamped"
    assert transfer.gardner_adequacy(3.0, 0.5) == "inadequate_low_fn"
    assert transfer.gardner_adequacy(np.nan, 0.5) == "unknown"
    with pytest.raises(ValueError):
        transfer.catheter_system(100.0, 0.5, FS)


@pytest.mark.parametrize("fn,zeta", [(12.0, 0.3), (25.0, 0.2), (15.0, 0.5), (8.0, 0.15)])
def test_flush_fit_recovers_fn_and_zeta(fn, zeta):
    x = transfer.synthetic_flush(FS, fn, zeta, baseline=80.0)
    events = flush.detect_flush_events(x, FS)
    assert len(events) == 1
    r = flush.fit_flush_response(x, FS, *events[0])
    assert r.r2 > 0.95 and r.n_extrema >= 2
    assert r.fn_hz == pytest.approx(fn, rel=0.15)
    assert r.zeta == pytest.approx(zeta, abs=0.1)
    assert r.baseline == pytest.approx(80.0, abs=3.0) and r.plateau == pytest.approx(300.0, rel=0.05)
    assert r.adequacy == transfer.gardner_adequacy(fn, zeta)


def test_overdamped_flush_and_pulsatile_background(true_abp):
    x = transfer.synthetic_flush(FS, 8.0, 1.5, baseline=80.0)
    ev = flush.detect_flush_events(x, FS)
    r = flush.fit_flush_response(x, FS, *ev[0])
    assert r.adequacy == "overdamped" and r.zeta > transfer.ZETA_MAX_ADEQUATE and r.n_extrema == 0
    # flush embedded in a pulsatile record is still detected and labelled
    n = int(3.6 * FS)
    base = true_abp.abp[:n]
    y = transfer.synthetic_flush(FS, 20.0, 0.25, baseline=base)
    rec = np.r_[true_abp.abp[:n], y, true_abp.abp[n:2 * n]]
    labels = flush.flush_labels(rec, FS)
    assert len(labels) == 1
    assert labels["adequacy"].iloc[0] in ("adequate", "underdamped")
    assert labels["fn_hz"].iloc[0] == pytest.approx(20.0, rel=0.35)
    grid = np.arange(0, 60, 1.0)
    prop = flush.propagate_labels(labels, grid, max_age_s=30)
    t_rel = labels["t_release_s"].iloc[0]
    assert (prop[grid < t_rel] == "unlabelled").all() and (prop[(grid >= t_rel) & (grid <= t_rel + 30)] != "unlabelled").all()
    assert flush.detect_flush_events(true_abp.abp, FS) == []


def test_flush_free_features_separate_damping_classes(true_abp):
    ya, fa = _measured(true_abp, 30.0, 0.5)
    yu, fu = _measured(true_abp, 8.0, 0.15)
    yo, fo = _measured(true_abp, 6.0, 1.5)
    assert set(sqi.BEAT_FEATURES) <= set(fa.columns) and fa["plausible"].mean() > 0.9
    assert fu["overshoot"].median() > fa["overshoot"].median() > fo["overshoot"].median()
    # over-damping blunts the upstroke; under-damping inflates PP (the denominator) and filters the
    # fastest part of the upstroke, so PP-normalised dP/dt is not a monotone under-damping marker
    assert fa["dpdt_max_norm"].median() > fo["dpdt_max_norm"].median()
    assert fo["rise_time"].median() > fa["rise_time"].median()
    sa, su = sqi.spectral_features(ya, FS), sqi.spectral_features(yu, FS)
    assert su["spectral_peakiness"] > sa["spectral_peakiness"]
    # classifier: windows from several synthetic records, grouped CV
    Xs, ys, gs = [], [], []
    for seed in range(6):
        rec = transfer.synthetic_true_abp(fs=FS, n_beats=40, hr=70 + 5 * seed, seed=seed)
        for cls, (fn, z) in enumerate([(30.0, 0.5), (8.0, 0.15)]):
            y = transfer.apply_catheter_system(rec.abp, FS, fn, z)
            beats = sqi.beat_damping_features(y, FS, sqi.detect_onsets(y, FS))
            w = sqi.window_features(y, FS, beats, window_s=5.0)
            Xs.append(w)
            ys.append(np.full(len(w), cls))
            gs.append(np.full(len(w), seed))
    X, y, g = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gs)
    assert set(sqi.FEATURE_NAMES) <= set(X.columns)
    assert sqi.cv_auroc(X, y, g, n_splits=3) > 0.9
    clf = sqi.fit_damping_classifier(X, y)
    assert clf.predict(X[list(sqi.FEATURE_NAMES)]).shape == (len(X),)


def test_impact_tables():
    rng = np.random.default_rng(0)
    n = 240
    minutes = pd.DataFrame({"t_center": np.arange(n) * 60.0 + 30, "sbp": 110 + 15 * rng.normal(size=n),
                            "dbp": 60 + 8 * rng.normal(size=n), "map": 75 + 10 * rng.normal(size=n),
                            "damping_class": np.where(np.arange(n) % 4 == 0, "underdamped", "adequate")})
    minutes.loc[minutes.damping_class == "underdamped", "sbp"] += 20  # underdamping inflates SBP
    nibp = pd.DataFrame({"t": np.arange(0, n * 60, 600.0) + 300, "sbp": 110.0, "dbp": 60.0, "map": 75.0})
    paired = impact.pair_nibp_abp(minutes, nibp, tolerance_s=90)
    assert len(paired) == len(nibp) and set(paired.columns) >= {"d_sbp", "damping_class"}
    disc = impact.discrepancy_by_class(paired)
    assert set(disc["damping_class"]) <= {"adequate", "underdamped"} and "d_sbp_mean" in disc
    tcs = impact.threshold_crossing_share(minutes)
    assert set(tcs["variable"]) == {"map", "sbp"}
    for var in ("map", "sbp"):
        sub = tcs[tcs.variable == var]
        assert abs(sub["flag_share"].sum() - 1) < 1e-9 or sub["n_flagged"].sum() == 0
    ev = impact.event_rate_ratio(np.array([90.0, 150.0, 210.0, 270.0, 330.0]), minutes)
    assert ev.loc[ev.damping_class == "adequate", "rate_ratio"].item() == 1.0
    assert ev["exposure_h"].sum() == pytest.approx(n / 60.0)
    point, lo, hi = impact.cluster_bootstrap(paired.assign(rec=np.arange(len(paired)) % 4), "rec",
                                             lambda d: d["d_sbp"].mean(), n_boot=30)
    assert lo <= point <= hi
