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


def _sbp_dbp(abp, onsets):
    f = sqi.beat_damping_features(abp, FS, onsets)
    return f["sbp"].median(), f["dbp"].median(), f["map"].median(), f


def test_underdamping_overestimates_and_overdamping_underestimates_sbp(true_abp):
    sbp0, dbp0, map0, _ = _sbp_dbp(true_abp.abp, true_abp.onsets)
    under = transfer.apply_catheter_system(true_abp.abp, FS, fn_hz=8.0, zeta=0.15)
    over = transfer.apply_catheter_system(true_abp.abp, FS, fn_hz=6.0, zeta=1.5)
    sbp_u, dbp_u, map_u, _ = _sbp_dbp(under, true_abp.onsets)
    sbp_o, dbp_o, map_o, _ = _sbp_dbp(over, true_abp.onsets)
    assert sbp_u > sbp0 + 3 and dbp_u <= dbp0 + 1
    assert sbp_o < sbp0 - 3 and dbp_o >= dbp0 - 1
    assert abs(map_u - map0) < 2.5 and abs(map_o - map0) < 2.5  # MAP is preserved
    adequate = transfer.apply_catheter_system(true_abp.abp, FS, fn_hz=30.0, zeta=0.5)
    sbp_a, _, _, _ = _sbp_dbp(adequate, true_abp.onsets)
    assert abs(sbp_a - sbp0) < 2


def test_gardner_adequacy_rules():
    assert transfer.gardner_adequacy(30.0, 0.5) == "adequate"
    assert transfer.gardner_adequacy(10.0, 0.2) == "underdamped"
    assert transfer.gardner_adequacy(15.0, 1.2) == "overdamped"
    assert transfer.gardner_adequacy(3.0, 0.5) == "inadequate_low_fn"
    assert transfer.gardner_adequacy(np.nan, 0.5) == "unknown"
    with pytest.raises(ValueError):
        transfer.catheter_system(100.0, 0.5, FS)


@pytest.mark.parametrize("fn,zeta", [(12.0, 0.3), (25.0, 0.2), (15.0, 0.5)])
def test_flush_ringing_recovers_fn_and_zeta(fn, zeta):
    x = transfer.synthetic_flush(FS, fn, zeta, baseline=80.0)
    events = flush.detect_flush_events(x, FS)
    assert len(events) == 1
    r = flush.ringing_parameters(x, FS, events[0][1])
    assert r.n_extrema >= 2
    assert r.fn_hz == pytest.approx(fn, rel=0.2)
    assert r.zeta == pytest.approx(zeta, abs=0.15)
    assert r.adequacy == transfer.gardner_adequacy(fn, zeta)


def test_overdamped_flush_has_no_ringing_and_labels_propagate(true_abp):
    x = transfer.synthetic_flush(FS, 8.0, 1.5, baseline=80.0)
    ev = flush.detect_flush_events(x, FS)
    r = flush.ringing_parameters(x, FS, ev[0][1])
    assert r.adequacy == "overdamped" and np.isnan(r.zeta)
    # flush embedded in a pulsatile record is still detected
    n = int(3.6 * FS)
    base = true_abp.abp[:n]
    y = transfer.synthetic_flush(FS, 20.0, 0.25, baseline=base)
    rec = np.r_[true_abp.abp[:n], y, true_abp.abp[n:2 * n]]
    labels = flush.flush_labels(rec, FS)
    assert len(labels) == 1 and labels["adequacy"].iloc[0] in ("adequate", "underdamped")
    grid = np.arange(0, 60, 1.0)
    prop = flush.propagate_labels(labels, grid, max_age_s=30)
    t_rel = labels["t_release_s"].iloc[0]
    assert (prop[grid < t_rel] == "unlabelled").all() and (prop[(grid >= t_rel) & (grid <= t_rel + 30)] != "unlabelled").all()


def test_flush_free_features_separate_damping_classes(true_abp):
    onsets = true_abp.onsets
    f_ok = sqi.beat_damping_features(transfer.apply_catheter_system(true_abp.abp, FS, 30.0, 0.5), FS, onsets)
    f_under = sqi.beat_damping_features(transfer.apply_catheter_system(true_abp.abp, FS, 8.0, 0.15), FS, onsets)
    f_over = sqi.beat_damping_features(transfer.apply_catheter_system(true_abp.abp, FS, 6.0, 1.5), FS, onsets)
    assert set(sqi.FEATURE_NAMES) <= set(f_ok.columns) and f_ok["plausible"].mean() > 0.9
    assert f_under["hf_ratio"].median() > f_ok["hf_ratio"].median()
    assert f_under["dpdt_max_norm"].median() > f_ok["dpdt_max_norm"].median() > f_over["dpdt_max_norm"].median()
    assert f_over["rise_time"].median() > f_ok["rise_time"].median()
    # classifier: windows from several synthetic records, grouped CV
    Xs, ys, gs = [], [], []
    for seed in range(6):
        rec = transfer.synthetic_true_abp(fs=FS, n_beats=40, hr=70 + 5 * seed, seed=seed)
        for cls, (fn, z) in enumerate([(30.0, 0.5), (8.0, 0.15)]):
            beats = sqi.beat_damping_features(transfer.apply_catheter_system(rec.abp, FS, fn, z), FS, rec.onsets)
            w = sqi.window_features(beats, FS, window_s=5.0)
            Xs.append(w)
            ys.append(np.full(len(w), cls))
            gs.append(np.full(len(w), seed))
    X, y, g = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gs)
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
