"""Synthetic-data tests for gaitshift."""
from __future__ import annotations

import numpy as np
import pytest

from gaitshift import benchmark, features, harmonize


def test_harmonize_pipeline_units_resampling_and_invariance():
    x = features.synthesize_gait(fs=100.0, duration_s=10.0, step_freq_hz=1.8, seed=0)
    y, fs = harmonize.harmonize(x / harmonize.G, 100.0, "g", fs_out=50.0)
    assert fs == 50.0 and y.shape == (500, 3)
    assert harmonize.to_m_s2(np.array([1000.0]), "mg")[0] == pytest.approx(harmonize.G)
    with pytest.raises(ValueError):
        harmonize.to_m_s2(np.array([1.0]), "furlongs")
    # gravity removed: vertical dynamic channel has ~zero mean
    assert abs(y[:, 1].mean()) < 0.3
    # rotating the sensor should leave the invariant channels almost unchanged
    theta = np.deg2rad(35)
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    y_rot, _ = harmonize.harmonize((x @ R.T) / harmonize.G, 100.0, "g", fs_out=50.0)
    assert np.corrcoef(y[:, 0], y_rot[:, 0])[0, 1] > 0.99
    w = harmonize.window(y, 50.0, win_s=5.0, step_s=2.5)
    assert w.shape == (3, 250, 3)
    assert harmonize.REGISTRY["daphnet"].placement == "ankle"


def test_gait_features_track_cadence_and_freezing():
    fs = 100.0
    x = features.synthesize_gait(fs=fs, duration_s=20.0, step_freq_hz=1.8, seed=1)
    ch = harmonize.orientation_invariant(x, fs)
    f = features.feature_vector(ch, fs)
    assert f["step_freq_hz"] == pytest.approx(1.8, abs=0.15)
    assert f["stride_regularity"] > 0.7 and f["freeze_index"] < 0.5
    x_fog = features.synthesize_gait(fs=fs, duration_s=20.0, step_freq_hz=1.8, fog_segments=[(5, 15)], seed=1)
    f_fog = features.feature_vector(harmonize.orientation_invariant(x_fog, fs), fs)
    assert f_fog["freeze_index"] > f["freeze_index"]
    # asymmetric gait lowers the harmonic ratio relative to symmetric gait
    x_asym = features.synthesize_gait(fs=fs, duration_s=20.0, step_freq_hz=1.8, asymmetry=0.8, seed=1)
    f_asym = features.feature_vector(harmonize.orientation_invariant(x_asym, fs), fs)
    assert f_asym["harmonic_ratio"] < f["harmonic_ratio"]
    # VGRF stride times from a synthetic force profile with 1.1 s strides
    t = np.arange(0, 20, 1 / fs)
    force = np.clip(600 * np.sin(2 * np.pi * t / 1.1), 0, None)
    strides = features.vgrf_stride_times(force, fs)
    assert np.median(strides) == pytest.approx(1.1, abs=0.02)
    assert features.stride_time_cv(strides) < 5.0


def _synthetic_benchmark(seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, subj, ds = [], [], [], []
    for d, offset in (("A", 0.0), ("B", 1.5), ("C", -1.0)):
        for s in range(30):
            label = s % 2
            for _w in range(6):
                feat = rng.normal(size=4) + label * np.array([1.2, 0.0, 0.8, 0.0]) + offset * np.array([0, 1, 0, 1])
                X.append(feat)
                y.append(label)
                subj.append(f"{d}{s}")
                ds.append(d)
    return np.array(X), np.array(y), np.array(subj), np.array(ds)


def test_lodo_benchmark_and_shift_diagnostics():
    X, y, subj, ds = _synthetic_benchmark()
    lodo = benchmark.leave_one_dataset_out(X, y, subj, ds, n_boot=50)
    assert set(lodo["held_out"]) == {"A", "B", "C"}
    assert (lodo["auroc"] > 0.7).all()
    assert ((lodo["auroc_lo"] <= lodo["auroc"]) & (lodo["auroc"] <= lodo["auroc_hi"])).all()
    idref = benchmark.in_distribution_reference(X, y, subj, ds)
    gap = benchmark.deployment_gap(idref, lodo, metric="auroc")
    assert len(gap) == 3 and gap["gap"].abs().max() < 0.3
    shift = benchmark.domain_shift_c2st(X[ds == "A"], X[ds == "B"], subj[ds == "A"], subj[ds == "B"])
    assert shift > 0.8  # datasets differ by a mean offset in two features
    null = benchmark.dataset_identity_null(X, y, subj, ds)
    assert (null["auroc"] < 0.7).all()
