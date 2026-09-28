"""Synthetic-data tests for iedbench (no downloads, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iedbench import detectors, downstream, events, morphology  # noqa: E402
from iedbench.events import Event  # noqa: E402


def pink_noise(n: int, rng, fs: float = 512.0, hp_hz: float = 0.5) -> np.ndarray:
    """1/f noise high-passed at ``hp_hz`` (EEG is always high-passed; removes unrealistic drift)."""
    w = rng.normal(size=n)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec = np.fft.rfft(w) / np.sqrt(np.maximum(f, fs / n))
    spec[f < hp_hz] = 0.0
    x = np.fft.irfft(spec, n)
    return (x - x.mean()) / (x.std() + 1e-9)


def synthetic_record(fs: float = 512.0, seconds: float = 60.0, n_ch: int = 4, n_spikes: int = 20, seed: int = 0,
                     amp: float = 6.0):
    """Pink-noise background with planted spike + slow-wave events on channel 0 (with a field on channel 1)."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    X = np.stack([pink_noise(n, rng) for _ in range(n_ch)])
    tpl = detectors.synthetic_spike_template(fs, spike_ms=40, slow_ms=200, slow_amp=-0.4)
    tpl = tpl / tpl.max()
    times = np.sort(rng.uniform(2, seconds - 2, n_spikes))
    ref = []
    off = int(np.argmax(tpl))
    for t in times:
        s = int(t * fs) - off
        X[0, s:s + tpl.size] += amp * tpl
        X[1, s:s + tpl.size] += 0.6 * amp * tpl
        ref.append(Event("c0", t, 0.04))
        ref.append(Event("c1", t, 0.04))
    return X, ref, [f"c{i}" for i in range(n_ch)]


def test_match_events_and_sweep():
    ref = [Event("a", 1.0), Event("a", 2.0), Event("b", 3.0)]
    pred = [Event("a", 1.05, score=0.9), Event("a", 2.5, score=0.8), Event("b", 3.02, score=0.7), Event("b", 9.0, score=0.2)]
    m = events.match_events(pred, ref, tol_s=0.1)
    assert m["tp"] == 2 and m["fp"] == 2 and m["fn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == pytest.approx(2 / 3)
    m2 = events.match_events(pred, ref, tol_s=0.1, per_channel=False)
    assert m2["tp"] == 2
    sw = events.threshold_sweep(pred, ref, duration_s=600, n_channels=2, thresholds=[0.1, 0.75])
    assert sw.loc[0, "n_pred"] == 4 and sw.loc[1, "n_pred"] == 2
    rec, thr = events.sensitivity_at_fp_rate(sw, fp_per_min_max=0.05)
    assert thr == 0.75
    mask = np.zeros((1, 100), bool)
    mask[0, 10:15] = True
    ev = events.events_from_mask(mask, 100.0, ["x"])
    assert len(ev) == 1 and ev[0].duration_s == pytest.approx(0.05)


def test_envelope_and_template_detectors_find_planted_spikes():
    X, ref, names = synthetic_record()
    ref0 = [r for r in ref if r.channel == "c0"]
    ev = detectors.envelope_detector(X, 512.0, names, k=4.0)
    m = events.match_events([e for e in ev if e.channel == "c0"], ref0, tol_s=0.1)
    assert m["recall"] > 0.7
    assert events.fp_per_minute(m["fp"], 60.0) < 10.0
    ev_t = detectors.template_detector(X, 512.0, names, thr=0.7)
    m_t = events.match_events([e for e in ev_t if e.channel == "c0"], ref0, tol_s=0.1)
    assert m_t["recall"] > 0.8 and m_t["precision"] > 0.5
    # normal background produces few detections at the default threshold
    Xn = np.stack([pink_noise(int(512 * 60), np.random.default_rng(5)) for _ in range(2)])
    fp = events.fp_per_minute(len(detectors.template_detector(Xn, 512.0)), 60.0, 2)
    assert fp < 3.0
    # threshold sweep machinery: lowering the threshold can only add detections
    sw = events.threshold_sweep(detectors.template_detector(X, 512.0, names, thr=0.3), ref, 60.0, 4, [0.3, 0.7])
    assert sw.loc[0, "n_pred"] >= sw.loc[1, "n_pred"]


def test_morphology_features_and_criteria():
    fs = 512.0
    X, ref, names = synthetic_record(n_spikes=5, seed=3, amp=10.0)
    idx = int(ref[0].t_s * fs)
    f = morphology.ifcn_features(X[0], fs, idx)
    assert 15 < f["duration_ms"] < 90
    assert f["rel_amplitude"] > 2 and f["slow_wave"] > 0.05
    # a spike without after-going slow wave scores lower on the slow-wave feature
    x_plain = np.zeros(int(3 * fs))
    x_plain[int(1.5 * fs) - 30:int(1.5 * fs) + 30] += 10 * np.exp(-0.5 * (np.arange(-30, 30) / 8.7) ** 2)
    assert morphology.ifcn_features(x_plain, fs, int(1.5 * fs))["slow_wave"] < f["slow_wave"]
    n_field, mx = morphology.spatial_field(X, fs, idx, 0)
    assert n_field >= 1 and mx > 0.5
    F = morphology.features_matrix(X, fs, [(0, idx), (2, idx)])
    assert F.shape == (2, len(morphology.FEATURE_NAMES))
    assert morphology.criteria_count(dict(zip(morphology.FEATURE_NAMES, F[0]))) >= 3


def test_morphology_classifier_detector():
    recs = [synthetic_record(seed=s, n_spikes=25) for s in range(3)]
    det = detectors.MorphologyClassifierDetector(min_prom_mad=2.5).fit([(X, 512.0, ref, names) for X, ref, names in recs])
    X, ref, names = synthetic_record(seed=9, n_spikes=20)
    ev = det.detect(X, 512.0, names, thr=0.5)
    m = events.match_events([e for e in ev if e.channel == "c0"], [r for r in ref if r.channel == "c0"], tol_s=0.1)
    assert m["recall"] > 0.6
    assert "sharpness" in det.coefficients()


def test_downstream_soz_and_vigilance():
    chans = [f"ch{i}" for i in range(8)]
    ev_a = [Event("ch0", t) for t in np.linspace(1, 590, 30)] + [Event("ch1", t) for t in np.linspace(1, 590, 10)]
    ev_b = [Event("ch2", t) for t in np.linspace(1, 590, 30)] + [Event("ch0", t) for t in np.linspace(1, 590, 5)]
    ra = downstream.spike_rate_per_channel(ev_a, 600, chans)
    rb = downstream.spike_rate_per_channel(ev_b, 600, chans)
    assert ra["ch0"] == pytest.approx(3.0)
    assert downstream.soz_ranking_auc(ra, ["ch0", "ch1"]) == 1.0
    assert downstream.top_k_overlap(ra, rb, k=2) < 1.0
    mv = downstream.soz_multiverse({"a": ra, "b": rb}, ["ch0", "ch1"])
    assert mv.attrs["auc_range"] > 0
    p = downstream.channel_permutation_null(ra, ["ch0", "ch1"], n_perm=200)
    assert p < 0.1
    st = downstream.vigilance_stratified_rates(ev_a, [(0, 300, "wake"), (300, 600, "N2")])
    assert set(st["stage"]) == {"wake", "N2"} and st["n_events"].sum() == 40
    lo, hi = downstream.poisson_rate_ci(10, 10.0)
    assert lo < 1.0 < hi
