"""Synthetic-data tests for szforecast (no downloads, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from szforecast import ecg_hrv, forecast_eval, fusion, windows  # noqa: E402


def synthetic_ecg(fs: float = 250.0, seconds: float = 120.0, hr: float = 72.0, rr_jitter: float = 0.02,
                  noise: float = 0.05, seed: int = 0):
    """ECG-like train of narrow biphasic pulses with slightly irregular RR intervals. Returns (signal, beat_times)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    beats, tt = [], 0.5
    while tt < seconds - 0.5:
        beats.append(tt)
        tt += 60.0 / hr + rng.normal(0, rr_jitter)
    beats = np.asarray(beats)
    x = np.zeros_like(t)
    for b in beats:
        x += 1.0 * np.exp(-0.5 * ((t - b) / 0.012) ** 2) - 0.25 * np.exp(-0.5 * ((t - b - 0.03) / 0.02) ** 2)
    x += 0.1 * np.sin(2 * np.pi * 0.3 * t)  # baseline wander
    x += noise * rng.normal(size=t.size)
    return x, beats


def test_r_peak_detection_recovers_beats():
    x, beats = synthetic_ecg()
    peaks = ecg_hrv.detect_r_peaks(x, 250.0)
    assert abs(peaks.size - beats.size) <= 2
    # each true beat has a detection within 40 ms
    d = np.min(np.abs(peaks[None, :] / 250.0 - beats[:, None]), axis=1)
    assert np.median(d) < 0.02 and np.mean(d < 0.04) > 0.95


def test_hrv_features_reasonable():
    x, beats = synthetic_ecg(hr=60.0, rr_jitter=0.03)
    peaks = ecg_hrv.detect_r_peaks(x, 250.0)
    rr = ecg_hrv.rr_from_peaks(peaks, 250.0)
    f = ecg_hrv.hrv_features(rr)
    assert 55 < f["mean_hr"] < 65
    assert f["sqi"] < 0.1
    assert f["rmssd"] > 0 and f["sdnn"] > 0 and np.isfinite(f["lf"]) and np.isfinite(f["hf"])
    df = ecg_hrv.sliding_hrv(x, 250.0, win_s=60.0, step_s=30.0)
    assert len(df) == 3 and "t_start" in df.columns


def test_label_windows_sph_sop():
    starts = np.arange(0, 7200, 60.0)  # 2 h of 1-min windows
    onsets, offsets = [5400.0], [5460.0]  # seizure at 90 min
    lab = windows.label_windows(starts, 60.0, onsets, offsets, sph_s=300, sop_s=1800, postictal_s=600,
                                interictal_gap_s=3000)
    pre = starts[lab == windows.PREICTAL]
    assert pre.min() >= 5400 - 300 - 1800 and pre.max() + 60 <= 5400 - 300
    assert (lab[(starts >= 5100) & (starts < 5460 + 600)] == windows.EXCLUDED).all()
    assert (lab[starts < 5400 - 3000 - 60] == windows.INTERICTAL).all()


def test_surrogates_preserve_counts_and_folds_have_no_leakage():
    sur = windows.seizure_surrogates([100, 900, 2500], [130, 950, 2530], 4000, n=20, seed=1)
    assert len(sur) == 20 and all(s[0].size == 3 for s in sur)
    assert any(not np.allclose(s[0], [100, 900, 2500]) for s in sur)
    subs = np.repeat([f"s{i}" for i in range(10)], 5)
    folds = windows.patient_folds(subs, n_folds=5, seizure_counts=np.ones(50))
    for f in range(5):
        windows.assert_no_subject_leakage(subs[folds != f], subs[folds == f])
    with pytest.raises(ValueError):
        windows.assert_no_subject_leakage(["a", "b"], ["b"])


def test_alarm_scoring_and_chance_level():
    onsets = [3600.0, 7200.0]
    perfect = [3600 - 600, 7200 - 900]
    sc = forecast_eval.evaluate_alarms(perfect, onsets, sph_s=300, sop_s=1800, total_time_s=10800)
    assert sc.sensitivity == 1.0 and sc.n_false_alarms == 0
    bad = [100.0, 3600 - 100]  # too early / inside SPH
    sc2 = forecast_eval.evaluate_alarms(bad, onsets, 300, 1800, 10800)
    assert sc2.sensitivity == 0.0 and sc2.n_false_alarms == 2
    assert forecast_eval.chance_sensitivity(0.0, 1800) == 0.0
    assert 0.39 < forecast_eval.chance_sensitivity(1.0, 1800) < 0.40
    assert forecast_eval.binomial_pvalue(2, 2, 0.1) == pytest.approx(0.01)
    assert forecast_eval.improvement_over_chance(sc) == pytest.approx(1.0 - 2 * 1800 / 10800)


def test_firing_power_refractory():
    prob = np.r_[np.zeros(20), np.ones(40), np.zeros(20)]
    al = forecast_eval.firing_power(prob, window_len_s=60, sop_s=600, alarm_thr=0.5)
    assert al.sum() >= 1
    idx = np.where(al)[0]
    assert np.all(np.diff(idx) > 10)


def test_late_fusion_and_residualize():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.binomial(1, 0.5, n)
    groups = np.repeat(np.arange(20), 20)
    eeg = rng.normal(size=(n, 6)) + 1.0 * y[:, None]
    ecg = rng.normal(size=(n, 4)) + 0.8 * y[:, None]
    ecg[rng.random(ecg.shape) < 0.05] = np.nan  # missing features
    lf = fusion.LateFusion(n_splits=4).fit({"eeg": eeg, "ecg": ecg}, y, groups)
    p = lf.predict_proba({"eeg": eeg, "ecg": ecg})
    assert forecast_eval.auroc(y, p) > 0.85
    assert set(lf.modality_weights()) == {"eeg", "ecg"}
    # residualization removes a covariate-driven effect
    cov = rng.normal(size=n)
    X = np.column_stack([cov * 3 + rng.normal(size=n), rng.normal(size=n)])
    R = fusion.residualize(X, cov)
    assert abs(np.corrcoef(R[:, 0], cov)[0, 1]) < 0.1
    assert np.isfinite(R).all()
