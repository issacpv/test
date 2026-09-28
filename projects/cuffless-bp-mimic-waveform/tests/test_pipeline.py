"""Synthetic-signal tests for beat detection, SQI, evaluation and the PTT model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuffless_bp import beats, evaluation, models, sqi, wfdb_loader  # noqa: E402


def synth_window(fs: float = 125.0, dur_s: float = 10.0, hr: float = 72.0, pat_s: float = 0.25,
                 sbp: float = 120.0, dbp: float = 80.0, noise: float = 0.02, seed: int = 0):
    """ECG-like spikes, PPG pulses delayed by ``pat_s`` and an ABP waveform with known SBP/DBP."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    ibi = 60.0 / hr
    r_times = np.arange(0.3, dur_s, ibi) + rng.normal(0, 0.005, size=len(np.arange(0.3, dur_s, ibi)))
    ecg = np.zeros(n)
    for rt in r_times:
        ecg += np.exp(-0.5 * ((t - rt) / 0.012) ** 2)          # narrow R wave
        ecg += 0.15 * np.exp(-0.5 * ((t - rt - 0.25) / 0.05) ** 2)  # T wave
    ppg = np.zeros(n)
    abp = np.zeros(n)
    for rt in r_times:
        onset = rt + pat_s
        tau = t - onset
        pulse = np.where(tau >= 0, (tau / 0.12) * np.exp(1 - tau / 0.12), 0.0)  # gamma-like upstroke
        ppg += pulse
        abp += pulse
    ppg = ppg / (ppg.max() + 1e-12) + 2.0
    abp = dbp + (sbp - dbp) * abp / (abp.max() + 1e-12)
    ecg += rng.normal(0, noise, n)
    ppg += rng.normal(0, noise * 0.2, n)
    abp += rng.normal(0, noise * 10, n)
    return ecg, ppg, abp, r_times


def test_beat_detection_recovers_hr_and_pat():
    fs = 125.0
    ecg, ppg, abp, r_times = synth_window(fs, hr=72, pat_s=0.25)
    r = beats.detect_r_peaks(ecg, fs)
    assert abs(len(r) - len(r_times)) <= 1
    hr_est = 60.0 / np.median(np.diff(r) / fs)
    assert abs(hr_est - 72) < 2
    feats = beats.pat_features(ecg, ppg, abp, fs)
    assert abs(feats["pat_foot"] - 0.25) < 0.04
    assert feats["pat_peak"] > feats["pat_foot"]
    assert abs(feats["ptt_abp_ppg"]) < 0.03  # same onset in ABP and PPG for the synthetic window
    assert abs(feats["sbp"] - 120) < 3 and abs(feats["dbp"] - 80) < 3
    assert 80 < feats["map"] < 120


def test_pat_shift_is_detected():
    fs = 125.0
    e1, p1, a1, _ = synth_window(fs, pat_s=0.20, seed=1)
    e2, p2, a2, _ = synth_window(fs, pat_s=0.30, seed=2)
    f1 = beats.pat_features(e1, p1, a1, fs)
    f2 = beats.pat_features(e2, p2, a2, fs)
    assert f2["pat_foot"] - f1["pat_foot"] > 0.07


def test_sqi_accepts_clean_and_rejects_bad_windows():
    fs = 125.0
    ecg, ppg, abp, _ = synth_window(fs)
    q = sqi.window_quality(ecg, ppg, abp, fs)
    assert q["ppg"]["accept"] and q["abp"]["accept"] and q["hr_agree"]
    flat = np.full_like(ppg, 2.0)
    assert not sqi.ppg_sqi(flat, fs)["accept"]
    bad_abp = abp.copy()
    bad_abp[300:400] = 400.0  # implausible pressure
    assert not sqi.abp_sqi(bad_abp, fs)["accept"]
    noisy_ppg = ppg + np.random.default_rng(0).normal(0, 1.0, len(ppg))
    assert sqi.ppg_sqi(noisy_ppg, fs)["template_corr"] < sqi.ppg_sqi(ppg, fs)["template_corr"]


def test_splits_and_leakage_audit():
    subjects = np.repeat(np.arange(20), 10)
    folds = evaluation.subject_independent_split(subjects, n_splits=5, seed=1)
    assert len(folds) == 5
    for tr, te in folds:
        rep = evaluation.leakage_audit(tr, te, subjects)
        assert rep["n_shared_subjects"] == 0
    tr, te = evaluation.segment_level_split(len(subjects), 0.2)
    with pytest.raises(ValueError):
        evaluation.leakage_audit(tr, te, subjects)
    rep = evaluation.leakage_audit(tr, te, subjects, strict=False)
    assert rep["n_shared_subjects"] > 0
    times = np.tile(np.arange(10) * 10.0, 20)
    tr, te = evaluation.time_blocked_split(subjects, times, 0.7)
    assert len(tr) == 140 and len(te) == 60
    assert times[te].min() >= 70


def test_standards_metrics():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(100), 5)
    y = rng.normal(120, 15, len(subjects))
    good = y + rng.normal(1.0, 4.0, len(y))
    bad = y + rng.normal(6.0, 12.0, len(y))
    g = evaluation.aami_iso_evaluation(y, good, subjects)
    b = evaluation.aami_iso_evaluation(y, bad, subjects)
    assert g["criterion1_pass"] and g["enough_subjects"] and not b["criterion1_pass"]
    assert evaluation.ieee_1708_grade(y, good)["grade"] == "A"
    assert evaluation.ieee_1708_grade(y, bad)["grade"] == "D"
    assert evaluation.bhs_grade(y, good)["grade"] in ("A", "B")
    ba = evaluation.bland_altman_repeated(y, good, subjects)
    assert abs(ba["bias"] - 1.0) < 1.0 and 3 < ba["sd_total"] < 6
    base = evaluation.trivial_baselines(y[:250], subjects[:250], y[250:], subjects[250:])
    assert np.allclose(base["population_mean"], y[:250].mean())


def test_change_tracking_and_bolus_table():
    rng = np.random.default_rng(1)
    subjects = np.repeat(np.arange(30), 20)
    offsets = np.repeat(rng.normal(0, 20, 30), 20)
    y = 120 + offsets + rng.normal(0, 12, len(subjects))
    perfect_shape = y - offsets + np.repeat(rng.normal(0, 30, 30), 20)  # tracks within-subject change, wrong level
    ct = evaluation.change_tracking(y, perfect_shape, subjects)
    assert ct["r_within_pooled"] > 0.95 and ct["direction_agreement"] > 0.95
    level_only = 120 + offsets  # right level, no tracking
    ct2 = evaluation.change_tracking(y, level_only, subjects)
    assert np.isnan(ct2["r_within_pooled"]) or abs(ct2["r_within_pooled"]) < 0.1

    ts = pd.date_range("2150-01-01 10:00", periods=60, freq="10s")
    pred = pd.DataFrame({"subject_id": 1, "timestamp": ts, "sbp": np.r_[np.full(30, 100.0), np.full(30, 125.0)],
                         "sbp_pred": np.r_[np.full(30, 110.0), np.full(30, 128.0)]})
    ev = pd.DataFrame({"subject_id": [1], "starttime": [ts[30]], "label": ["phenylephrine"]})
    tbl = evaluation.bolus_response_table(pred, ev, pre=(-180, -30), post=(60, 290))
    assert len(tbl) == 1 and abs(tbl["delta_true"].iloc[0] - 25) < 1e-9 and abs(tbl["delta_pred"].iloc[0] - 18) < 1e-9


def test_moens_korteweg_fit_and_calibration():
    rng = np.random.default_rng(2)
    n_subj, n_win = 40, 25
    subjects = np.repeat(np.arange(n_subj), n_win)
    a_subj = np.repeat(rng.normal(20, 15, n_subj), n_win)      # geometry differs per subject
    ptt = np.exp(rng.normal(np.log(0.25), 0.15, len(subjects)))
    sbp = a_subj - 60.0 * np.log(ptt) + rng.normal(0, 3, len(subjects))   # b = -60 -> alpha = 1/30
    m = models.MoensKortewegPTT().fit(ptt, sbp, subjects)
    assert abs(m.b + 60.0) < 8 and m.alpha > 0
    pop_pred = m.predict(ptt)
    cal = m.calibrate_intercepts(ptt[::n_win], sbp[::n_win], subjects[::n_win])  # one point per subject
    cal_pred = m.predict_calibrated(ptt, subjects, cal)
    assert np.mean(np.abs(cal_pred - sbp)) < np.mean(np.abs(pop_pred - sbp))
    df = pd.DataFrame({"pat_foot": ptt, "hr": rng.normal(70, 10, len(ptt)), "sbp": sbp})
    ridge, cols = models.fit_feature_ridge(df, "sbp", features=("pat_foot", "hr"))
    assert cols == ["pat_foot", "hr"] and np.isfinite(ridge.predict(df[cols].to_numpy())).all()


def test_loader_helpers_without_network():
    assert wfdb_loader.subject_id_from_record("p00/p000020/p000020-2183-04-28-17-47") == 20
    assert wfdb_loader.subject_id_from_record("waves/p100/p10014354/81739927/81739927") == 10014354
    roles = wfdb_loader.signal_roles(["V", "II", "PLETH", "ABP", "RESP"])
    assert roles == {"ecg": 1, "ppg": 2, "abp": 3}
    x = np.random.default_rng(0).normal(size=(125 * 30, 3))
    wins = list(wfdb_loader.iter_array_windows(x, 125.0, "vitaldb", "1", 1, window_s=10, hop_s=5))
    assert len(wins) == 5 and wins[1].start_s == 5.0
    sql = wfdb_loader.mimiciv_bolus_sql("/data/mimiciv")
    assert "inputevents" in sql and "221749" in sql
    X = models.windows_to_tensor(np.stack([w.ecg for w in wins]), np.stack([w.ppg for w in wins]))
    assert X.shape == (5, 2, 1250)


@pytest.mark.skipif(not models.TORCH_AVAILABLE, reason="torch not installed")
def test_cnn_smoke():
    X = np.random.default_rng(0).normal(size=(32, 2, 1250)).astype(np.float32)
    Y = np.random.default_rng(1).normal(size=(32, 2)).astype(np.float32)
    net = models.SmallCNN1D()
    models.train_cnn(net, X, Y, epochs=1, batch_size=16, device="cpu")
    assert models.predict_cnn(net, X, device="cpu").shape == (32, 2)
