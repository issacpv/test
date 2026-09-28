"""Synthetic-data tests for doa_xfer (no network, no VitalDB access)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doa_xfer import eeg_features, targets, transfer_eval, vitaldb_io  # noqa: E402


def colored_noise(n: int, fs: float, exponent: float, rng) -> np.ndarray:
    w = rng.normal(size=n)
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = np.fft.rfft(w) / np.sqrt(np.maximum(f, f[1]) ** exponent)
    spec[f < 0.5] = 0
    x = np.fft.irfft(spec, n)
    return 20 * (x - x.mean()) / (x.std() + 1e-9)


def synthetic_anesthesia_epochs(depth: np.ndarray, fs: float = 128.0, seed: int = 0) -> np.ndarray:
    """Deeper -> more alpha/delta, steeper aperiodic slope; depth in [0, 1]."""
    rng = np.random.default_rng(seed)
    L = int(4 * fs)
    t = np.arange(L) / fs
    out = np.zeros((depth.size, L))
    for i, d in enumerate(depth):
        x = colored_noise(L, fs, 1.0 + 1.5 * d, rng)
        x += 30 * d * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        x += 40 * d * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2 * np.pi))
        out[i] = x
    return out


def test_case_selection_and_alignment():
    cases = [dict(caseid="1", subjectid="a", age="55", ane_type="General"),
             dict(caseid="2", subjectid="b", age="60", ane_type="General"),
             dict(caseid="3", subjectid="c", age="12", ane_type="General"),
             dict(caseid="4", subjectid="d", age="70", ane_type="Spinal")]
    trks = [dict(caseid="1", tname="BIS/EEG1_WAV", tid="t1"), dict(caseid="1", tname="Orchestra/PPF20_CE", tid="t2"),
            dict(caseid="2", tname="BIS/EEG1_WAV", tid="t3"), dict(caseid="2", tname="Primus/EXP_SEVO", tid="t4"),
            dict(caseid="3", tname="BIS/EEG1_WAV", tid="t5"), dict(caseid="3", tname="Primus/EXP_SEVO", tid="t6"),
            dict(caseid="4", tname="BIS/EEG1_WAV", tid="t7"), dict(caseid="4", tname="Primus/EXP_SEVO", tid="t8")]
    sel = vitaldb_io.select_cases(cases, trks)
    assert {(r["caseid"], r["agent"]) for r in sel} == {("1", "propofol"), ("2", "sevoflurane")}
    assert vitaldb_io.agent_of({"Orchestra/PPF20_CE", "Primus/EXP_SEVO"}) == "mixed"
    df = pd.DataFrame({"t": [0, 10, 20, 100], "v": [1.0, 2.0, np.nan, 4.0]})
    grid = np.arange(0, 130, 5.0)
    a = vitaldb_io.align_numeric(df, grid, max_gap_s=30)
    assert a[0] == 1.0 and a[2] == 2.0 and a[4] == 2.0 and np.isnan(a[12]) and a[20] == 4.0
    ep, t0 = vitaldb_io.epoch_eeg(np.arange(128 * 10, dtype=float), 128.0)
    assert ep.shape == (4, 512) and t0[1] == 2.0
    flags = vitaldb_io.artifact_flags(np.vstack([np.zeros(512), np.full(512, 600.0), np.random.default_rng(0).normal(size=512)]))
    assert flags.tolist() == [True, True, False]
    assert vitaldb_io.frontal_channel_picks(["Fp1", "Cz", "F8", "O1"]) == [0, 2]


def test_features_track_depth():
    depth = np.linspace(0, 1, 40)
    ep = synthetic_anesthesia_epochs(depth)
    f = eeg_features.epoch_features(ep, 128.0)
    assert set(eeg_features.SIX_FEATURES) <= set(f)
    assert np.corrcoef(f["log_alpha"], depth)[0, 1] > 0.8
    assert np.corrcoef(f["aperiodic_exponent"], depth)[0, 1] > 0.6
    assert np.corrcoef(f["sef95"], depth)[0, 1] < -0.5
    # burst suppression ratio: half the epoch flat
    x = np.r_[np.zeros(256), 50 * np.sin(np.linspace(0, 40, 256))]
    assert eeg_features.burst_suppression_ratio(x, 128.0) == pytest.approx(0.5, abs=0.01)
    f_, P_ = eeg_features.welch_psd(colored_noise(4096, 128.0, 2.0, np.random.default_rng(1))[None], 128.0)
    expo, _ = eeg_features.aperiodic_fit(f_, P_[0])
    assert 1.5 < expo < 2.6
    sm = eeg_features.smooth_trajectory(np.r_[np.ones(10), np.nan, np.ones(10)], 2.0, 10.0)
    assert np.isfinite(sm).all()


def test_targets():
    assert targets.mac_age_adjusted(40) == pytest.approx(1.80)
    assert targets.mac_age_adjusted(80) < targets.mac_age_adjusted(20)
    assert targets.mac_fraction(np.array([1.8]), 40)[0] == pytest.approx(1.0)
    assert targets.normalized_propofol(np.array([3.0]))[0] == 1.0
    step = 2.0
    proxy = np.sin(np.linspace(0, 20, 300))
    bis = targets.shift_series(proxy, -20.0, step)  # BIS lags proxy by 20 s
    assert abs(targets.estimate_bis_lag(bis, proxy, step) - 20.0) <= step
    t = np.arange(0, 7200, 60.0)
    ph = targets.phase_labels(t, anestart=600, opstart=1500, opend=6000, aneend=6600)
    assert ph[0] == "pre" and ph[15] == "induction" and ph[50] == "maintenance" and ph[105] == "emergence" and ph[-1] == "post"
    depth = np.where((t > 900) & (t < 6300), 0.8, 0.1) + 0.01 * np.random.default_rng(0).normal(size=t.size)
    tt = targets.transition_times(t, depth, ph)
    assert 840 <= tt["loc_s"] <= 1020 and 6180 <= tt["roc_s"] <= 6360
    bt = targets.bis_threshold_times(t, np.where(depth > 0.5, 40, 95))
    assert bt["loc_s"] == tt["loc_s"]


def test_pk_and_transfer_eval():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, 500)
    assert transfer_eval.prediction_probability(y, y) == 1.0
    assert transfer_eval.prediction_probability(-y, y) == 0.0
    assert abs(transfer_eval.prediction_probability(rng.normal(size=500), y) - 0.5) < 0.08
    assert transfer_eval.prediction_probability(np.zeros(500), y) == pytest.approx(0.5)
    assert transfer_eval.lin_ccc(y, y) == pytest.approx(1.0)
    ba = transfer_eval.bland_altman(y + 0.1, y)
    assert ba["bias"] == pytest.approx(0.1)
    jk = transfer_eval.pk_jackknife([y[:250], y[250:]], [y[:250] + 0.01 * rng.normal(size=250), y[250:]])
    assert jk["n_cases"] == 2 and 0.9 < jk["pk"] <= 1.0
    # agent transfer on synthetic features: arm B has a shifted feature scale
    def arm(shift, seed):
        d = rng.uniform(0, 1, 300)
        X = np.column_stack([d + 0.2 * rng.normal(size=300) + shift, rng.normal(size=300)])
        return dict(X=X, y=d, case=np.repeat(np.arange(10), 30) + seed * 100)
    tab = transfer_eval.agent_transfer_table({"propofol": arm(0.0, 1), "sevoflurane": arm(0.5, 2)}, n_splits=3)
    assert len(tab) == 4 and (tab["pk"] > 0.8).all()
    drop = transfer_eval.transfer_drop(tab)
    assert set(drop["train"]) == {"propofol", "sevoflurane"}
    strat = transfer_eval.age_stratified(transfer_eval.prediction_probability, y, y, rng.uniform(20, 90, 500))
    assert strat["n"].sum() == 500
    null = transfer_eval.label_permutation_null(y, y, np.repeat(np.arange(10), 50), n_perm=50)
    assert null["p"] < 0.05 and null["null_mean"] < 0.7
    assert transfer_eval.calibration_slope(y, 2 * y) == pytest.approx(2.0)
