"""Synthetic-data tests for cuffless_preg (no real data required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cuffless_preg import cohort, metrics, ptt_model, signals


def _synthetic_record(fs: float = 125.0, dur_s: float = 40.0, hr: float = 80.0, pat_s: float = 0.25, seed: int = 0):
    """ECG-like spikes, PPG-like pulses delayed by pat_s, and an ABP-like waveform."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur_s, 1 / fs)
    ecg = 0.02 * rng.standard_normal(t.size)
    ppg = np.zeros_like(t)
    abp = np.zeros_like(t)
    beat_times = np.arange(0.5, dur_s - 1.0, 60.0 / hr)
    for bt in beat_times:
        ecg += np.exp(-((t - bt) ** 2) / (2 * 0.008**2))  # narrow R wave
        # PPG pulse: fast rise (gamma-like) starting at bt + pat
        u = t - (bt + pat_s)
        pulse = np.where(u >= 0, (u / 0.12) * np.exp(1 - u / 0.12), 0.0)
        ppg += pulse
        # ABP pulse starting a bit earlier (proximal), SBP 120 / DBP 70
        ua = t - (bt + 0.15)
        abp += np.where(ua >= 0, (ua / 0.10) * np.exp(1 - ua / 0.10), 0.0) * 50.0
    abp += 70.0
    ppg += 0.01 * rng.standard_normal(t.size)
    return ecg, ppg, abp


def test_r_peaks_and_pat_recovered():
    fs = 125.0
    ecg, ppg, abp = _synthetic_record(fs=fs, pat_s=0.25)
    r = signals.detect_r_peaks(ecg, fs)
    assert 45 <= len(r) <= 60  # ~52 beats at 80 bpm in 39 s
    tab = signals.build_beat_table(ecg, ppg, abp, fs)
    assert not tab.empty
    med_pat = tab["pat_s"].median()
    assert abs(med_pat - 0.25) < 0.03
    assert tab["hr_bpm"].median() == pytest.approx(80.0, abs=2.0)
    # ABP matched: SBP ~ 120, DBP ~ 70
    assert tab["sbp"].dropna().median() == pytest.approx(120.0, abs=5.0)
    assert tab["dbp"].dropna().median() == pytest.approx(70.0, abs=5.0)
    # proximal PAT (R -> ABP foot) should be ~0.15 s, shorter than PPG PAT
    assert tab["pat_prox_s"].dropna().median() < med_pat
    assert tab["good"].mean() > 0.8


def test_pregnancy_flags_icd9_and_icd10():
    diag9 = pd.DataFrame({"hadm_id": [1, 1, 2, 3], "icd9_code": ["6425", "V270", "4019", "65971"]})
    f9 = cohort.flag_pregnancy(diag9).set_index("hadm_id")
    assert f9.loc[1, "pregnant"] and f9.loc[1, "hdp"] and f9.loc[1, "hdp_severe"]
    assert not f9.loc[2, "pregnant"]
    assert f9.loc[3, "pregnant"] and not f9.loc[3, "hdp"]
    diag10 = pd.DataFrame({"hadm_id": [10, 11, 12], "icd_code": ["O14.12", "Z37.0", "I10"], "icd_version": [10, 10, 10]})
    f10 = cohort.flag_pregnancy(diag10).set_index("hadm_id")
    assert f10.loc[10, "hdp_severe"] and f10.loc[11, "pregnant"] and not f10.loc[12, "pregnant"]
    grp = cohort.assign_group(f10)
    assert set(grp) == {"hdp", "pregnant_normotensive", "control"}


def test_matched_controls_respect_strata():
    adm = pd.DataFrame(
        {
            "hadm_id": range(1, 13),
            "subject_id": range(1, 13),
            "gender": ["F"] * 12,
            "age": [30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 30, 25],
            "first_careunit": ["MICU"] * 5 + ["SICU"] * 5 + ["MICU", "MICU"],
            "ever_pregnant": [True] + [False] * 11,
        }
    )
    m = cohort.select_matched_controls(adm, pd.Series([1]), ratio=3, age_band=5)
    assert len(m) == 3
    assert (m["stratum"] == "6|MICU").all()
    assert 1 not in set(m["control_hadm_id"])


def test_tier_assignment():
    ri = pd.DataFrame(
        {
            "record": ["a", "b", "c"],
            "subject_id": [1, 2, 3],
            "signals": [("II", "PLETH", "ABP"), ("II", "PLETH"), ("PLETH", "ABP")],
            "duration_min": [30, 30, 30],
            "n_nibp": [0, 10, 10],
        }
    )
    out = cohort.tier_records(ri)
    assert list(out["tier"]) == ["A", "B", ""]


def test_mixed_model_detects_slope_shift():
    rng = np.random.default_rng(1)
    rows = []
    for sid in range(24):
        group = "control" if sid < 12 else "hdp"
        slope = -60.0 if group == "control" else -90.0
        a = rng.normal(0, 4)
        lp = rng.normal(0, 0.12, 150)
        hr = rng.normal(80, 8, 150)
        sbp = 120 + a + slope * lp + 0.1 * (hr - 80) + rng.normal(0, 3, 150)
        rows.append(pd.DataFrame({"subject_id": sid, "group": group, "log_pat": lp, "hr_bpm": hr, "sbp": sbp}))
    df = pd.concat(rows, ignore_index=True)
    res = ptt_model.fit_mixed_pat_model(df, random_slope=False)
    tab = ptt_model.interaction_table(res)
    slope_shift = tab[(tab["group"] == "hdp") & (tab["term"] == "slope_shift")]["estimate"].iloc[0]
    assert -45 < slope_shift < -15
    cal = ptt_model.fit_subject_calibrations(df)
    assert len(cal) == 24 and (cal["slope"] < 0).all()


def test_error_decomposition_and_calibration():
    rng = np.random.default_rng(2)
    ref = rng.normal(120, 10, 500)
    pred = 0.7 * ref + 30 + rng.normal(0, 2, 500)  # biased and mis-scaled
    dec = ptt_model.error_decomposition(pred, ref)
    assert dec["bias_frac"] + dec["slope_frac"] + dec["residual_part"] / dec["mse_total"] == pytest.approx(1.0, abs=1e-9)
    assert dec["slope_part"] > 0
    t = np.arange(500) * 1.0
    corr, mask = ptt_model.one_point_calibration(pred, ref, t, calib_window_s=60)
    assert mask.sum() == 500 - 61
    assert abs(np.mean(ref[mask] - corr[mask])) < abs(np.mean(ref[mask] - pred[mask]))


def test_mechanistic_shift_signs():
    tab = ptt_model.simulated_calibration_shift().set_index("group")
    # pregnancy: more compliant -> longer PAT at same pressure -> positive SBP offset at reference PAT
    assert tab.loc["pregnant_normotensive", "pat_ms_at_130"] > tab.loc["control", "pat_ms_at_130"]
    assert tab.loc["pregnant_normotensive", "sbp_offset_at_ref_pat"] > 0
    # preeclampsia: stiffer -> shorter PAT, steeper (more negative) linear slope (mmHg per ms);
    # the log-slope depends on alpha only and is unchanged when only E0 is scaled
    assert tab.loc["hdp", "pat_ms_at_130"] < tab.loc["control", "pat_ms_at_130"]
    assert abs(tab.loc["hdp", "slope_lin_per_ms"]) > abs(tab.loc["control", "slope_lin_per_ms"])
    assert tab.loc["hdp", "sbp_offset_at_ref_pat"] < 0
    assert tab.loc["hdp", "slope_ln"] == pytest.approx(tab.loc["control", "slope_ln"], rel=0.15)


def test_metrics():
    rng = np.random.default_rng(3)
    err = rng.normal(2.0, 6.0, 1000)
    subj = np.repeat(np.arange(20), 50)
    c1 = metrics.iso81060_criterion1(err)
    assert c1["pass"] and abs(c1["me"] - 2.0) < 1.0
    c2 = metrics.iso81060_criterion2(err, subj)
    assert c2["pass"] and 4.7 < c2["sd_limit"] <= 6.95
    g = metrics.bhs_grade(rng.normal(0, 3, 1000))
    assert g["grade"] == "A"
    drift = metrics.calibration_drift(err, np.arange(1000) * 3.0, bin_s=600)
    assert "abs_err_slope_per_hour" in drift.attrs and len(drift) == 5
    df = pd.DataFrame({"subject_id": subj, "err": err})
    b = metrics.bootstrap_by_subject(df, lambda d: float(d["err"].mean()), n_boot=100)
    assert b["ci_low"] <= b["estimate"] <= b["ci_high"]
    per = pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10, "v": np.r_[rng.normal(0, 1, 10), rng.normal(3, 1, 10)]})
    p = metrics.permutation_group_difference(per, "v", "group", "a", "b", n_perm=500)
    assert p["p_value"] < 0.05
