"""Synthetic Windkessel tests for pulse_contour (no downloads, no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulse_contour import agreement, beats, estimators, references  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    return beats.synthetic_abp(fs=125, n_beats=60, hr=80, sv_ml=70, sv_variation=0.25, seed=1)


def test_synthetic_waveform_is_physiologic(synth):
    assert 90 < synth.abp.max() < 200 and 40 < synth.abp.min() < 100
    assert len(synth.onsets) == 60 and np.all(np.diff(synth.onsets) > 0.5 * synth.fs)


def test_onset_detection_matches_true_onsets(synth):
    det = beats.detect_onsets(synth.abp, synth.fs)
    assert abs(len(det) - len(synth.onsets)) <= 1
    # every true onset (except possibly the last) has a detection within 60 ms
    tol = int(0.06 * synth.fs)
    hits = [np.min(np.abs(det - o)) <= tol for o in synth.onsets[:-1]]
    assert np.mean(hits) > 0.95


def test_beat_features_and_estimators_track_true_sv(synth):
    f = beats.beat_features(synth.abp, synth.fs, synth.onsets)
    assert len(f) == len(synth.onsets) - 1 and f["ok"].mean() > 0.95
    assert (f["sbp"] > f["map"]).all() and (f["map"] > f["dbp"]).all() and (f["t_sys"] > 0.1).all()
    assert np.nanmedian(f["tau"]) == pytest.approx(1.5, rel=0.35)  # R*C of the generator
    sv_true = synth.sv_true[: len(f)]
    for method in estimators.METHODS:
        est = estimators.estimate(f, method)
        rho = spearmanr(est, sv_true, nan_policy="omit").correlation
        assert rho > 0.8, (method, rho)
    cal = estimators.calibrate(estimators.estimate(f, "systolic_area"), sv_true, "ratio")
    sv_cal = cal.apply(estimators.estimate(f, "systolic_area"))
    assert abs(np.nanmean(sv_cal) - sv_true.mean()) < 1e-6
    coef = estimators.fit_impedance_correction(f, sv_true)
    corr = estimators.corrected_area(f, coef)
    assert spearmanr(corr, sv_true).correlation > 0.8
    w = estimators.beat_to_window(f, sv_cal, synth.fs, window_s=10.0)
    assert {"t_center", "sv", "hr", "n_beats"} <= set(w.columns) and w["n_beats"].sum() == len(f)


def test_agreement_statistics():
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(20), 5)
    ref = 70 + 10 * rng.normal(size=100)
    est = ref + 3 + 4 * rng.normal(size=100) + np.repeat(rng.normal(0, 2, 20), 5)
    ba = agreement.bland_altman(ref, est, subj)
    assert 1 < ba.bias < 5 and ba.loa_low < ba.bias < ba.loa_high and ba.n_subjects == 20
    assert ba.var_between >= 0 and ba.var_within > 0
    ba_simple = agreement.bland_altman(ref, est)
    assert abs(ba_simple.bias - ba.bias) < 1e-9
    pe = agreement.percentage_error(ba, ref.mean())
    assert 5 < pe < 40
    perfect = agreement.bland_altman(ref, ref)
    assert agreement.percentage_error(perfect, ref.mean()) == 0.0
    df = pd.DataFrame({"s": subj, "t": np.tile(np.arange(5), 20), "ref": ref, "est": est})
    d = agreement.trend_deltas(df, "s", "t", "ref", "est")
    assert len(d) == 80
    fq = agreement.four_quadrant(d["d_ref"], d["d_est"], exclusion=5.0)
    assert fq["n_included"] <= fq["n_total"] and fq["concordance_pct"] > 60
    same = agreement.four_quadrant(d["d_ref"], d["d_ref"], exclusion=1.0)
    assert same["concordance_pct"] == 100.0
    pol = agreement.polar_concordance(d["d_ref"], d["d_ref"], exclusion=1.0)
    assert pol["angular_bias_deg"] == pytest.approx(0.0, abs=1e-9) and pol["radial_loa_pct"] == 100.0
    point, lo, hi = agreement.bootstrap_by_subject(lambda x: agreement.bland_altman(x["ref"], x["est"]).bias, df, "s", n_boot=50)
    assert lo <= point <= hi


def test_echo_parsing_and_reference_helpers():
    report = "Aortic Valve - LVOT diam: 2.0 cm\nAortic Valve - LVOT VTI: 20 cm\nLeft Ventricle - Ejection Fraction: >= 55 %"
    m = references.parse_echo_measurements(report)
    assert m["lvot_diam_cm"] == 2.0 and m["lvot_vti_cm"] == 20.0 and m["lvef_pct"] == 55.0
    assert references.echo_reference(m) == pytest.approx(np.pi * 1.0 * 20.0)
    assert references.parse_echo_measurements("LVOT diameter = 21 mm")["lvot_diam_cm"] == 2.1
    assert np.isnan(references.echo_reference({}))
    d_items = pd.DataFrame({"itemid": [1, 2, 3], "label": ["Cardiac Output (thermodilution)", "Heart Rate", "Cardiac Output (CCO)"],
                            "category": ["Hemodynamics"] * 3})
    co = references.cardiac_output_items(d_items)
    assert co["itemid"].tolist() == [1, 3]
    est = pd.DataFrame({"t_center": np.arange(0, 600, 20.0), "sv": 60 + np.arange(30)})
    ref = pd.DataFrame({"t": [100.0, 400.0, 5000.0], "sv_ref": [65.0, 80.0, 90.0]})
    al = references.align_reference(est, ref, window_s=30.0)
    assert len(al) == 2 and al["sv_est"].iloc[0] == pytest.approx(65.0, abs=1.0)
