"""Synthetic-data tests for fluid_resp (no waveform downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluid_resp import abp_beats as ab  # noqa: E402
from fluid_resp import boluses as bo  # noqa: E402
from fluid_resp import evaluate as ev  # noqa: E402
from fluid_resp import ppv as pv  # noqa: E402


def test_beat_detection_and_ppv_recovery():
    rng = np.random.default_rng(0)
    fs, dur, hr = 125.0, 60.0, 80.0
    t, abp = ab.synthetic_abp(fs=fs, duration_s=dur, hr_bpm=hr, rr_bpm=15, ppv_pct=15.0, rng=rng)
    beats = ab.detect_beats(abp, fs)
    assert abs(len(beats) - hr * dur / 60.0) <= 3
    mask = ab.sqi_mask(beats)
    assert mask.mean() > 0.9
    res = pv.ppv_from_beats(beats, mask)
    assert res["median"] == pytest.approx(15.0, abs=3.0)
    rr = pv.respiratory_rate_from_pp(beats["t_sys"].to_numpy(), beats["pp"].to_numpy())
    assert rr == pytest.approx(15.0, abs=2.0)
    # no modulation -> PPV near zero
    t0, abp0 = ab.synthetic_abp(fs=fs, duration_s=30, hr_bpm=hr, ppv_pct=0.0, noise_sd=0.2, rng=rng)
    b0 = ab.detect_beats(abp0, fs)
    assert pv.ppv_from_beats(b0)["median"] < 3.0
    spv = pv.spv_series(beats["t_sys"].to_numpy(), beats["sbp"].to_numpy())
    assert spv["spv_mmhg"].median() > 0


def test_sqi_flags_flatline_and_irregularity():
    rng = np.random.default_rng(1)
    fs = 125.0
    _, abp = ab.synthetic_abp(fs=fs, duration_s=20, ppv_pct=10, rng=rng)
    abp[int(5 * fs):int(10 * fs)] = 60.0  # 5-s flat line
    assert ab.flatline_fraction(abp, fs) == pytest.approx(0.25, abs=0.06)
    _, irregular = ab.synthetic_abp(fs=fs, duration_s=40, hr_bpm=90, ppv_pct=5, hr_jitter_s=0.2, rng=rng)
    b_irr = ab.detect_beats(irregular, fs)
    _, regular = ab.synthetic_abp(fs=fs, duration_s=40, hr_bpm=90, ppv_pct=5, rng=rng)
    b_reg = ab.detect_beats(regular, fs)
    assert ab.rr_irregularity(b_irr) > ab.rr_irregularity(b_reg)


def _clinical_fixture():
    t0 = pd.Timestamp("2150-01-01 00:00")
    d_items = pd.DataFrame({"itemid": [1, 2, 3, 4], "label": ["NaCl 0.9%", "Albumin 5%", "Packed Red Blood Cells", "Norepinephrine"]})
    ie = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1],
            "itemid": [1, 1, 1, 2, 4],
            "starttime": [t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=6), t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(hours=10, minutes=20), t0 + pd.Timedelta(hours=1)],
            "endtime": [t0 + pd.Timedelta(hours=2, minutes=20), t0 + pd.Timedelta(hours=8), t0 + pd.Timedelta(hours=10, minutes=15), t0 + pd.Timedelta(hours=10, minutes=40), t0 + pd.Timedelta(hours=20)],
            "amount": [500.0, 1000.0, 0.5, 250.0, 10.0],
            "amountuom": ["ml", "ml", "L", "ml", "mg"],
        }
    )
    times = pd.date_range(t0, t0 + pd.Timedelta(hours=24), freq="min")
    rng = np.random.default_rng(3)
    base = 70 + 5 * np.sin(np.arange(len(times)) / 200.0) + rng.normal(0, 1.5, len(times))
    # planted response after the 02:00 bolus
    post = (times >= t0 + pd.Timedelta(hours=2, minutes=20)) & (times < t0 + pd.Timedelta(hours=3))
    base[post] += 12.0
    num = pd.DataFrame({"stay_id": 1, "time": times, "map": base, "pp": 45 + rng.normal(0, 2, len(times)), "hr": 90.0})
    return d_items, ie, num


def test_bolus_identification_response_and_sham():
    d_items, ie, num = _clinical_fixture()
    fluids = bo.resolve_itemids(d_items, bo.FLUID_PATTERNS["crystalloid"] + "|" + bo.FLUID_PATTERNS["colloid"])
    assert set(fluids["itemid"]) == {1, 2}
    b = bo.identify_boluses(ie, fluids["itemid"])
    # 500 mL/20 min yes; 1000 mL over 2 h no; 500 mL (0.5 L)/15 min yes; albumin 250 mL/20 min yes
    assert len(b) == 3 and (b["amount_ml"] >= 250).all()
    iso = b.set_index("start")["isolated"]
    assert iso.iloc[0] and not iso.iloc[1] and not iso.iloc[2]  # the two 10:00 boluses overlap each other's windows
    resp = bo.response_windows(num, b)
    lab = bo.classify_response(resp, "map", 10.0)
    assert bool(lab.iloc[0]) and not bool(lab.iloc[1])
    vaso = bo.resolve_itemids(d_items, bo.VASOPRESSOR_PATTERN)
    flag = bo.vasopressor_change_flag(ie, vaso["itemid"], b)
    assert not flag.any()  # norepinephrine runs continuously across the windows
    sham = bo.sham_windows(num, b, resp, n_per_bolus=2, rng=np.random.default_rng(0))
    assert len(sham) > 0 and (sham["matched_bolus_id"].isin(b["bolus_id"])).all()
    assert (np.abs(sham["pre_map_mean"].to_numpy() - resp.set_index("bolus_id").loc[sham["matched_bolus_id"], "pre_map_mean"].to_numpy()) <= 5.0).all()


def test_prerequisites_and_pbw():
    assert bo.predicted_body_weight(175, male=True) == pytest.approx(70.6, abs=0.1)
    f = bo.prerequisite_flags(560, 70.0, "CMV/AutoFlow", 14, 15, 0.02)
    assert f["all_ok"]
    f2 = bo.prerequisite_flags(420, 70.0, "PSV/SBT", 0, 22, 0.02)
    assert not f2["vt_ok"] and not f2["controlled_mode"] and not f2["all_ok"]


def test_evaluation_helpers():
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.4, 300)
    score = 10 + 6 * y + rng.normal(0, 3, 300)
    groups = rng.integers(0, 60, 300)
    a = ev.auroc_ci(y, score, groups, n_boot=100, rng=rng)
    assert 0.7 < a["auroc"] < 1.0 and a["ci_lo"] <= a["auroc"] <= a["ci_hi"]
    gz = ev.gray_zone(score, y, n_boot=50, rng=rng)
    assert gz["lower"] <= gz["upper"] and 0 <= gz["fraction_inside"] <= 1
    ar = ev.attributable_response(60, 100, 30, 100)
    assert ar["risk_difference"] == pytest.approx(0.3) and ar["attributable_fraction"] == pytest.approx(0.5) and ar["p_value"] < 0.01
    df = pd.DataFrame({"y": y, "s": score, "stratum": rng.integers(0, 2, 300)})
    st = ev.stratified_auroc(df, "s", "y", "stratum", n_boot=20, rng=rng)
    assert len(st) == 2
    ba = ev.bland_altman([10, 12, 14], [11, 11, 15])
    assert ba["n"] == 3 and ba["loa_lo"] < ba["bias"] < ba["loa_hi"]
