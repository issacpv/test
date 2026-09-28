"""Synthetic-data tests for circadian_bias (no PhysioNet data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circadian_bias import circular, clock_ablation, label_timing  # noqa: E402


# ------------------------------------------------------------ circular
def test_rayleigh_uniform_vs_peaked():
    rng = np.random.default_rng(0)
    uni = rng.uniform(0, 24, 500)
    peaked = (rng.vonmises(2 * np.pi * 15 / 24, 4.0, 500) % (2 * np.pi)) * 24 / (2 * np.pi)
    ru, rp = circular.rayleigh_test(uni), circular.rayleigh_test(peaked)
    assert ru["p"] > 0.01 and ru["R"] < 0.15
    assert rp["p"] < 1e-6 and rp["R"] > 0.5 and abs(rp["mean_hour"] - 15) < 1.0
    pt = circular.peak_trough_ratio(peaked)
    assert pt["ratio"] > 3 and abs(pt["peak_hour"] - 15) <= 2
    boot = circular.bootstrap_R(peaked, n_boot=100)
    assert boot["ci_low"] <= boot["R"] <= boot["ci_high"]


def test_shift_change_locking_and_differences():
    rng = np.random.default_rng(1)
    locked = np.concatenate([rng.normal(7, 0.4, 300), rng.normal(19, 0.4, 300)]) % 24
    res = circular.shift_change_locking(locked, n_perm=300)
    assert res["ratio"] > 3 and res["p_binomial"] < 1e-6 and 0 < res["p_rotation"] <= 1
    flat = circular.shift_change_locking(rng.uniform(0, 24, 600), n_perm=300)
    assert flat["p_binomial"] > 0.01 and flat["p_rotation"] > 0.01
    d = circular.circular_difference_hours(np.array([23.0, 1.0]), np.array([1.0, 23.0]))
    assert np.allclose(d, [2.0, -2.0])
    ev = pd.DataFrame({"event_type": ["a"] * 300 + ["b"] * 300, "clock_hour": np.concatenate([locked[:300], rng.uniform(0, 24, 300)])})
    audit = circular.timing_audit(ev, min_n=50)
    assert set(audit["event_type"]) == {"a", "b"} and audit.set_index("event_type").loc["a", "R"] > audit.set_index("event_type").loc["b", "R"]


# ------------------------------------------------------------ label timing
def _synthetic_stays(n: int = 400, seed: int = 2):
    rng = np.random.default_rng(seed)
    admit = pd.to_datetime("2150-01-01") + pd.to_timedelta(rng.uniform(0, 24 * 30, n), unit="h")
    los_h = rng.gamma(3, 24, n)
    raw_out = admit + pd.to_timedelta(los_h, unit="h")
    # discharge is rounded to the next 14:00 (rounds) -> strong clock structure
    out = raw_out.normalize() + pd.Timedelta(hours=14)
    out = out.where(out > raw_out, out + pd.Timedelta(days=1))
    return pd.DataFrame({"stay_id": np.arange(n), "admit": admit, "icu_discharge": out, "raw_discharge": raw_out})


def test_event_table_and_delays():
    st = _synthetic_stays()
    frames = {"icu_discharge": pd.DataFrame({"stay_id": st.stay_id, "event_time": st.icu_discharge,
                                             "storetime": st.icu_discharge + pd.Timedelta(minutes=30)})}
    ev = label_timing.event_table_from_frames(frames)
    assert (ev["clock_hour"].round(3) == 14.0).all() and (ev["store_delay_min"] == 30).all()
    labs = pd.DataFrame({"charttime": st.admit, "storetime": st.admit + pd.to_timedelta(np.where(st.admit.dt.hour < 8, 90, 20), unit="m"),
                         "flag": np.where(np.arange(len(st)) % 2 == 0, "abnormal", None)})
    d = label_timing.documentation_delay_by_hour(labs, abnormal_col="flag")
    night = d[d["hour_bin"] < 8]["median"].mean()
    day = d[d["hour_bin"] >= 8]["median"].mean()
    assert night > day


def test_windowed_labels_and_phase_randomisation():
    st = _synthetic_stays()
    rng = np.random.default_rng(3)
    pt = st["admit"] + pd.Timedelta(hours=24)
    y = label_timing.windowed_label(st["icu_discharge"], pt, horizon_h=24)
    assert set(y.unique()) <= {0, 1} and 0 < y.mean() < 1
    shifted = label_timing.phase_randomize(st["icu_discharge"], rng, mode="uniform_24h", group=st["stay_id"])
    assert abs((shifted - st["icu_discharge"]).dt.total_seconds().abs().max() / 3600.0) <= 12.0
    R_before = circular.resultant_vector(circular.clock_hour(st["icu_discharge"]).to_numpy())[0]
    R_after = circular.resultant_vector(circular.clock_hour(shifted).to_numpy())[0]
    assert R_before > 0.9 and R_after < 0.3
    flips = label_timing.label_flip_rate(st["icu_discharge"], pt, horizon_h=24, n_rep=30, group=st["stay_id"])
    assert flips["flip_rate"] > 0.05
    same_day = label_timing.phase_randomize(st["icu_discharge"], rng, mode="same_day")
    assert (same_day.dt.normalize() == st["icu_discharge"].dt.normalize()).all()
    haz = label_timing.hazard_by_hour(st["icu_discharge"], [np.arange(0, 48) for _ in range(len(st))])
    assert haz.loc[14, "rate_per_hour"] == haz["rate_per_hour"].max()
    kernel = np.zeros(24)
    kernel[0], kernel[1], kernel[2] = 0.5, 0.3, 0.2
    truth = np.zeros(24)
    truth[10] = 100.0
    observed = np.real(np.fft.ifft(np.fft.fft(truth) * np.fft.fft(kernel)))
    est = label_timing.richardson_lucy_circular(observed, kernel, n_iter=200)
    assert int(np.argmax(est)) == 10


# ------------------------------------------------------------ clock ablation
def test_clock_ablation_detects_clock_dependent_label():
    rng = np.random.default_rng(4)
    n = 1500
    admit = pd.to_datetime("2150-01-01") + pd.to_timedelta(rng.uniform(0, 24 * 30, n), unit="h")
    pt = admit + pd.to_timedelta(rng.uniform(6, 48, n), unit="h")
    phys = pd.DataFrame({"hr": rng.normal(90, 15, n), "map": rng.normal(75, 10, n), "lactate": rng.gamma(2, 1, n)})
    clk = clock_ablation.clock_features(pt, admit)
    assert list(clk.columns) == clock_ablation.CLOCK_COLUMNS
    hour = circular.clock_hour(pt).to_numpy()
    logit = 0.03 * (phys["hr"] - 90) + 0.5 * phys["lactate"] - 2.0 + 1.5 * clk["hour_cos"]  # label depends on the clock
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    res = clock_ablation.clock_ablation(phys, clk, y, groups=np.arange(n), hour=hour, n_splits=5, hour_bins=4)
    assert res.clock_share > 0.05
    assert res.clock_increment > 0.02
    assert set(res.by_hour["feature_set"]) == {"physiology_only", "physiology_plus_clock", "clock_only"}
    d = clock_ablation.paired_bootstrap_delta(y.to_numpy(), res.predictions["physiology_plus_clock"], res.predictions["physiology_only"], n_boot=100)
    assert d["ci_low"] <= d["delta"] <= d["ci_high"]
    probe = clock_ablation.clock_probe(phys, hour, groups=np.arange(n))
    assert probe < 0.2  # physiology here carries no clock information
