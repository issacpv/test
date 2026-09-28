"""Synthetic-data tests for sepsis_multiverse (no PhysioNet data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sepsis_multiverse import benchmark, multiverse as mv, sofa, suspected_infection as si  # noqa: E402


def make_hourly(n_stays: int = 40, n_hours: int = 72, seed: int = 0):
    """Synthetic hourly table: half the stays deteriorate (SOFA rises) around hour 30."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stays):
        septic = s % 2 == 0
        for h in range(-12, n_hours):
            det = septic and h >= 30
            rows.append({"stay_id": s, "hour": h,
                         "pao2": rng.normal(90, 5) if h % 6 == 0 else np.nan, "fio2": 0.4 if det else 0.21,
                         "vent": 1.0 if det else 0.0,
                         "platelets": (60 if det else 250) + rng.normal(0, 5) if h % 12 == 0 else np.nan,
                         "bilirubin": 3.0 if det else 0.5, "map": 60 if det else 80,
                         "norepinephrine": 0.2 if det else 0.0, "epinephrine": 0.0, "dopamine": 0.0, "dobutamine": 0.0,
                         "gcs": 13 if det else 15, "creatinine": (2.5 if det else 0.9) if h % 24 == 0 else np.nan,
                         "urine_output": 10.0 if det else 60.0})
    hourly = pd.DataFrame(rows).set_index(["stay_id", "hour"])
    abx = pd.DataFrame({"stay_id": [s for s in range(n_stays) if s % 2 == 0] + [1, 3],
                        "time_h": [28.0] * (n_stays // 2) + [5.0, 50.0], "drug": "vancomycin"})
    cx = pd.DataFrame({"stay_id": [s for s in range(n_stays) if s % 2 == 0] + [1],
                       "time_h": [27.0] * (n_stays // 2) + [4.0], "specimen": "blood"})
    return hourly, abx, cx


# ------------------------------------------------------------ sofa
def test_sofa_components_and_total():
    hourly, _, _ = make_hourly(n_stays=2)
    out = sofa.hourly_sofa(hourly)
    healthy = out.xs(1, level=0)
    sick = out.xs(0, level=0)
    assert healthy["sofa"].max() <= 1  # occasional NaN components score 0 under assume_normal
    assert sick.loc[40, "sofa"] >= 10
    assert sick.loc[40, "cardiovascular"] == 4 and sick.loc[40, "cns"] == 1 and sick.loc[40, "liver"] == 2
    assert sick.loc[10, "sofa"] <= 1
    strict = sofa.hourly_sofa(hourly, sofa.SofaConfig(assume_normal=False, carry_forward_h={k: 0 for k in sofa.DEFAULT_CARRY_FORWARD_H}))
    assert strict["sofa"].isna().mean() > 0.5  # without carry-forward most hours lack a lab value


def test_sofa_baseline_methods():
    s = pd.Series([1, 1, 2, 3, 4, 6], index=[0, 1, 2, 3, 4, 5])
    assert sofa.sofa_baseline(s, 4, "zero") == 0
    assert sofa.sofa_baseline(s, 4, "min_lookback", lookback_h=2) == 2  # hours 2 and 3 -> min(2, 3)
    assert sofa.sofa_baseline(s, 4, "min_prior") == 1
    assert sofa.sofa_baseline(s, 4, "first_icu") == 1
    with pytest.raises(ValueError):
        sofa.sofa_baseline(s, 4, "bogus")


# ------------------------------------------------------------ suspected infection
def test_si_rules_and_windows():
    abx = pd.DataFrame({"stay_id": [1, 1, 2, 3, 3], "time_h": [10.0, 40.0, 5.0, 0.0, 90.0], "drug": ["a", "a", "b", "c", "c"]})
    cx = pd.DataFrame({"stay_id": [1, 2, 3], "time_h": [12.0, 100.0, 60.0], "specimen": "blood"})
    t = si.suspected_infection_times(abx, cx, si.SIRule("culture_abx"))
    assert t[1] == 10.0 and 2 not in t.index  # culture 95 h after abx: outside 24 h abx-first window
    # stay 3: abx at 0 and culture at 60 exceed the 24-h abx-first window; culture at 60 then abx at 90 is a
    # valid culture-first pair (30 h <= 72 h), so the suspicion time is the earlier of the pair = 60
    assert t[3] == 60.0
    t_abx = si.suspected_infection_times(abx, cx, si.SIRule("culture_abx", suspicion_time="antibiotic"))
    assert t_abx[3] == 90.0
    t_only = si.suspected_infection_times(abx, None, si.SIRule("abx_only"))
    assert t_only[2] == 5.0
    t_2p = si.suspected_infection_times(abx, None, si.SIRule("abx_2plus", repeat_window_h=96))
    assert t_2p[1] == 10.0 and t_2p[3] == 0.0 and 2 not in t_2p.index
    with pytest.raises(ValueError):
        si.SIRule("nonsense")
    filt = si.filter_antibiotics(pd.DataFrame({"stay_id": [1, 1], "time_h": [1, 2], "drug": ["Vancomycin IV", "bacitracin"],
                                               "route": ["iv", "topical"]}), allowed={"vancomycin", "bacitracin"})
    assert len(filt) == 1


# ------------------------------------------------------------ multiverse
def test_label_sepsis_and_grid_geometry():
    hourly, abx, cx = make_hourly()
    lab = mv.label_sepsis(hourly, abx, cx, mv.SepsisSpec())
    septic = lab.onset_h.notna()
    assert septic.sum() == 20 and (lab.onset_h[septic] == 27.0).all()  # onset = suspicion time (earlier of pair)
    lab2 = mv.label_sepsis(hourly, abx, cx, mv.SepsisSpec(onset_convention="sofa_rise"))
    assert (lab2.onset_h.dropna() == 30.0).all()
    lab3 = mv.label_sepsis(hourly, abx, cx, mv.SepsisSpec(sofa_baseline="min_lookback", sofa_delta=20))
    assert lab3.onset_h.notna().sum() == 0
    grid = mv.default_grid()
    assert len(grid) == 288 and len({s.name() for s in grid}) == len(grid)
    specs = [mv.SepsisSpec(), mv.SepsisSpec(onset_convention="sofa_rise"), mv.SepsisSpec(si_rule="abx_only", suspicion_time="antibiotic"),
             mv.SepsisSpec(sofa_delta=20)]
    mort = pd.Series((np.arange(40) % 4 == 0).astype(float), index=pd.Index(range(40), name="stay_id"))
    summary, onsets = mv.run_grid(hourly, abx, cx, specs, mortality=mort)
    assert list(summary["n_septic"]) == [20, 20, 20, 0]
    J = mv.pairwise_jaccard(onsets)
    assert J.iloc[0, 1] == 1.0 and np.isnan(J.iloc[0, 3]) or J.iloc[0, 3] == 0.0
    shift = mv.onset_shift(onsets[specs[0].name()], onsets[specs[1].name()])
    assert shift["median_shift_h"] == 3.0 and shift["n_both"] == 20
    dec = mv.variance_decomposition(summary, "n_septic", dimensions=["onset_convention", "sofa_delta", "si_rule"])
    assert dec.iloc[0]["term"] == "sofa_delta"
    curve = mv.specification_curve(summary, "n_septic")
    assert list(curve["rank"]) == [1, 2, 3, 4]


# ------------------------------------------------------------ benchmark
def test_early_warning_labels_are_leakage_safe_and_transfer_matrix_runs():
    hourly, abx, cx = make_hourly(n_stays=60)
    lab = mv.label_sepsis(hourly, abx, cx, mv.SepsisSpec())
    y, keep = benchmark.early_warning_labels(hourly.index, lab.onset_h, horizon_h=12)
    hours = hourly.index.get_level_values(1)
    stays = hourly.index.get_level_values(0)
    on = lab.onset_h.reindex(stays).to_numpy()
    assert not keep[np.isfinite(on) & (hours >= on)].any()  # no post-onset rows kept
    assert y[(stays == 0) & (hours == 20)].item() == 1 and y[(stays == 0) & (hours == 10)].item() == 0
    X = benchmark.hourly_features(hourly, concepts=["map", "platelets", "gcs"])
    split = benchmark.patient_split(stays.unique(), seed=1)
    lab_b = mv.label_sepsis(hourly, abx, cx, mv.SepsisSpec(onset_convention="sofa_rise"))
    yb, keepb = benchmark.early_warning_labels(hourly.index, lab_b.onset_h, horizon_h=12)
    M = benchmark.definition_transfer_matrix(X, {"a": y, "b": yb}, {"a": keep, "b": keepb}, split)
    assert M.shape == (2, 2) and np.isfinite(M.to_numpy()).all()
    table = pd.DataFrame({"lr": [0.8, 0.82, 0.78], "gbm": [0.85, 0.86, 0.84], "gru": [0.83, 0.8, 0.81]})
    st = benchmark.ranking_stability(table)
    assert 0 <= st["kendall_w"] <= 1
    rr = benchmark.range_across_vs_within(table)
    assert rr["ratio"] > 0
    ev = benchmark.evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]))
    assert ev["auroc"] == 1.0
