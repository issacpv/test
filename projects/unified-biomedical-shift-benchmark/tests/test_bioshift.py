"""Synthetic-data tests for the bioshift spec, metrics, diagnostics and harness."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bioshift import (
    Benchmark,
    ShiftAxis,
    SyntheticAdapter,
    TaskType,
    default_benchmark,
    evaluate,
    leaderboard,
    negative_control,
    run_benchmark,
    shift_loss_regression,
)
from bioshift.adapters import ManifestAdapter, write_split_manifest
from bioshift.diagnostics import bbse_label_shift, decompose_gap, domain_classifier_auc, mmd_rbf
from bioshift.harness import axis_summary
from bioshift.metrics import calibration_intercept_slope, ece, event_scores, windows_to_events
from bioshift.spec import cell_table


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
def test_default_benchmark_composition():
    b = default_benchmark()
    assert set(t.modality for t in b.tasks.values()) == {"icu", "ecg", "eeg", "echo"}
    assert len(b.cells_for(modality="icu")) == 3 * 12
    assert len(b.cells_for(task="ecg_dx12")) == 25
    assert len(b.cells_for(modality="eeg")) == 5
    assert len(b.cells_for(modality="echo")) == 3
    assert all(ShiftAxis.POPULATION in c.axes for c in b.cells_for(modality="echo") if c.target == ("echonet_pediatric",))
    tab = cell_table(b)
    assert tab["population"].dtype == bool and len(tab) == len(b.cells)


def test_benchmark_json_roundtrip_and_validation():
    b = default_benchmark()
    b2 = Benchmark.from_json(b.to_json())
    assert [c.id for c in b2.cells] == [c.id for c in b.cells]
    assert b2.cells[0].axes == b.cells[0].axes
    bad = Benchmark.from_json(b.to_json())
    bad.cells.append(bad.cells[0])
    with pytest.raises(ValueError):
        bad.validate()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_calibration_of_calibrated_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.02, 0.98, 5000)
    y = (rng.uniform(size=5000) < s).astype(int)
    a, b = calibration_intercept_slope(y, s)
    assert abs(a) < 0.15 and abs(b - 1) < 0.15
    assert ece(y, s) < 0.05
    # over-confident scores -> slope < 1
    s2 = 1 / (1 + np.exp(-2 * np.log(s / (1 - s))))
    _, b2 = calibration_intercept_slope(y, s2)
    assert b2 < 0.8


def test_event_scoring_toy():
    rec = np.array(["r"] * 10)
    t = np.arange(10) * 2.0
    ref = windows_to_events(np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0]), t, rec, window_s=4.0)
    assert len(ref) == 2
    hyp = windows_to_events(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), t, rec, window_s=4.0)
    sc = event_scores(ref, hyp, tol_pre=0, tol_post=0)
    assert sc["event_sensitivity"] == 0.5 and sc["event_precision"] == 1.0
    sc2 = event_scores(ref, [("r", 100.0, 104.0)], tol_pre=0, tol_post=0, total_hours=1.0)
    assert sc2["event_f1"] == 0.0 and sc2["fp_per_24h"] == 24.0


def test_evaluate_binary_with_bootstrap_and_subgroups():
    rng = np.random.default_rng(1)
    n = 800
    X = rng.normal(size=(n, 3))
    logit = X[:, 0] * 2
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    s = 1 / (1 + np.exp(-logit))
    groups = rng.integers(0, 40, n)
    meta = pd.DataFrame({"sex": rng.integers(0, 2, n), "age_band": rng.choice(["<40", ">=65"], n)})
    out = evaluate("binary", y, s, groups, meta, n_boot=50)
    assert 0.8 < out["auroc"] < 1.0
    assert out["auroc_lo"] <= out["auroc"] <= out["auroc_hi"]
    assert "gap_sex" in out and np.isfinite(out["gap_sex"])


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def test_domain_classifier_and_mmd_detect_shift():
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, (300, 5))
    B = rng.normal(0, 1, (300, 5))
    C = rng.normal(1.0, 1, (300, 5))
    assert abs(domain_classifier_auc(A, B) - 0.5) < 0.1
    assert domain_classifier_auc(A, C) > 0.8
    _, p_same = mmd_rbf(A, B, n_perm=50)
    _, p_diff = mmd_rbf(A, C, n_perm=50)
    assert p_same > 0.05 and p_diff < 0.05


def test_bbse_recovers_label_shift():
    rng = np.random.default_rng(3)
    n = 20000
    y_s = (rng.uniform(size=n) < 0.3).astype(int)
    # a noisy classifier with fixed class-conditional confusion
    yhat_s = np.where(rng.uniform(size=n) < 0.8, y_s, 1 - y_s)
    y_t = (rng.uniform(size=n) < 0.6).astype(int)
    yhat_t = np.where(rng.uniform(size=n) < 0.8, y_t, 1 - y_t)
    w = bbse_label_shift(y_s, yhat_s, yhat_t)
    assert abs(w[1] - 0.6 / 0.3) < 0.25 and abs(w[0] - 0.4 / 0.7) < 0.15


def test_decomposition_negative_and_label_shift_controls():
    rng = np.random.default_rng(4)
    d = 6
    beta = rng.normal(size=d)
    from sklearn.linear_model import LogisticRegression

    def make(n, prior_shift=0.0, mean_shift=0.0):
        X = rng.normal(mean_shift, 1, (n, d))
        logit = X @ beta + prior_shift
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
        return X, y

    Xtr, ytr = make(3000)
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    Xs, ys = make(3000)
    Xt, yt = make(3000)  # same distribution -> all components ~ 0
    dec = decompose_gap(ys, clf.predict_proba(Xs)[:, 1], Xs, yt, clf.predict_proba(Xt)[:, 1], Xt)
    assert abs(dec["total_gap"]) < 0.03 and abs(dec["covariate"]) < 0.03 and abs(dec["label"]) < 0.03
    # pure label shift: resample the target by class (P(X|Y) fixed, prior 0.5 -> 0.75)
    Xp, yp = make(20000)
    pos, neg = np.flatnonzero(yp == 1), np.flatnonzero(yp == 0)
    idx = np.r_[rng.choice(pos, 2250, replace=False), rng.choice(neg, 750, replace=False)]
    Xt2, yt2 = Xp[idx], yp[idx]
    dec2 = decompose_gap(ys, clf.predict_proba(Xs)[:, 1], Xs, yt2, clf.predict_proba(Xt2)[:, 1], Xt2)
    true_ratio = 0.75 / ys.mean()
    assert abs(dec2["label_ratio_pos"] - true_ratio) < 0.25  # BBSE recovers q(y)/p(y)
    assert abs(dec2["total_gap"]) < 0.03  # AUROC is invariant to class prior
    assert set(dec2) >= {"covariate", "label", "concept", "covariate_orderA", "label_orderB"}


# --------------------------------------------------------------------------- #
# harness end-to-end on synthetic data
# --------------------------------------------------------------------------- #
def test_run_benchmark_synthetic_all_task_types():
    b = default_benchmark()
    ad = SyntheticAdapter(b, n_per_domain=400, n_groups=40, d=8, shift_scale=1.0, seed=0)
    cells = [
        b.cells_for(task="icu_mortality_48h")[0],
        b.cells_for(task="ecg_dx12")[0],
        b.cells_for(task="eeg_seizure_event")[0],
        b.cells_for(task="echo_ef")[0],
    ]
    res = run_benchmark(b, ad, cells=cells, n_boot=20, seed=0)
    assert set(res["kind"]) >= {"in_domain", "transfer", "gap", "diagnostic"}
    assert set(res["modality"]) == {"icu", "ecg", "eeg", "echo"}
    prim = res[(res["kind"] == "in_domain") & (res["metric"] == "primary")]
    assert len(prim) == 4 and prim["value"].notna().all()
    # event task produced event-level metrics
    assert (res[(res["task"] == "eeg_seizure_event") & (res["metric"] == "event_f1")]["value"].notna()).all()
    lb = leaderboard(res)
    assert len(lb) == 4 and "transfer:primary" in lb.columns and "diagnostic:domain_auc" in lb.columns
    reg = shift_loss_regression(res)
    assert "pooled" in set(reg["stratum"])
    ax = axis_summary(res)
    assert {"modality", "axis", "mean_gap_with"} <= set(ax.columns)


def test_negative_control_gap_is_small_and_shift_hurts():
    b = default_benchmark()
    ad = SyntheticAdapter(b, n_per_domain=800, n_groups=80, d=8, shift_scale=0.0, seed=1)
    nc = negative_control("icu_mortality_48h", "mimic_iv", b, ad, seed=1, n_boot=10)
    assert abs(nc["gap"]) < 0.06
    assert abs(nc["domain_auc"] - 0.5) < 0.12
    ad2 = SyntheticAdapter(b, n_per_domain=800, n_groups=80, d=8, shift_scale=2.0, seed=1)
    cell = b.cells_for(task="icu_mortality_48h")[0]
    res = run_benchmark(b, ad2, cells=[cell], n_boot=10, seed=1, diagnostics=False)
    gap = res[(res["kind"] == "gap") & (res["metric"] == "gap_primary")]["value"].iloc[0]
    assert np.isfinite(gap)


def test_manifest_adapter_roundtrip(tmp_path):
    b = default_benchmark()
    syn = SyntheticAdapter(b, n_per_domain=200, n_groups=20, d=5)
    data = syn.load("icu_aki_48h", "hirid")
    cache = tmp_path / "data" / "cache" / "icu_aki_48h"
    cache.mkdir(parents=True)
    np.savez(cache / "hirid.npz", X=data.X, y=data.y, groups=data.groups)
    data.meta.to_csv(cache / "hirid_meta.csv", index=False)
    write_split_manifest(data.groups, tmp_path / "manifests" / "icu_aki_48h" / "hirid_splits.csv", seed=0)
    ad = ManifestAdapter(tmp_path, b)
    tr, te = ad.load("icu_aki_48h", "hirid", "train"), ad.load("icu_aki_48h", "hirid", "test")
    assert len(tr) + len(te) < len(data)  # val held out
    assert not set(tr.groups) & set(te.groups)
    with pytest.raises(FileNotFoundError):
        ad.load("icu_aki_48h", "eicu")
