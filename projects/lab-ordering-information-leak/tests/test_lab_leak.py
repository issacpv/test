"""Synthetic-data tests for lab_leak (no dataset downloads needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab_leak import attribution, ordering, simulate, sql  # noqa: E402

CFG = ordering.FeatureConfig(window_hours=24.0, morning_window=(4, 8))


def _cohort(gamma: float, seed: int = 0, n: int = 600, site_shift: float = 0.0):
    long, stays = simulate.simulate_cohort(simulate.SimConfig(n_stays=n, gamma=gamma, seed=seed, site_shift=site_shift))
    V, M = ordering.build_features(long, stays, CFG)
    return long, stays, V, M


def test_simulation_schema_and_schedule_annotation():
    long, stays = simulate.simulate_cohort(simulate.SimConfig(n_stays=50, seed=1))
    assert set(["stay_id", "t_hours", "hod", "itemid", "valuenum", "priority"]) <= set(long.columns)
    ann = ordering.annotate_schedule(long, stays, CFG)
    assert ann["routine"].sum() > 0 and ann["off_schedule"].sum() > 0
    assert not (ann["routine"] & ann["is_stat"]).any()
    # morning draws are more frequent per hour than off-hours draws
    per_hour = ann.groupby("hod").size()
    assert per_hour.loc[4:7].mean() > per_hour.drop(index=range(4, 8)).mean()


def test_features_align_and_totals_consistent():
    long, stays, V, M = _cohort(gamma=1.0, n=200)
    assert len(V) == len(M) == len(stays)
    assert (V.index == stays["stay_id"]).all()
    items = sorted(long["itemid"].unique())
    n_cols = [f"{it}_n" for it in items]
    np.testing.assert_allclose(M[n_cols].sum(axis=1), M["n_total"])
    assert ((M["n_off_total"] + (M[[f"{it}_n" for it in items]].sum(axis=1) - M["n_off_total"])) == M["n_total"]).all()
    # a never-ordered item has NaN values and 'window' hours since last
    never = M[f"{items[0]}_any"] == 0
    if never.any():
        assert V.loc[never, f"{items[0]}_last"].isna().all()
        assert (M.loc[never, f"{items[0]}_hrs_since"] == CFG.window_hours).all()
    # pending results: collected in the window, resulted after it -> masks only, never values
    long2 = long.copy()
    long2["result_delay_hours"] = 0.0
    late = long2["t_hours"] > 22.0
    long2.loc[late, "result_delay_hours"] = 5.0
    V2, M2 = ordering.build_features(long2, stays, CFG)
    assert "n_pending_total" in M2 and M2["n_pending_total"].sum() == int(late.sum())
    assert (M2["n_total"] + M2["n_pending_total"]).sum() == M["n_total"].sum()
    assert V2.count().sum() <= V.count().sum()


def test_mask_only_signal_grows_with_ordering_informativeness():
    res = {}
    for gamma in (0.0, 2.0):
        long, stays, V, M = _cohort(gamma=gamma, n=800, seed=3)
        y = stays["y"].to_numpy()
        n_tr = 500
        att = attribution.two_player_shapley(V.iloc[:n_tr], M.iloc[:n_tr], y[:n_tr], V.iloc[n_tr:], M.iloc[n_tr:], y[n_tr:])
        res[gamma] = att
    assert res[2.0].auroc_masks_only > res[0.0].auroc_masks_only + 0.1
    assert res[2.0].mask_share > res[0.0].mask_share
    for att in res.values():
        np.testing.assert_allclose(att.shapley_values + att.shapley_masks, att.gain_both, atol=1e-9)


def test_thinning_hurts_mask_models_more_and_mask_dropout_is_more_robust():
    long, stays, V, M = _cohort(gamma=2.0, n=900, seed=5)
    y = stays["y"].to_numpy()
    tr = stays["stay_id"] < 600
    Vtr, Mtr, ytr = V[tr.to_numpy()], M[tr.to_numpy()], y[tr.to_numpy()]
    long_te, stays_te = long[long["stay_id"] >= 600], stays[~tr].reset_index(drop=True)
    long_tr, stays_tr = long[long["stay_id"] < 600], stays[tr].reset_index(drop=True)
    feature_fn = lambda l, s: ordering.build_features(l, s, CFG)  # noqa: E731
    plain = attribution.make_model("lr").fit(pd.concat([Vtr, Mtr], axis=1), ytr)
    value_only = attribution.make_model("lr").fit(Vtr, ytr)
    robust = attribution.fit_order_dropout(long_tr, stays_tr, ytr, feature_fn, kind="lr")
    curve = attribution.thinning_stress_curve(
        {"VM": (plain, "VM"), "V": (value_only, "V"), "VM_dropout": (robust, "VM")},
        feature_fn, long_te, stays_te, keep_fracs=(1.0, 0.5, 0.2), n_rep=2)
    for metric in ("auroc", "loglik"):
        area = attribution.degradation_area(curve, metric)
        assert set(area.index) == {"VM", "V", "VM_dropout"}
        assert area["VM_dropout"] < area["VM"], metric
    full = curve[curve.keep_frac == 1.0].groupby("model")["auroc"].mean()
    assert full["VM"] > 0.6
    # feature-level mask dropout keeps the tables aligned and consistent
    Vd, Md = attribution.mask_dropout(Vtr, Mtr, p=0.5, rng=np.random.default_rng(1))
    assert Vd.shape == Vtr.shape and (Md["n_total"] <= Mtr["n_total"]).all() and (Md["n_total"] >= 0).all()


def test_thin_and_standardize_and_weights():
    long, stays, V, M = _cohort(gamma=1.0, n=150, seed=7)
    rng = np.random.default_rng(0)
    thinned = ordering.thin_orders(long, 0.5, rng)
    assert 0.35 * len(long) < len(thinned) < 0.65 * len(long)
    off_only = ordering.thin_orders(long, 0.0, rng, stays=stays, cfg=CFG, subset="off_schedule")
    std = ordering.schedule_standardize(long, stays, CFG)
    assert len(off_only) == len(std) and ordering.annotate_schedule(std, stays, CFG)["routine"].all()
    prof = ordering.leakage_timing_profile(long, stays)
    assert prof.shape[0] == 2 and (prof.to_numpy() >= 0).all()
    w = attribution.intensity_weights(rng.poisson(20, 300), rng.poisson(8, 300))
    assert abs(w.mean() - 1) < 1e-9 and w.min() > 0
    # weights should up-weight low-intensity source stays
    n_src = rng.poisson(20, 300)
    w = attribution.intensity_weights(n_src, rng.poisson(8, 300))
    assert np.corrcoef(n_src, w)[0, 1] < 0


def test_sql_templates_render():
    s = sql.mimic_labs_sql("/data/mimiciv/3.1", sql.MIMIC_LAB_ITEMS.values())
    assert "labevents.csv.gz" in s and "50912" in s and "priority" in s
    assert "poe.csv.gz" in sql.mimic_lab_orders_sql("/x")
    e = sql.eicu_labs_sql("/data/eicu", sql.EICU_LAB_NAMES.values())
    assert "lab.csv.gz" in e and "'WBC x 1000'" in e
    assert "hospitalid" in sql.eicu_stays_sql("/x") and "anchor_year_group" in sql.mimic_stays_sql("/x")
