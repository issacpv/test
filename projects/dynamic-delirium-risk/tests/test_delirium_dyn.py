"""Synthetic-data tests for delirium_dyn (no PhysioNet data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delirium_dyn import assessments as asm  # noqa: E402
from delirium_dyn import features as ft  # noqa: E402
from delirium_dyn import landmark as lmk  # noqa: E402
from delirium_dyn import models as md  # noqa: E402


def _chart_fixture():
    d_items = pd.DataFrame(
        {
            "itemid": [1, 2, 3, 4, 5, 6],
            "label": ["Delirium assessment", "CAM-ICU MS Change", "CAM-ICU Inattention", "CAM-ICU RASS LOC", "CAM-ICU Disorganized thinking", "Richmond-RAS Scale"],
        }
    )
    t0 = pd.Timestamp("2150-01-01 08:00")
    rows = [
        # window 0: features positive -> delirium
        (1, 2, t0 + pd.Timedelta(hours=2), "Yes"), (1, 3, t0 + pd.Timedelta(hours=2), "Yes"), (1, 4, t0 + pd.Timedelta(hours=2), "Yes"),
        (1, 6, t0 + pd.Timedelta(hours=1), "-1"),
        # window 1: summary negative
        (1, 1, t0 + pd.Timedelta(hours=14), "Negative"), (1, 6, t0 + pd.Timedelta(hours=13), "0 Alert and calm"),
        # window 2: RASS -5 only -> coma
        (1, 6, t0 + pd.Timedelta(hours=26), "-5"),
        # window 3: RASS -2 only -> unscreened
        (1, 6, t0 + pd.Timedelta(hours=38), "-2"),
    ]
    ce = pd.DataFrame(rows, columns=["stay_id", "itemid", "charttime", "value"])
    stays = pd.DataFrame({"stay_id": [1], "intime": [t0], "outtime": [t0 + pd.Timedelta(hours=48)]})
    return d_items, ce, stays


def test_parsing_and_window_state_rule():
    d_items, ce, stays = _chart_fixture()
    tl = asm.assessment_timeline(ce, d_items)
    assert set(tl["kind"]) == {"cam", "rass"}
    assert asm.parse_rass("-3 Moderate sedation") == -3 and asm.parse_cam_value("UTA") == "uta"
    w = asm.window_states(tl, stays, window_hours=12)
    assert w["state"].tolist() == ["delirium", "normal", "coma", "unscreened", "discharged"]
    death = pd.Series({1: stays["outtime"].iloc[0]})
    w2 = asm.window_states(tl, stays, window_hours=12, deathtime=death)
    assert w2["state"].iloc[-1] == "dead"


def test_landmark_schemes_are_paired_and_prevalence_ordering():
    sim = asm.simulate_stays(n_stays=60, rng=np.random.default_rng(0))
    w = sim["windows"]
    summ = lmk.label_scheme_summary(w, horizon=1).set_index("scheme")
    assert summ.loc["multistate", "n_rows"] == summ.loc["binary_coma_negative", "n_rows"]
    assert summ.loc["binary_drop_coma", "n_rows"] <= summ.loc["multistate", "n_rows"]
    # coding coma/unscreened as negative dilutes prevalence relative to dropping them
    assert summ.loc["binary_coma_negative", "delirium_prevalence"] <= summ.loc["binary_drop_coma", "delirium_prevalence"]
    lm = lmk.build_landmark_dataset(w, horizon=1, scheme="multistate")
    assert not (lm["cur_state"].isin(["discharged", "dead"])).any()
    assert lm["landmark_idx"].min() >= 2
    tc = lmk.transition_counts(w)
    assert tc.values.sum() > 0
    dcfd = lmk.delirium_coma_free_days(w)
    assert dcfd["dcfd"].between(0, 14).all()


def test_features_blocks_and_no_future_information():
    sim = asm.simulate_stays(n_stays=20, rng=np.random.default_rng(1))
    w, sed = sim["windows"], sim["sedation"]
    f = ft.assemble_features(w, sedation=sed)
    blocks = ft.feature_blocks(f.columns)
    assert blocks["sedation"] and blocks["assessment"]
    # first window of each stay has no previous-window information
    first = f[f["window_idx"] == 0]
    assert first["sed_rass_mean_prev"].isna().all() and first["asm_frac_screened"].isna().all()
    # inputevents-based exposure apportioning
    d_items = pd.DataFrame({"itemid": [10, 11], "label": ["Propofol", "Midazolam (Versed)"]})
    t0 = pd.Timestamp("2150-01-01 00:00")
    ie = pd.DataFrame({"stay_id": [1, 1], "itemid": [10, 11], "starttime": [t0 - pd.Timedelta(hours=6), t0 - pd.Timedelta(hours=30)],
                       "endtime": [t0 + pd.Timedelta(hours=6), t0 - pd.Timedelta(hours=20)], "amount": [1200.0, 5.0]})
    win = pd.DataFrame({"stay_id": [1], "window_idx": [0], "w_start": [t0]})
    ex = ft.sedative_exposure_from_inputevents(ie, d_items, win, lookback_hours=12)
    assert ex["sed_propofol_12h"].iloc[0] == pytest.approx(600.0)  # half of the infusion overlaps the lookback
    assert ex["sed_benzo_12h"].iloc[0] == 0.0


def test_transition_intensities_recover_planted_benzo_effect():
    sim = asm.simulate_stays(n_stays=400, benzo_effect=1.2, rng=np.random.default_rng(2))
    w = sim["windows"].copy()
    w["state"] = w["true_state"]  # use hidden states to test the estimator itself
    ex = sim["sedation"].copy()
    ex["benzo"] = (ex["benzo_mg"] > 0).astype(float)
    res = md.transition_intensities(w, ex[["stay_id", "window_idx", "benzo"]], ["benzo"], "normal", "delirium")
    assert res.loc["benzo", "rate_ratio"] > 1.5 and res.loc["benzo", "ci_lo"] > 1.0
    null = asm.simulate_stays(n_stays=400, benzo_effect=0.0, rng=np.random.default_rng(3))
    wn = null["windows"].copy()
    wn["state"] = wn["true_state"]
    exn = null["sedation"].copy()
    exn["benzo"] = (exn["benzo_mg"] > 0).astype(float)
    res0 = md.transition_intensities(wn, exn[["stay_id", "window_idx", "benzo"]], ["benzo"], "normal", "delirium")
    assert res0.loc["benzo", "ci_lo"] < 1.0 < res0.loc["benzo", "ci_hi"]


def test_ablation_ipaw_and_landmark_auroc_run():
    sim = asm.simulate_stays(n_stays=150, rng=np.random.default_rng(4))
    w, sed = sim["windows"], sim["sedation"]
    f = ft.assemble_features(w, sedation=sed)
    lm = lmk.build_landmark_dataset(w, horizon=1, scheme="binary_coma_negative").merge(f, left_on=["stay_id", "landmark_idx"], right_on=["stay_id", "window_idx"], how="left")
    blocks = ft.feature_blocks([c for c in f.columns if c not in ("stay_id", "window_idx")])
    feats = blocks["sedation"] + blocks["assessment"]
    res = md.sedation_leakage_ablation(lm, feats, blocks["sedation"], blocks["assessment"], admission_cols=["asm_n_prev_delirium"], n_splits=3)
    assert set(res.index) == {"admission_only", "physiology_only", "physiology_plus_assessment", "full"}
    assert np.isfinite(res["auroc"]).all()
    wts = md.ipaw_weights(w, f, blocks["assessment"] + blocks["sedation"])
    assert (wts.loc[wts["assessed"] == 1, "ipaw"] > 0).all() and (wts.loc[wts["assessed"] == 0, "ipaw"] == 0).all()
    p = md.cross_validated_predictions(lm[feats], lm["label"], lm["stay_id"], n_splits=3)
    per = md.auroc_by_landmark(lm, p, min_n=10)
    assert len(per) > 0
