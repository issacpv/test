"""Synthetic tests for nursing_risk."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nursing_risk import dynamic, evaluate, items, labels


def _d_items():
    return pd.DataFrame(
        {
            "itemid": [224054, 224055, 224056, 224057, 224058, 224059, 228500, 228501, 224093, 999],
            "label": ["Braden Sensory Perception", "Braden Moisture", "Braden Activity", "Braden Mobility", "Braden Nutrition", "Braden Friction/Shear", "Pressure Ulcer Stage #1", "Impaired Skin Site #1", "Richmond-RAS Scale", "Temperature Fahrenheit"],
        }
    )


def test_item_resolution_and_braden_total():
    res = items.resolve_items(_d_items())
    assert items.verify_expected(res, items.EXPECTED_BRADEN_ITEMIDS) == []
    assert set(res.loc[res["concept"] == "skin_pi_stage", "itemid"]) == {228500}
    assert set(res.loc[res["concept"] == "rass", "itemid"]) == {224093}
    t0 = pd.Timestamp("2150-01-01 08:00")
    rows = []
    for k, iid in enumerate([224054, 224055, 224056, 224057, 224058, 224059]):
        rows.append({"stay_id": 1, "charttime": t0 + pd.Timedelta(minutes=5 * k), "itemid": iid, "valuenum": 2 if iid != 224059 else 3})
    rows.append({"stay_id": 1, "charttime": t0 + pd.Timedelta(hours=12), "itemid": 224054, "valuenum": 3})  # lone component: no total
    rows.append({"stay_id": 1, "charttime": t0 + pd.Timedelta(hours=13), "itemid": 224055, "valuenum": 9})  # out of range dropped
    ce = pd.DataFrame(rows)
    bt = items.braden_table(ce, res)
    totals = bt.dropna(subset=["braden_total"])
    assert len(totals) == 1 and totals["braden_total"].iloc[0] == 2 * 5 + 3
    stays = pd.DataFrame({"stay_id": [1], "intime": [t0], "outtime": [t0 + pd.Timedelta(days=2)]})
    fq = items.assessment_frequency(bt, stays)
    assert fq.loc[0, "n_assessments"] == 1 and abs(fq.loc[0, "assessments_per_24h"] - 0.5) < 1e-9


def test_icd_and_nursing_labels_multiverse():
    assert labels.icd_pi_stage("L89.153", 10) == 3 and labels.icd_pi_stage("L89.006", 10) == 0 and labels.icd_pi_stage("I10", 10) is None
    assert labels.icd_pi_stage("707.23", 9) == 3 and labels.icd_pi_stage("70703", 9) == 0
    assert labels.parse_stage_value("Stage 2") == 2 and labels.parse_stage_value("Unstageable") == 0 and labels.parse_stage_value("Intact") is None
    diag = pd.DataFrame({"hadm_id": [10, 10, 20, 30], "icd_code": ["L89.153", "I10", "L89.91", "70720"], "icd_version": [10, 10, 10, 9]})
    icd = labels.icd_labels(diag).set_index("hadm_id")
    assert icd.loc[10, "icd_stage2plus"] and not icd.loc[20, "icd_stage2plus"] and icd.loc[30, "icd_any"]
    t0 = pd.Timestamp("2150-01-01 00:00")
    res = items.resolve_items(_d_items())
    ce = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "hadm_id": [10, 10, 20, 30],
            "charttime": [t0 + pd.Timedelta(hours=6), t0 + pd.Timedelta(hours=60), t0 + pd.Timedelta(hours=80), t0 + pd.Timedelta(hours=100)],
            "itemid": [228500, 228500, 228500, 228500],
            "value": ["Stage 1", "Stage 2", "Stage 2", "Intact"],
        }
    )
    adm = pd.DataFrame({"hadm_id": [10, 20, 30], "admittime": [t0, t0, t0]})
    n1 = labels.nursing_first_pi(ce, res, min_stage=1)
    n2 = labels.nursing_first_pi(ce, res, min_stage=2)
    assert n1.set_index("stay_id").loc[1, "first_pi_stage"] == 1 and n2.set_index("stay_id").loc[1, "first_pi_stage"] == 2
    stays = pd.DataFrame({"stay_id": [1, 2, 3], "hadm_id": [10, 20, 30]})
    per_stay, summary = labels.label_multiverse(stays, labels.icd_labels(diag), n1, n2, adm)
    ps = per_stay.set_index("stay_id")
    assert not ps.loc[1, "nurs1_ha24"]  # first documentation at 6 h -> probable POA
    assert ps.loc[1, "nurs2_ha24"]  # stage-2 first documented at 60 h -> hospital acquired
    assert ps.loc[2, "nurs1_ha24"] and ps.loc[2, "nurs1_ha48"]
    assert set(summary["definition"]) >= {"icd_any", "icd_stage2plus", "nurs1_ha24", "nurs2_ha48", "icd_and_nurs1_ha24"}
    # three ICD-coded stays; only stay 1 has PI documentation within 24 h of admission
    assert abs(summary.set_index("definition").loc["icd_any", "probable_poa_frac_of_icd"] - 1 / 3) < 1e-9
    falls = labels.inpatient_fall_icd10(pd.DataFrame({"hadm_id": [1, 1, 2], "icd_code": ["W06.XXXA", "Y92.230", "W06.XXXA"], "icd_version": [10, 10, 10]})).set_index("hadm_id")
    assert falls.loc[1, "inpatient_fall"] and not falls.loc[2, "inpatient_fall"]


def _synthetic_icu(n_stays: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2150-01-01")
    stays, braden_rows, events, prev = [], [], [], []
    for s in range(n_stays):
        intime = t0 + pd.Timedelta(days=int(rng.integers(0, 100)))
        los_d = float(rng.gamma(3, 2.5)) + 1.0
        outtime = intime + pd.Timedelta(days=los_d)
        base = float(rng.normal(16, 3))
        drift = float(rng.normal(-0.3, 0.4))  # per day; falling Braden = rising risk
        age = float(rng.normal(65, 12))
        stays.append({"stay_id": s, "intime": intime, "outtime": outtime, "age": age})
        hazard_acc = 0.0
        e_time = pd.NaT
        n_per_day = 2 + int(base < 13)  # nurses reassess low-risk-score patients more often (informative observation)
        for h in np.arange(2, los_d * 24, 24 / n_per_day):
            total = float(np.clip(round(base + drift * h / 24 + rng.normal(0, 1)), 6, 23))
            braden_rows.append({"stay_id": s, "charttime": intime + pd.Timedelta(hours=float(h)), "braden_total": total})
            if pd.isna(e_time):
                hazard_acc += np.exp(-0.35 * (total - 12)) * 0.02
                if rng.random() < np.exp(-0.35 * (total - 12)) * 0.015:
                    e_time = intime + pd.Timedelta(hours=float(h) + 1)
        if pd.notna(e_time) and e_time < outtime:
            events.append({"stay_id": s, "first_pi_time": e_time})
        if base < 13 and rng.random() < 0.6:
            prev.append({"stay_id": s, "charttime": intime + pd.Timedelta(hours=30)})
    return pd.DataFrame(stays), pd.DataFrame(braden_rows), pd.DataFrame(events), pd.DataFrame(prev)


def test_landmark_dataset_leakage_guards_and_models():
    stays, braden, events, prev = _synthetic_icu()
    lm = dynamic.build_landmark_dataset(braden, stays, events, prevention=prev, landmark_h=24, horizon_h=72, max_landmark_days=10)
    assert len(lm) > 500 and lm["outcome"].sum() > 20
    # no landmark after the event; outcome only if event within horizon
    ev = events.set_index("stay_id")["first_pi_time"]
    for _, r in lm[lm["stay_id"].isin(ev.index)].iterrows():
        e = ev[r["stay_id"]]
        assert r["landmark_time"] < e
        assert r["outcome"] == int(r["landmark_time"] < e <= r["landmark_time"] + pd.Timedelta(hours=72))
    # features only from the past: hours_since_last must be >= 0 and n_assess_24h bounded
    assert (lm["hours_since_last"].dropna() >= 0).all() and lm["n_assess_24h"].max() <= 4
    assert lm["prevention_24h"].isin([0, 1]).all()
    # split by stay
    rng = np.random.default_rng(1)
    test_stays = set(rng.choice(stays["stay_id"].to_numpy(), 100, replace=False))
    tr, te = lm[~lm["stay_id"].isin(test_stays)], lm[lm["stay_id"].isin(test_stays)]
    dyn = dynamic.fit_landmark_supermodel(tr)
    stat = dynamic.fit_static_baseline(tr)
    p_dyn, p_stat = dynamic.predict_landmark(dyn, te), dynamic.predict_landmark(stat, te)
    m = evaluate.metrics_by_landmark(te, p_dyn)
    pooled = m[m["landmark_day"] == "pooled"].iloc[0]
    assert pooled["auroc"] > 0.6
    diff = evaluate.paired_auroc_difference(te, p_dyn, p_stat, n_boot=30)
    assert diff["difference"] > -0.05  # trajectory model is at least as good as the static baseline
    ci = evaluate.bootstrap_auroc_by_stay(te, p_dyn, n_boot=30)
    assert ci["ci_low"] <= ci["auroc"] <= ci["ci_high"]
    perm = dynamic.permute_observation_features(te, seed=2)
    p_perm = dynamic.predict_landmark(dyn, perm)
    assert np.isfinite(p_perm).all()
    ta = evaluate.treatment_aware_summary(te, p_dyn)
    assert set(ta["prevention"]) <= {0, 1}
    dc = evaluate.decision_curve(te["outcome"].to_numpy(), p_dyn)
    assert len(dc) == 5
