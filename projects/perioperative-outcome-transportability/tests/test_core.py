"""Synthetic tests for periop_transport."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from periop_transport import cohorts, features, metrics, shift


def test_mimic_definitions_and_multiverse():
    transfers = pd.DataFrame({"hadm_id": [1, 1, 2, 3], "careunit": ["Emergency Department", "PACU", "Medicine", "PACU"], "intime": pd.to_datetime(["2100-01-01 08:00", "2100-01-01 12:00", "2100-02-01 00:00", "2100-03-01 00:00"])})
    procs = pd.DataFrame({"hadm_id": [1, 2, 4], "icd_code": ["0DTJ0ZZ", "8872", "0DJ08ZZ"], "icd_version": [10, 9, 10], "chartdate": pd.to_datetime(["2100-01-01", "2100-02-01", "2100-04-01"])})
    services = pd.DataFrame({"hadm_id": [1, 2, 3], "curr_service": ["SURG", "MED", "ORTHO"]})
    d = cohorts.mimic_surgical_definitions(transfers, procs, services).set_index("hadm_id")
    assert d.loc[1, "pacu"] and d.loc[1, "proc"] and d.loc[1, "svc"] and d.loc[1, "pacu_and_proc"]
    assert not d.loc[2, "proc"]  # 88.72 is diagnostic ultrasound (chapter 88)
    assert not d.loc[4, "proc"]  # inspection root operation excluded
    assert d.loc[3, "pacu"] and not d.loc[3, "proc"] and d.loc[3, "pacu_or_proc"]
    assert d.loc[1, "surgery_time"] == pd.Timestamp("2100-01-01 12:00")
    mv = cohorts.definition_multiverse(d.reset_index(), outcome=pd.Series({1: 1, 2: 0, 3: 0, 4: 0}))
    assert set(mv["definition"]) == set(cohorts.DEFINITIONS)
    assert mv.set_index("definition").loc["pacu", "n"] == 2
    assert mv.set_index("definition").loc["pacu", "jaccard_pacu"] == 1.0


def _enc():
    t0 = pd.Timestamp("2100-01-01 10:00")
    return pd.DataFrame(
        {
            "op_id": [1, 2, 3],
            "subject_id": [10, 20, 30],
            "hadm_id": [100, 200, 300],
            "op_start": [t0, t0, t0],
            "op_end": [t0 + pd.Timedelta(hours=3)] * 3,
            "death_time": [t0 + pd.Timedelta(days=5), pd.NaT, t0 + pd.Timedelta(days=45)],
            "service": ["SURG", "CSURG", "SURG"],
        }
    )


def test_labels_mortality_icu_aki():
    enc = _enc()
    mort = cohorts.label_mortality(enc)
    assert list(mort) == [1, 0, 0]
    t0 = pd.Timestamp("2100-01-01 10:00")
    icu = pd.DataFrame({"hadm_id": [100, 200, 300], "intime": [t0 + pd.Timedelta(hours=4), t0 + pd.Timedelta(hours=4), t0 - pd.Timedelta(days=1)], "outtime": [t0 + pd.Timedelta(days=2)] * 3})
    icu_lab = cohorts.label_unplanned_icu(enc, icu)
    assert list(icu_lab) == [1, 0, 0]  # 2 is planned (CSURG); 3 was already in ICU
    cr = pd.DataFrame(
        {
            "subject_id": [10, 10, 10, 20, 20, 30],
            "charttime": [t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=2), t0 - pd.Timedelta(days=2), t0 + pd.Timedelta(days=1), t0 - pd.Timedelta(days=1)],
            "value": [1.0, 1.2, 1.6, 0.8, 0.9, 1.0],
        }
    )
    aki = cohorts.label_aki_kdigo(enc, cr).set_index("op_id")
    assert aki.loc[1, "stage"] == 1 and aki.loc[1, "aki"] == 1
    assert aki.loc[2, "stage"] == 0 and aki.loc[2, "aki"] == 0
    assert np.isnan(aki.loc[3, "stage"]) and aki.loc[3, "n_post"] == 0
    assert cohorts.kdigo_stage(3.5, 1.0, None) == 3 and cohorts.kdigo_stage(1.25, 1.0, 0.9) == 1


def test_features_ontology_labs_charlson():
    ok = features.verify_ontology({"creatinine": "50912", "hemoglobin": "51222"}, {"50912": "Creatinine", "51222": "Hemoglobin"})
    assert ok == []
    bad = features.verify_ontology({"creatinine": "50912"}, {"50912": "Glucose"})
    assert len(bad) == 1
    enc = _enc()
    t0 = pd.Timestamp("2100-01-01 10:00")
    labs = pd.DataFrame({"subject_id": [10, 10, 20], "charttime": [t0 - pd.Timedelta(days=2), t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(hours=1)], "concept": ["creatinine", "creatinine", "creatinine"], "value": [0.9, 1.1, 2.0]})
    wide = features.last_value_before(labs, enc)
    assert wide.loc[1, "creatinine"] == 1.1
    assert wide.loc[2, "creatinine_missing"] == 1  # postoperative value must not leak
    diag = pd.DataFrame({"hadm_id": [100, 100, 200, 300], "icd_code": ["I50.9", "C78.0", "250.00", "N18.4"], "icd_version": [10, 10, 9, 10]})
    ch = features.charlson_from_icd(diag).set_index("hadm_id")
    assert ch.loc[100, "charlson_index"] == 1 + 6  # CHF + metastatic (cancer not double counted)
    assert ch.loc[200, "charlson_index"] == 1 and ch.loc[300, "charlson_index"] == 2
    assert features.surgery_group_from_code("0DTJ0ZZ", "icd10pcs") == "gastrointestinal"
    assert features.surgery_group_from_code("4562", "icd9") == "gastrointestinal"
    assert features.surgery_group_from_code("27447", "cpt") == "musculoskeletal"
    enc2 = enc.assign(age=[60, 70, 50], sex=["F", "M", "F"], emergency=[0, 1, 0], anesthesia=["general", "regional", "general"], surgery_group=["gastrointestinal", "musculoskeletal", "other"])
    X = features.assemble_preop_features(enc2, wide, ch.reset_index())
    assert "surgery_group_gastrointestinal" in X.columns and X.loc[1, "female"] == 1 and X.loc[1, "charlson_index"] == 7


def _synth_site(n: int, rng: np.random.Generator, shift_x: float = 0.0, beta: np.ndarray | None = None, intercept: float = -3.0):
    beta = beta if beta is not None else np.array([1.0, 0.8, -0.5])
    X = rng.normal(shift_x, 1.0, (n, 3))
    p = 1 / (1 + np.exp(-(intercept + X @ beta)))
    y = rng.binomial(1, p)
    return X, y


def test_bbse_recovers_label_shift_and_decomposition_negative_control():
    rng = np.random.default_rng(0)
    from sklearn.linear_model import LogisticRegression

    Xs, ys = _synth_site(6000, rng)
    model = LogisticRegression().fit(Xs[:3000], ys[:3000])
    Xh, yh = Xs[3000:], ys[3000:]
    # target: same P(X), same P(Y|X) -> decomposition components should be ~0
    Xt, yt = _synth_site(4000, rng)
    ph, pt = model.predict_proba(Xh)[:, 1], model.predict_proba(Xt)[:, 1]
    w_cov = shift.domain_classifier_weights(Xh, Xt)
    w_lab, q1 = shift.bbse_label_shift(yh, (ph > 0.5).astype(int), (pt > 0.5).astype(int))
    assert abs(q1 - yt.mean()) < 0.05
    dec = shift.decompose_gap(yh, ph, yt, pt, w_cov, w_lab, lambda y, p, w: metrics.auroc(y, p, w))
    assert abs(dec["gap"]) < 0.03 and abs(dec["covariate"]) < 0.03 and abs(dec["label"]) < 0.03
    # induced label shift: subsample positives at the target to 1/3
    keep = np.where((yt == 0) | (rng.random(len(yt)) < 1 / 3))[0]
    w_lab2, q1_2 = shift.bbse_label_shift(yh, (ph > 0.5).astype(int), (pt[keep] > 0.5).astype(int))
    assert abs(q1_2 - yt[keep].mean()) < 0.05
    assert w_lab2[1] < 1.0 < w_lab2[0]


def test_recalibration_and_metrics():
    rng = np.random.default_rng(1)
    X, y = _synth_site(5000, rng)
    z = -3.0 + X @ np.array([1.0, 0.8, -0.5])
    p_true = 1 / (1 + np.exp(-z))
    p_bad = 1 / (1 + np.exp(-(0.6 * z + 1.0)))  # under-scaled and biased
    a, b = metrics.calibration_intercept_slope(y, p_bad)
    assert b > 1.2 and a < -0.3
    f = shift.recalibrate(p_bad[:2500], y[:2500], "platt")
    a2, b2 = metrics.calibration_intercept_slope(y[2500:], f(p_bad[2500:]))
    assert abs(b2 - 1.0) < 0.15 and abs(a2) < 0.2
    g = shift.recalibrate(p_bad[:2500], y[:2500], "intercept")
    a3, _ = metrics.calibration_intercept_slope(y[2500:], g(p_bad[2500:]))
    assert abs(a3) < abs(a)
    s = metrics.calibration_summary(y, p_true)
    assert 0.7 < s["auroc"] < 0.95 and abs(s["cal_slope"] - 1.0) < 0.15 and s["ece"] < 0.05
    nb = metrics.net_benefit(y, p_true)
    assert (nb["net_benefit"] >= nb["net_benefit_all"] - 1e-9).all()
    grp = rng.integers(0, 2, len(y))
    st = metrics.subgroup_table(y, p_true, grp)
    assert len(st) == 2 and "gap_auroc" in st.attrs
    ci = metrics.bootstrap_ci(y, p_true, metrics.auroc, n_boot=50)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]
    curve = shift.few_shot_curve(p_bad, y, sizes=(100, 500), n_rep=3)
    assert set(curve["method"]) == {"intercept", "platt", "temperature"} and set(curve["n"]) == {100, 500}
