"""Synthetic MIMIC-IV-ED-shaped tables exercising ed_triage (no real data required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ed_triage import cohort, mistriage, text_features, validation  # noqa: E402

ERAS = ["2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]


def make_tables(n_subjects: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    subjects = np.arange(10000, 10000 + n_subjects)
    patients = pd.DataFrame({
        "subject_id": subjects,
        "gender": rng.choice(["M", "F"], n_subjects),
        "anchor_age": rng.integers(15, 90, n_subjects),
        "anchor_year": 2150,
        "anchor_year_group": rng.choice(ERAS, n_subjects),
        "dod": pd.NaT,
    })
    stays, triage, adms, icus, pyx = [], [], [], [], []
    stay_id, hadm_id = 30000000, 20000000
    complaints = ["Chest pain", "SOB", "abd pain, n/v", "Fall", "Headache", "Fever", "SI", "back pain"]
    for s in subjects:
        n_visits = rng.integers(1, 4)
        t0 = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        for v in range(n_visits):
            intime = t0 + pd.Timedelta(hours=float(v * rng.integers(10, 2000)))
            outtime = intime + pd.Timedelta(hours=float(rng.uniform(1, 12)))
            acuity = int(rng.choice([1, 2, 3, 4, 5], p=[0.03, 0.27, 0.5, 0.15, 0.05]))
            admitted = rng.random() < (0.7 if acuity <= 2 else 0.25)
            h = None
            if admitted:
                h = hadm_id
                hadm_id += 1
                death = outtime + pd.Timedelta(hours=5) if rng.random() < 0.03 else pd.NaT
                adms.append({"subject_id": s, "hadm_id": h, "admittime": outtime, "dischtime": outtime + pd.Timedelta(days=3),
                             "deathtime": death, "language": rng.choice(["ENGLISH", "SPANISH", "?"], p=[0.8, 0.15, 0.05]),
                             "insurance": rng.choice(["Medicare", "Medicaid", "Other"])})
                if rng.random() < (0.4 if acuity <= 2 else 0.1):
                    icus.append({"subject_id": s, "hadm_id": h, "stay_id": stay_id + 500000,
                                 "intime": outtime + pd.Timedelta(hours=float(rng.uniform(0.5, 20))), "first_careunit": "MICU"})
            stays.append({"subject_id": s, "hadm_id": h, "stay_id": stay_id, "intime": intime, "outtime": outtime,
                          "gender": patients.loc[patients.subject_id == s, "gender"].iloc[0],
                          "race": rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - PUERTO RICAN", "ASIAN - CHINESE", "UNKNOWN", "OTHER"]),
                          "arrival_transport": rng.choice(["AMBULANCE", "WALK IN"]),
                          "disposition": "ADMITTED" if admitted else "HOME"})
            triage.append({"subject_id": s, "stay_id": stay_id, "temperature": rng.normal(98.6, 1), "heartrate": rng.normal(85, 15),
                           "resprate": rng.normal(18, 3), "o2sat": rng.normal(97, 2), "sbp": rng.normal(130, 20), "dbp": rng.normal(80, 10),
                           "pain": str(int(rng.integers(0, 11))), "acuity": acuity, "chiefcomplaint": rng.choice(complaints)})
            for _ in range(int(rng.integers(0, 4))):
                pyx.append({"stay_id": stay_id, "name": "acetaminophen"})
            stay_id += 1
    return (pd.DataFrame(stays), pd.DataFrame(triage), patients, pd.DataFrame(adms), pd.DataFrame(icus), pd.DataFrame(pyx))


@pytest.fixture(scope="module")
def cohort_df():
    edstays, triage, patients, admissions, icustays, pyxis = make_tables()
    return cohort.build_cohort_pandas(edstays, triage, patients, admissions, icustays, pyxis)


def test_cohort_outcomes(cohort_df):
    df = cohort_df
    assert {"hospitalization", "critical_outcome", "revisit_72h", "age", "era", "race6", "nonenglish", "n_meds_pyxis"} <= set(df.columns)
    assert (df["age"] >= 18).all()
    assert df["hospitalization"].isin([0, 1]).all() and df["critical_outcome"].sum() > 0
    # critical outcome only possible for admitted visits in this synthetic generator
    assert (df.loc[df["critical_outcome"] == 1, "hospitalization"] == 1).all()
    assert set(df["race6"].unique()) <= {"White", "Black", "Hispanic", "Asian", "Other", "Unknown"}
    X, cols = cohort.structured_matrix(df)
    assert X.shape == (len(df), len(cols)) and "pain_num" in cols


def test_revisit_definition():
    edstays, triage, patients, admissions, icustays, pyxis = make_tables(n_subjects=5, seed=3)
    # force a revisit 10 h after discharge for the first subject
    s = edstays.iloc[0]
    extra = s.copy()
    extra["stay_id"] = 99999999
    extra["intime"] = s["outtime"] + pd.Timedelta(hours=10)
    extra["outtime"] = extra["intime"] + pd.Timedelta(hours=2)
    edstays = pd.concat([edstays, extra.to_frame().T], ignore_index=True)
    edstays["stay_id"] = edstays["stay_id"].astype(int)
    tri = triage.iloc[[0]].copy()
    tri["stay_id"] = 99999999
    triage = pd.concat([triage, tri], ignore_index=True)
    df = cohort.build_cohort_pandas(edstays, triage, patients, admissions, icustays, pyxis)
    assert df.loc[df["stay_id"] == s["stay_id"], "revisit_72h"].iloc[0] == 1


def test_text_features():
    assert text_features.normalize_complaint("SOB, CP") == "shortness of breath chest pain"
    assert text_features.split_complaints("abd pain / n/v; fall") == ["abdominal pain", "nausea vomiting", "fall"]
    assert text_features.complaint_category("Chest pain") == "cardiac"
    docs = ["chest pain"] * 10 + ["shortness of breath"] * 10 + ["fall"] * 10
    vec = text_features.TfidfComplaintFeatures(min_df=1).fit(docs)
    X = vec.transform(["chest pain", "fall"])
    assert X.shape == (2, vec.vocabulary_size()) and X.nnz > 0


def test_mistriage_and_odds_ratios(cohort_df):
    df = mistriage.flag_mistriage(cohort_df)
    assert ((df["under_triage"] == 1) <= (df["acuity"] >= 3)).all()
    assert ((df["over_triage"] == 1) <= (df["acuity"] <= 2)).all()
    rates = mistriage.subgroup_rates(df, "under_triage", "race6", denominator="eligible_under")
    assert {"rate", "ci_low", "ci_high", "risk_ratio_vs_ref"} <= set(rates.columns)
    assert ((rates["ci_low"] <= rates["rate"]) & (rates["rate"] <= rates["ci_high"])).all()
    p, lo, hi = mistriage.wilson_ci(5, 50)
    assert lo < p < hi
    # IRLS logistic regression recovers a known coefficient
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(int)
    fit = mistriage.fit_logit(x[:, None], y, ["x"])
    assert fit.converged and abs(fit.coef[1] - 1.2) < 0.15
    tab = fit.table()
    assert (tab["OR_low"] <= tab["OR"]).all() and (tab["OR"] <= tab["OR_high"]).all()
    aor = mistriage.adjusted_odds_ratios(df, "under_triage", "race6", "White", ["age", "heartrate"], {"gender": "M"})
    assert set(aor["term"].str.startswith("race6=")) == {True}
    assert mistriage.e_value(2.0) > 1.0


def test_model_vs_nurse(cohort_df):
    df = mistriage.flag_mistriage(cohort_df)
    rng = np.random.default_rng(1)
    scores = df["critical_outcome"].to_numpy() * 0.5 + rng.random(len(df)) * 0.5
    cmp = mistriage.compare_model_vs_nurse(df, scores, "race6")
    assert {"nurse_under_triage", "model_under_triage"} <= set(cmp.columns)
    assert "threshold" in cmp.attrs and "tpr_gap" in cmp.attrs["model_eo"]
    y = df["critical_outcome"].to_numpy()
    obs, p = mistriage.permutation_null_gap(y, (scores > 0.5).astype(int), df["race6"].to_numpy(), df["acuity"].to_numpy(), n_perm=20)
    assert 0 <= obs <= 1 and 0 < p <= 1


def test_temporal_validation(cohort_df):
    df = cohort_df
    splits = list(validation.rolling_origin_splits(df))
    assert len(splits) >= 1 and all(len(np.intersect1d(s.train_idx, s.test_idx)) == 0 for s in splits)
    rep = validation.drift_report(df, ["heartrate", "age"], ref_eras=[ERAS[0]])
    assert "psi_heartrate" in rep.columns and rep["psi_heartrate"].ge(0).all()

    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xtr, _ = cohort.structured_matrix(tr)
        Xte, _ = cohort.structured_matrix(te)
        med = np.nanmedian(Xtr, axis=0)
        Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, tr["hospitalization"])
        return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]

    res = validation.run_temporal_benchmark(df, fit_predict, "hospitalization")
    assert {"test_auroc", "test_cal_slope", "test_nb@0.10"} <= set(res.columns)
    assert res["test_auroc"].between(0.5, 1.0).all()
    y = df["hospitalization"].to_numpy()
    nb = validation.net_benefit(y, validation.esi_as_score(df["acuity"].to_numpy()), [0.1, 0.3])
    assert len(nb) == 2
    slope, intercept = validation.calibration_slope_intercept(y, np.clip(y * 0.6 + 0.2, 0.01, 0.99))
    assert slope > 0
    assert 0 <= validation.expected_calibration_error(y, np.full(len(y), y.mean())) < 0.05
