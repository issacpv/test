"""Synthetic tests for ed_language."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ed_language import cohort, disparity, text_features


def test_harmonisation():
    assert cohort.harmonize_language("ENGLISH") == "english"
    assert cohort.harmonize_language("Spanish") == "spanish"
    assert cohort.harmonize_language("?") == "unknown"
    assert cohort.harmonize_language("Mandarin") == "chinese"
    assert cohort.harmonize_language("Swahili") == "other"
    assert cohort.harmonize_race("HISPANIC/LATINO - PUERTO RICAN") == "hispanic_latino"
    assert cohort.harmonize_race("BLACK/AFRICAN AMERICAN") == "black"
    assert cohort.harmonize_race("UNKNOWN") == "unknown"
    adm = pd.DataFrame({"subject_id": [1, 1, 1, 2, 3, 3], "language": ["ENGLISH", "?", "ENGLISH", "?", "Spanish", "ENGLISH"]})
    lg = cohort.language_by_subject(adm).set_index("subject_id")
    assert lg.loc[1, "language_group"] == "english" and not lg.loc[1, "language_conflict"]
    assert lg.loc[2, "language_group"] == "unknown"
    assert lg.loc[3, "language_conflict"]


def _synthetic_ed(n_subj: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2150-01-01")
    rows, adm_rows, icu_rows, tri_rows = [], [], [], []
    stay = 1
    hadm = 1000
    for s in range(1, n_subj + 1):
        nep = rng.random() < 0.3
        lang = "Spanish" if nep else "ENGLISH"
        adm_rows.append({"subject_id": s, "hadm_id": hadm + s, "language": lang, "insurance": "Medicaid", "marital_status": "SINGLE", "hospital_expire_flag": int(rng.random() < 0.02), "race": "HISPANIC/LATINO - PUERTO RICAN" if nep else "WHITE"})
        for v in range(2):
            intime = t0 + pd.Timedelta(days=int(rng.integers(0, 300)), hours=int(rng.integers(0, 24)))
            los = rng.gamma(4, 1.2) + (0.8 if nep else 0)
            acuity = int(rng.integers(1, 6))
            admitted = rng.random() < 0.3 + 0.1 * nep
            rows.append({"stay_id": stay, "subject_id": s, "hadm_id": hadm + s if admitted else np.nan, "intime": intime, "outtime": intime + pd.Timedelta(hours=los), "disposition": "ADMITTED" if admitted else "HOME", "arrival_transport": "WALK IN", "race": "HISPANIC/LATINO - PUERTO RICAN" if nep else "WHITE", "gender": "F"})
            n_tok = max(1, int(rng.poisson(2.0 if nep else 4.0)))
            cc = " ".join(rng.choice(["abd", "pain", "chest", "dyspnea", "fever", "fall", "weakness", "n/v"], n_tok)) + (", language barrier" if nep and rng.random() < 0.15 else "")
            tri_rows.append({"stay_id": stay, "temperature": 98.5, "heartrate": 80, "resprate": 16, "o2sat": 98, "sbp": 120, "dbp": 80, "pain": np.nan if rng.random() < (0.3 if nep else 0.1) else 3, "acuity": acuity, "chiefcomplaint": cc})
            if admitted and rng.random() < 0.1:
                icu_rows.append({"hadm_id": hadm + s, "intime": intime + pd.Timedelta(hours=los + 2)})
            stay += 1
    edstays = pd.DataFrame(rows)
    triage = pd.DataFrame(tri_rows)
    patients = pd.DataFrame({"subject_id": range(1, n_subj + 1), "anchor_age": rng.integers(20, 80, n_subj), "anchor_year": 2150, "anchor_year_group": "2017 - 2019"})
    admissions = pd.DataFrame(adm_rows)
    icustays = pd.DataFrame(icu_rows) if icu_rows else pd.DataFrame(columns=["hadm_id", "intime"])
    return edstays, triage, patients, admissions, icustays


def test_visit_table_and_text_features():
    edstays, triage, patients, admissions, icustays = _synthetic_ed()
    v = cohort.build_visit_table(edstays, triage, patients, admissions, icustays)
    assert len(v) == len(edstays)
    assert set(v["language_group"]) <= {"english", "spanish"}
    assert v["ed_los_h"].gt(0).all()
    assert v["critical"].isin([0, 1]).all() and v["ed_return"].isin([0, 1]).all()
    assert v.loc[v["nep"] == 1, "pain_missing"].mean() > v.loc[v["nep"] == 0, "pain_missing"].mean()
    f = text_features.chief_complaint_features(v["chiefcomplaint"])
    assert f.loc[v["nep"] == 1, "cc_n_tokens"].mean() < f.loc[v["nep"] == 0, "cc_n_tokens"].mean()
    assert f["cc_barrier"].sum() > 0 and (f.loc[v["nep"] == 0, "cc_barrier"] == 0).all()
    assert text_features.interpreter_mentions("History obtained via Spanish interpreter. Interpreter present.") == 2
    assert text_features.interpreter_mentions("No barrier noted.") == 0
    notes = pd.DataFrame({"hadm_id": [1, 1, 2], "text": ["used a phone interpreter", "none", "n/a"]})
    im = text_features.interpreter_mentions_by_hadm(notes).set_index("hadm_id")
    assert im.loc[1, "any_interpreter"] == 1 and im.loc[2, "any_interpreter"] == 0
    aud = cohort.selection_audit(v)
    assert "n_known" in aud.columns


def test_disparity_statistics():
    rng = np.random.default_rng(1)
    n = 3000
    nep = rng.binomial(1, 0.3, n)
    acuity = rng.integers(1, 6, n)
    age = rng.normal(55, 15, n)
    med = 4 - 2 * nep + rng.normal(0, 1, n)  # informativeness lower for NEP
    logit = -3 + 0.6 * nep - 0.25 * med - 0.4 * (acuity - 3) + 0.01 * (age - 55)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"subject_id": np.arange(n), "nep": nep, "acuity": acuity, "age": age, "cc_n_tokens": med, "critical": y, "language_group": np.where(nep == 1, "spanish", "english"), "sex": rng.choice(["F", "M"], n)})
    odds = disparity.adjusted_odds(df, "critical", "nep", ["age", "C(acuity)"])
    assert len(odds) == 1 and odds["or"].iloc[0] > 1.0
    ut = disparity.under_triage_by_group(df)
    assert set(ut["language_group"]) == {"english", "spanish"}
    esi = disparity.esi_discrimination_by_group(df, n_boot=50)
    assert (esi["auroc"] > 0.5).all()
    m = disparity.exact_match(df, "nep", ["acuity", "sex"], ratio=2)
    assert m.groupby("match_id").size().between(2, 3).all()
    med_res = disparity.mediation_analysis(df, "nep", "cc_n_tokens", "critical", ["age"], n_boot=20, cluster=None)
    assert med_res["total"] > 0 and med_res["nie"] > 0 and 0 < med_res["proportion_mediated"] < 1
    p = 1 / (1 + np.exp(-logit))
    fair = disparity.group_fairness(y, p, df["language_group"].to_numpy())
    assert len(fair) == 2 and "gap_sensitivity" in fair.attrs
    lo, hi = disparity.wilson_ci(5, 50)
    assert lo < 0.1 < hi
