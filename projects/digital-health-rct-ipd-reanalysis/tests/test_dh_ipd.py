"""Synthetic-IPD tests: engagement causal estimators, attrition MI, IECV and the registry client."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dh_ipd import (
    build_params,
    cace_wald,
    classify_repository,
    complete_case,
    fit_interaction_model,
    flatten_study,
    internal_external_cv,
    is_digital_intervention,
    itt,
    multiple_imputation,
    naive_per_protocol,
    predict_benefit,
    principal_score_effects,
    rubin_combine,
    sharing_summary,
    simulate_ipd,
    studies_to_frame,
    tipping_point,
)
from dh_ipd.engagement import bootstrap_estimates


@pytest.fixture(scope="module")
def ipd():
    return simulate_ipd(n_trials=5, n_per_trial=1200, ate=0.3, engagement_effect=0.3, hte_severity=0.0, seed=11)


def test_simulation_structure(ipd):
    assert {"trial", "z", "baseline", "engaged", "dose", "y", "y_full", "dropout", "S"} <= set(ipd.columns)
    assert ipd["trial"].nunique() == 5 and 0.1 < ipd["y"].isna().mean() < 0.5
    assert (ipd.loc[ipd["z"] == 0, "engaged"] == 0).all()


def test_itt_recovers_full_data_effect(ipd):
    full = ipd.assign(y=ipd["y_full"])
    truth = 0.3 + 0.3 * ipd["S"].mean()
    est = itt(full)
    assert abs(est["estimate"] - truth) < 0.08
    assert est["ci_low"] < est["estimate"] < est["ci_high"]


def test_naive_per_protocol_is_biased_and_cace_is_not(ipd):
    full = ipd.assign(y=ipd["y_full"])
    pp = naive_per_protocol(full)
    cace = cace_wald(full)
    truth_engagers = 0.3 + 0.3  # ate + engagement effect (no HTE)
    assert abs(cace["estimate"] - truth_engagers) < 0.12
    # motivation confounding inflates the naive contrasts above the causal engager effect
    assert pp["engagers_vs_controls"] > cace["estimate"] + 0.1
    assert pp["engagers_vs_nonengagers"] > 0.3 + 0.1
    ps = principal_score_effects(full, n_boot=30, seed=0)
    assert set(ps["stratum"]) == {0, 1}
    e0 = ps.loc[ps["stratum"] == 0, "effect"].iloc[0]
    e1 = ps.loc[ps["stratum"] == 1, "effect"].iloc[0]
    assert e1 > e0  # engagers benefit more


def test_bootstrap_cluster_ci(ipd):
    b = bootstrap_estimates(lambda d: itt(d), ipd, n_boot=30, seed=1)
    assert b["ci_low"] <= b["estimate"] <= b["ci_high"]


def test_missing_data_methods(ipd):
    cc = complete_case(ipd)
    mar = multiple_imputation(ipd, method="mar", m=10, seed=2)
    j2r = multiple_imputation(ipd, method="j2r", m=10, seed=2)
    assert cc["missing_rate"] > 0.1
    assert mar["se"] > 0 and j2r["se"] > 0 and mar["fmi"] >= 0
    # jump-to-reference removes post-dropout benefit -> smaller effect than MAR
    assert j2r["estimate"] < mar["estimate"]
    tp = tipping_point(ipd, deltas=(0.0, 0.5, 1.0, 2.0), m=5, seed=3)
    assert (tp["estimate"].diff().dropna() <= 1e-9).all()  # monotone in delta
    assert "tipping_delta" in tp.attrs


def test_rubin_combine_basic():
    r = rubin_combine([0.3, 0.32, 0.28], [0.01, 0.01, 0.01])
    assert abs(r["estimate"] - 0.3) < 1e-9 and r["se"] > 0.1 and 0 <= r["fmi"] <= 1


def test_hte_iecv_recovers_severity_interaction():
    d = simulate_ipd(n_trials=6, n_per_trial=1500, ate=0.3, engagement_effect=0.0, hte_severity=0.4, trial_sd=0.05, seed=5)
    d = d.assign(y=d["y_full"])
    m = fit_interaction_model(d, moderators=("baseline",))
    assert abs(m["coef"]["z:baseline"] - 0.4) < 0.1
    assert predict_benefit(m, d).std() > 0.2
    cv = internal_external_cv(d, moderators=("baseline",))
    assert len(cv) == 6 and np.isfinite(cv["cal_slope"]).all()
    assert abs(cv.attrs["pooled_cal_slope"] - 1.0) < 0.35


def test_registry_flatten_classify_and_summary():
    study = {
        "hasResults": True,
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Smartphone app for depression"},
            "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2019-03"}, "primaryCompletionDateStruct": {"date": "2021-01"}},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": ["NA"], "designInfo": {"allocation": "RANDOMIZED"}, "enrollmentInfo": {"count": 240}},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Some University", "class": "OTHER"}},
            "armsInterventionsModule": {"interventions": [{"type": "BEHAVIORAL", "name": "CBT app"}]},
            "conditionsModule": {"conditions": ["Depression"]},
            "ipdSharingStatementModule": {"ipdSharing": "YES", "description": "Data will be shared via Vivli after publication", "infoTypes": ["STUDY_PROTOCOL", "SAP"], "url": "https://vivli.org"},
        },
    }
    drug = {"protocolSection": {"identificationModule": {"nctId": "NCT00000002", "briefTitle": "Sertraline vs placebo"}, "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "sertraline"}]}, "ipdSharingStatementModule": {"ipdSharing": "NO"}}}
    flat = flatten_study(study)
    assert flat["nct_id"] == "NCT00000001" and flat["ipd_info_types"] == "STUDY_PROTOCOL;SAP" and flat["enrollment"] == 240
    assert classify_repository(flat["ipd_description"]) == "vivli"
    assert classify_repository("Available upon reasonable request from the PI") == "on_request"
    assert classify_repository("") == "empty" and classify_repository("We will not share") == "none"
    assert is_digital_intervention(flat) and not is_digital_intervention(flatten_study(drug))
    df = studies_to_frame([study, drug, study])
    assert len(df) == 2 and df["ipd_yes"].sum() == 1 and df.loc[0, "start_year"] == 2019
    summ = sharing_summary(df, by=("sponsor_class",))
    assert summ["n"].sum() == 2 and summ["n_vivli"].sum() == 1
    params = build_params("smartphone app", start_year=2016, ipd_yes_only=True)
    assert "AREA[IPDSharing]Yes" in params["query.term"] and params["pageSize"] == "100"
