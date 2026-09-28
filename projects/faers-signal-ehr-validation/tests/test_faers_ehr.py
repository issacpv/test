"""Synthetic tests for disproportionality statistics, cohort/PS matching, outcomes and client helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faers_ehr import cohort, disproportionality as dp, openfda_client as ofc, outcomes, rxnorm  # noqa: E402


def test_prr_ror_ic_hand_values():
    a, b, c, d = 20, 980, 100, 98900
    r = dp.prr(a, b, c, d)
    assert abs(r["prr"] - (20 / 1000) / (100 / 99000)) < 1e-9
    assert r["prr_lo"] < r["prr"] < r["prr_hi"] and r["chi2"] > 4
    o = dp.ror(a, b, c, d)
    assert abs(o["ror"] - (20 / 980) / (100 / 98900)) < 1e-9
    ic = dp.bcpnn_ic(a, r["expected"])
    assert ic["ic025"] < ic["ic"] < ic["ic975"] and ic["ic025"] > 0
    # no disproportion -> IC025 < 0
    e0 = dp.expected_counts(pd.DataFrame({"a": [5], "b": [995], "c": [500], "d": [99500]})).iloc[0]
    assert dp.bcpnn_ic(5, e0)["ic025"] < 0


def test_mgps_shrinks_small_counts_and_fits_prior():
    rng = np.random.default_rng(0)
    n_cells = 3000
    expected = rng.gamma(2.0, 2.0, n_cells)
    lam = np.where(rng.uniform(size=n_cells) < 0.9, rng.gamma(20, 1 / 20, n_cells), rng.gamma(2, 2.0, n_cells))
    observed = rng.poisson(lam * expected)
    prior = dp.fit_mgps_prior(observed, expected)
    assert 0.0 < prior.p < 1.0 and prior.alpha1 > 0
    small = dp.ebgm(3, 1.0, prior)   # crude RR = 3 on tiny counts
    large = dp.ebgm(300, 100.0, prior)
    assert small["ebgm"] < 3.0 and small["eb05"] < small["ebgm"]
    assert abs(large["ebgm"] - 3.0) < 0.3
    assert (3.0 - small["ebgm"]) > (3.0 - large["ebgm"])


def test_signal_table_and_performance():
    rng = np.random.default_rng(1)
    rows = []
    truth = []
    for i in range(60):
        n = 1_000_000
        drug_total = int(rng.integers(200, 5000))
        event_total = int(rng.integers(500, 20000))
        base = drug_total * event_total / n
        real = i % 3 == 0
        a = rng.poisson(base * (4.0 if real else 1.0))
        rows.append({"drug": f"d{i}", "outcome": "x", "a": a, "b": drug_total - a, "c": event_total - a,
                     "d": n - drug_total - event_total + a, "n": n})
        truth.append(1 if real else 0)
    tbl = dp.signal_table(pd.DataFrame(rows))
    for col in ("prr", "ror", "ic025", "eb05", "signal_prr", "signal_ebgm"):
        assert col in tbl
    perf = dp.signal_performance(tbl["ic"], pd.Series(truth), tbl["signal_ic"])
    assert perf["auroc"] > 0.9 and perf["ppv"] > 0.6
    lo, hi = dp.wilson_interval(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0


def test_stratified_expected_sums_over_strata():
    long = pd.DataFrame({"drug": ["A", "A", "B", "B"] * 2, "event": ["E", "F", "E", "F"] * 2,
                         "stratum": ["s1"] * 4 + ["s2"] * 4, "n": [10, 90, 20, 880, 5, 5, 50, 940]})
    out = dp.stratified_expected(long)
    row = out[(out.drug == "A") & (out.event == "E")].iloc[0]
    e_s1 = 100 * 30 / 1000
    e_s2 = 10 * 55 / 1000
    assert abs(row["expected"] - (e_s1 + e_s2)) < 1e-9 and row["observed"] == 15


def test_openfda_query_builders_and_dedup():
    q = ofc.drug_query("amiodarone")
    assert 'patient.drug.openfda.generic_name:"AMIODARONE"' in q and "drugcharacterization:1" in q
    eq = ofc.event_query(ofc.OUTCOME_PT["hyperkalaemia"])
    assert "HYPERKALAEMIA" in eq and "+OR+" in eq
    assert ofc._next_link('<https://api.fda.gov/drug/event.json?search_after=1>; rel="next"') == \
        "https://api.fda.gov/drug/event.json?search_after=1"
    reports = [{"safetyreportid": "1", "safetyreportversion": "1", "receivedate": "20200101",
                "primarysource": {"qualification": "1"},
                "patient": {"drug": [{"medicinalproduct": "X", "drugcharacterization": "1", "openfda": {"generic_name": ["X"]}}],
                            "reaction": [{"reactionmeddrapt": "HYPERKALAEMIA"}]}},
               {"safetyreportid": "1", "safetyreportversion": "2", "receivedate": "20200201",
                "primarysource": {"qualification": "1"},
                "patient": {"drug": [{"medicinalproduct": "X", "drugcharacterization": "1", "openfda": {"generic_name": ["X"]}}],
                            "reaction": [{"reactionmeddrapt": "HYPERKALAEMIA"}]}}]
    df = ofc.reports_to_frame(reports)
    assert len(df) == 2 and len(ofc.deduplicate_reports(df)) == 1


def test_rxnorm_helpers_offline():
    assert rxnorm.normalize_ndc("00006-0227-31".replace("-", "")) == "00006022731"
    assert rxnorm.normalize_ndc(6022731) == "00006022731"
    assert rxnorm.normalize_ndc("0") is None and rxnorm.normalize_ndc(None) is None
    assert rxnorm.ingredient_key("Metoprolol Tartrate 25 mg") == "metoprolol"
    assert rxnorm.ingredient_key("PIPERACILLIN-TAZOBACTAM") == "piperacillin tazobactam"


def test_incident_outcome_and_kdigo_and_qtc():
    idx = pd.DataFrame({"subject_id": [1, 2, 3], "index_time": pd.to_datetime(["2150-01-10"] * 3)})
    labs = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3, 3],
        "charttime": pd.to_datetime(["2150-01-08", "2150-01-12", "2150-01-13", "2150-01-09", "2150-01-12",
                                     "2150-01-09", "2150-01-11"]),
        "lab": ["potassium"] * 7,
        "valuenum": [4.0, 4.5, 5.9, 5.6, 5.8, 4.2, 4.4],
    })
    defs = outcomes.make_outcomes()
    res = outcomes.incident_outcome(labs, idx, defs["hyperkalaemia"])
    r = res.set_index("subject_id")
    assert r.loc[1, "outcome"] == 1 and r.loc[1, "eligible"]
    assert not r.loc[2, "eligible"]  # baseline already high -> prevalent, not incident
    assert r.loc[3, "outcome"] == 0 and r.loc[3, "eligible"]

    creat = pd.DataFrame({"subject_id": [1, 1, 1], "charttime": pd.to_datetime(["2150-01-09", "2150-01-12", "2150-01-14"]),
                          "lab": ["creatinine"] * 3, "valuenum": [0.8, 1.0, 1.3]})
    aki = outcomes.incident_outcome(creat, idx.head(1), defs["aki"]).iloc[0]
    assert aki["outcome"] == 1 and aki["event_time"] == pd.Timestamp("2150-01-14")

    qtc = pd.DataFrame({"subject_id": [1, 1, 1], "ecg_time": pd.to_datetime(["2150-01-09", "2150-01-11", "2150-01-12"]),
                        "qtc_ms": [440.0, 470.0, 510.0], "wide_qrs": [False, False, False]})
    q = outcomes.qtc_outcome(qtc, idx.head(1)).iloc[0]
    assert q["outcome"] == 1 and abs(q["delta_qtc"] - 70) < 1e-9 and q["eligible"]
    sql = outcomes.labs_sql("/data/mimiciv", ("potassium", "creatinine"))
    assert "50971" in sql and "50912" in sql and "labevents" in sql
    assert "SQRT" in outcomes.qtc_sql("/data/ecg")


def test_new_user_cohort():
    ex = pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 3],
        "hadm_id": [10, 10, 20, 30, 31],
        "ingredient": ["a", "b", "b", "a", "a"],
        "index_time": pd.to_datetime(["2150-01-01", "2150-01-03", "2150-02-01", "2150-03-01", "2150-05-01"]),
        "prev_admin_time": pd.to_datetime([None, None, "2149-12-01", None, "2150-03-01"]),
        "admin_rank": [1, 1, 1, 1, 1],
    })
    c = cohort.new_user_cohort(ex, "a", "b", washout_days=180)
    assert set(c["subject_id"]) == {1, 3}          # subject 2 fails the washout
    assert c.set_index("subject_id").loc[1, "treated"] == 1 and c.set_index("subject_id").loc[1, "both_drugs"]
    assert c.set_index("subject_id").loc[3, "hadm_id"] == 30   # earliest qualifying admission


def test_propensity_matching_balances_and_recovers_effect():
    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({"age": rng.normal(60, 12, n), "sev": rng.normal(0, 1, n), "female": rng.integers(0, 2, n)})
    logit_t = -0.5 + 0.04 * (X["age"] - 60) + 0.8 * X["sev"]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_t))).astype(int)
    true_rr = 2.0
    base = 0.05 * np.exp(0.6 * X["sev"])              # confounder raises risk
    p = np.clip(base * np.where(t == 1, true_rr, 1.0), 0, 0.95)
    y = (rng.uniform(size=n) < p).astype(int)
    crude = cohort.risk_ratio(y, t)
    ps = cohort.propensity_scores(X, t)
    m = cohort.match_nearest(ps, t, caliper_sd=0.2)
    assert len(m) > 500
    idx = np.r_[m["treated_idx"], m["control_idx"]]
    tt = np.r_[np.ones(len(m)), np.zeros(len(m))]
    smd_before = cohort.standardized_mean_differences(X, t)
    smd_after = cohort.standardized_mean_differences(X.iloc[idx].reset_index(drop=True), tt)
    assert smd_before["sev"] > 0.3 and smd_after.max() < 0.1
    matched = cohort.matched_risk_ratio(y, m)
    assert abs(np.log(matched["rr"]) - np.log(true_rr)) < abs(np.log(crude["rr"]) - np.log(true_rr))
    w = cohort.iptw_weights(ps, t)
    smd_w = cohort.standardized_mean_differences(X, t, w)
    assert smd_w.max() < 0.15
    boot = cohort.bootstrap_effect(y, t, X, n_boot=10)
    assert boot["n_boot_ok"] >= 5 and boot["rr_lo"] < boot["rr"] < boot["rr_hi"]


def test_empirical_calibration_and_classification():
    rng = np.random.default_rng(4)
    se = rng.uniform(0.1, 0.3, 40)
    log_rr_nc = rng.normal(0.2, 0.25, 40) + rng.normal(0, 1, 40) * se   # biased, over-dispersed
    syserr = cohort.fit_systematic_error(log_rr_nc, se)
    assert 0.05 < syserr.mu < 0.35 and syserr.tau > 0.1
    naive_p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(0.4 / 0.15)))
    cal_p = syserr.calibrated_p(0.4, 0.15)
    assert cal_p > naive_p
    lo, hi = syserr.calibrated_ci(0.4, 0.15)
    assert lo < np.exp(0.4) < hi
    assert cohort.classify_pair(2.0, 1.4, 2.9, 0.01) == "positive"
    assert cohort.classify_pair(1.0, 0.8, 1.2, 0.9) == "negative"
    assert cohort.classify_pair(1.3, 0.9, 1.9, 0.2) == "indeterminate"
