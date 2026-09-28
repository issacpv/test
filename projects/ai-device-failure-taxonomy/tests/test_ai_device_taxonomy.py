"""Synthetic-data tests for ai_device_taxonomy (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_device_taxonomy import linkage as lk  # noqa: E402
from ai_device_taxonomy import rates as rt  # noqa: E402
from ai_device_taxonomy import taxonomy as tx  # noqa: E402
from ai_device_taxonomy.openfda_client import OpenFDAClient, flatten_device_event  # noqa: E402


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Two pages linked by a Link header, then a 404 for an unrelated query."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "page2" in url:
            return _Resp(200, {"results": [{"report_number": "3"}]}, {"Link": ""})
        if params and params.get("count"):
            return _Resp(200, {"results": [{"term": "Malfunction", "count": 10}, {"term": "Injury", "count": 2}]})
        if params and "nomatch" in str(params.get("search")):
            return _Resp(404, {"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
        return _Resp(200, {"results": [{"report_number": "1"}, {"report_number": "2"}]}, {"Link": '<https://api.fda.gov/device/event.json?page2=1>; rel="next"'})


def test_client_pagination_count_and_no_match():
    c = OpenFDAClient(session=_FakeSession(), api_key="k", max_per_minute=10000)
    recs = c.fetch_records("device/event", "device.device_report_product_code:QAS")
    assert [r["report_number"] for r in recs] == ["1", "2", "3"]
    cnt = c.count("device/event", None, "event_type.exact")
    assert cnt["count"].sum() == 12
    assert c.fetch_records("device/event", "nomatch") == []
    assert c.fetch_records("device/event", "x", max_records=1) == [{"report_number": "1"}]


def test_flatten_device_event():
    rec = {
        "report_number": "R1", "event_type": "Malfunction", "date_received": "20240102",
        "device": [{"brand_name": "AI Detect Pro", "generic_name": "software", "device_report_product_code": "QAS", "manufacturer_d_name": "Acme Medical Inc.", "openfda": {"device_class": "2"}}],
        "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": "The software missed the lesion."}, {"text_type_code": "Additional Manufacturer Narrative", "text": "Investigation ongoing."}],
        "product_problems": ["Incorrect Results"], "patient": [{"patient_problems": ["No Consequences"]}],
    }
    row = flatten_device_event(rec)
    assert row["product_code"] == "QAS" and row["text_event"].startswith("The software") and row["device_class"] == "2"
    assert row["product_problems"] == "Incorrect Results" and row["patient_problems"] == "No Consequences"


def test_linkage_tiers_and_report():
    ai = lk.canonicalise_ai_list(pd.DataFrame({
        "Date of Final Decision": ["2021-03-01", "2022-06-15"],
        "Submission Number": ["K210001", "K220002"],
        "Device": ["AI Detect Pro", "NeuroTriage"],
        "Company": ["Acme Medical, Inc.", "Brainy Health GmbH"],
        "Panel (Lead)": ["Radiology", "Radiology"],
        "Primary Product Code": ["QAS", "QAS"],
    }))
    assert ai["product_code"].tolist() == ["QAS", "QAS"] and ai["decision_date"].notna().all()
    maude = pd.DataFrame({
        "report_number": ["1", "2", "3", "4"],
        "product_code": ["QAS", "QAS", "QAS", "LLZ"],
        "brand_name": ["AI DETECT PRO v2", "Unknown viewer", "NEUROTRIAGE", "AI Detect Pro"],
        "manufacturer_d_name": ["ACME MEDICAL INC", "Brainy Health", "BRAINY HEALTH GMBH", "Acme"],
    })
    m = lk.candidate_matches(ai, maude)
    best = lk.best_match_per_report(m)
    tiers = dict(zip(best["report_number"], best["tier"]))
    assert tiers["1"] == 1 and tiers["3"] == 1 and tiers["2"] == 2
    assert "4" not in tiers  # different product code -> never matched
    rep = lk.linkage_report(m, ai)
    assert rep["n_reports_tier1"] == 2 and rep["n_ai_devices_with_reports"] == 2
    assert lk.similarity("Acme Medical, Inc.", "ACME MEDICAL INC") > 0.9
    k510 = pd.DataFrame({"k_number": ["K200009", "K210001", "K190001"], "product_code": ["QAS", "QAS", "QAS"], "applicant": ["Other Co", "Acme", "Old Co"], "device_name": ["Viewer", "AI Detect Pro", "Ancient"], "decision_date": ["2020-05-01", "2021-03-01", "2015-01-01"]})
    comp = lk.comparator_devices(k510, ai, years_window=2)
    assert set(comp["k_number"]) == {"K200009"}


def test_taxonomy_rules_classifier_and_agreement():
    texts = [
        "The algorithm produced a false negative and missed the nodule; the radiologist found it later.",
        "After the software update the application crashed and no output was generated for 3 hours.",
        "Images were sent to the wrong patient via the PACS interface.",
        "Poor image quality due to motion artifact; the study was not analyzable.",
        "Battery failure caused the device to shut down.",
        "User error: the technologist did not review the results.",
        "Unauthorized login attempt detected on the server.",
        "Routine maintenance, no issue.",
    ]
    df = pd.DataFrame({"text_event": texts})
    lab = tx.label_frame(df, extra_text_col=None)
    assert lab.loc[0, "tax_incorrect_output"] == 1 and lab.loc[0, "algorithm_implicated"] == 1
    assert lab.loc[1, "tax_no_output_or_delay"] == 1 and lab.loc[1, "tax_software_update_version"] == 1
    assert lab.loc[2, "tax_integration_interface"] == 1
    assert lab.loc[3, "tax_input_data_quality"] == 1
    assert lab.loc[4, "tax_hardware_of_device"] == 1
    assert lab.loc[5, "tax_user_workflow_human_factors"] == 1
    assert lab.loc[6, "tax_cybersecurity_access"] == 1
    assert lab.loc[7, "n_categories"] == 0
    assert "algorithm" not in tx.strip_brand_tokens("Acme Detect flagged", "Acme Detect", None).lower()
    # classifier on a larger weak-labelled synthetic corpus
    rng = np.random.default_rng(0)
    corpus = [texts[i % len(texts)] + f" case {rng.integers(1000)}" for i in range(80)]
    cdf = pd.DataFrame({"text_event": corpus})
    weak = tx.label_frame(cdf, extra_text_col=None)
    pipe = tx.train_weak_classifier(corpus, weak, min_positive=5)
    pred = tx.predict_categories(pipe, ["The AI missed the lesion, a false negative."])
    assert pred["pred_incorrect_output"].iloc[0] == 1
    ag = tx.agreement([1, 0, 1, 1, 0], [1, 0, 1, 0, 0])
    assert 0 < ag["kappa"] < 1 and ag["agreement"] == pytest.approx(0.8)


def test_rates_and_update_windows():
    rr = rt.rate_ratio(30, 10.0, 10, 10.0)
    assert rr["irr"] == pytest.approx(3.0) and rr["ci_lo"] < 3.0 < rr["ci_hi"] and rr["p_exact"] < 0.01
    rng = np.random.default_rng(1)
    n = 400
    ai = rng.binomial(1, 0.5, n)
    yrs = rng.uniform(0.5, 5, n)
    counts = rng.poisson(0.8 * yrs * np.exp(0.7 * ai))
    df = pd.DataFrame({"count": counts, "years": yrs, "ai": ai, "firm": rng.integers(0, 40, n)})
    fit = rt.negbin_rate_model(df, "count", "years", ["ai"], cluster_col="firm")
    assert fit.loc["ai", "ci_lo"] < np.exp(0.7) < fit.loc["ai", "ci_hi"]
    dy = rt.device_years(pd.Series(["2020-01-01", "2023-01-01"]), pd.Timestamp("2024-01-01"), reporting_start=pd.Timestamp("2021-01-01"))
    assert dy.iloc[0] == pytest.approx(3.0, abs=0.01) and dy.iloc[1] == pytest.approx(1.0, abs=0.01)
    ev = pd.Series(pd.to_datetime(["2023-01-10", "2023-02-01", "2023-07-15", "2023-08-01", "2023-09-01"]))
    pp = rt.pre_post_update_counts(ev, [pd.Timestamp("2023-06-01")], window_days=180)
    assert pp.loc[0, "pre"] == 2 and pp.loc[0, "post"] == 3
    mc = rt.monthly_counts(ev)
    assert mc.sum() == 5 and len(mc) == 9
    share = rt.category_share_table(pd.DataFrame({"tax_incorrect_output": [1, 0, 1, 1]}), ["incorrect_output"])
    assert share["share"].iloc[0] == pytest.approx(0.75) and share["ci_lo"].iloc[0] < 0.75 < share["ci_hi"].iloc[0]
