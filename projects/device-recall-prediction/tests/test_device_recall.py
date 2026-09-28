"""Synthetic tests for device_recall (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from device_recall import entity_linking as el  # noqa: E402
from device_recall import predicate_graph as pg  # noqa: E402
from device_recall import survival_text as st  # noqa: E402
from device_recall.openfda_device import flatten_maude, join_recall_class, explode_recall_submissions  # noqa: E402


# ------------------------------------------------------------ predicates
SUMMARY = """
510(k) SUMMARY  K213456
Device: ACME Insulin Pump Model 3.
Predicate Device: ACME Insulin Pump Model 2 (K193210). The subject device is
substantially equivalent to the predicate. Reference device: K181111 was used
for the infusion set testing. Also compared to P010045 in the literature.
Standards: ISO 10993. Contact: K. Smith.
"""


def test_extract_predicates_context_and_self_exclusion():
    preds = pg.extract_predicates(SUMMARY, self_number="K213456")
    nums = {p["number"]: p for p in preds}
    assert "K213456" not in nums
    assert nums["K193210"]["predicate_cue"] is True
    assert nums["K181111"]["reference_cue"] is True
    assert nums["P010045"]["kind"] == "P"


def test_summary_pdf_url_folders():
    assert pg.summary_pdf_url("K213456").endswith("/pdf21/K213456.pdf")
    assert pg.summary_pdf_url("K053456").endswith("/pdf5/K053456.pdf")
    assert pg.summary_pdf_url("K963456").endswith("/pdf/K963456.pdf")


def _chain_graph():
    # A -> B -> C -> D (root); E -> B; F cites A and a future device G (invalid)
    edges = pd.DataFrame(
        {
            "k_number": ["A", "B", "C", "E", "F", "F", "G"],
            "predicate": ["B", "C", "D", "B", "A", "G", "F"],
        }
    )
    dates = {n: pd.Timestamp(d) for n, d in zip("ABCDEFG", ["2020-01-01", "2016-01-01", "2012-01-01", "2008-01-01", "2019-01-01", "2021-01-01", "2022-01-01"])}
    G = pg.build_predicate_graph(edges)
    return G, dates


def test_predicate_depth_with_invalid_edge_removal():
    G, dates = _chain_graph()
    with pytest.raises(ValueError):
        pg.predicate_depth(G)  # F<->G cycle
    H = pg.remove_invalid_edges(G, dates)
    assert not H.has_edge("F", "G")  # G cleared after F
    depth = pg.predicate_depth(H)
    assert depth["D"] == 0 and depth["C"] == 1 and depth["B"] == 2 and depth["A"] == 3 and depth["F"] == 4
    feats = pg.graph_features(H)
    assert feats.loc["B", "n_children"] == 2
    assert feats.loc["A", "n_ancestors"] == 3


def test_ancestor_recall_exposure():
    G, dates = _chain_graph()
    H = pg.remove_invalid_edges(G, dates)
    recalls = pd.DataFrame(
        {"submission_number": ["C", "D"], "recall_date": [pd.Timestamp("2015-06-01"), pd.Timestamp("2021-06-01")], "recall_class": ["Class I", "Class II"]}
    )
    exp = pg.ancestor_recall_exposure(H, recalls, dates)
    assert exp.loc["A", "n_recalled_ancestors"] == 2
    assert exp.loc["A", "n_class1_ancestors"] == 1
    assert exp.loc["A", "generations_to_recalled_ancestor"] == 2
    assert bool(exp.loc["B", "direct_predicate_recalled_before_clearance"]) is True  # C recalled 2015 < B cleared 2016
    assert bool(exp.loc["A", "direct_predicate_recalled_before_clearance"]) is False
    assert bool(exp.loc["A", "any_ancestor_recalled_before_clearance"]) is True


def test_edges_from_summaries():
    e = pg.edges_from_summaries({"K213456": SUMMARY})
    assert set(e["predicate"]) >= {"K193210", "K181111"}


# --------------------------------------------------------- entity linking
def test_normalize_and_similarity():
    assert el.normalize_firm("Medtronic, Inc.") == "medtronic"
    assert el.normalize_firm("ABBOTT VASCULAR INC") == "abbott vascular"
    assert el.similarity("medtronic", "medtronic minimed") > 0.7
    assert el.similarity("medtronic", "boston scientific") < 0.5


def test_link_maude_to_510k_blocks_by_product_code_and_date():
    maude = pd.DataFrame(
        {
            "mdr_report_key": ["1", "2", "3"],
            "product_code": ["LZG", "LZG", "OYC"],
            "manufacturer_d_name": ["Medtronic MiniMed, Inc.", "Insulet Corp", "Medtronic MiniMed"],
            "brand_name": ["MiniMed 670G", "Omnipod", "Guardian"],
            "date_received": pd.to_datetime(["2020-05-01", "2020-06-01", "2020-07-01"]),
        }
    )
    k510 = pd.DataFrame(
        {
            "k_number": ["K1", "K2", "K3"],
            "product_code": ["LZG", "LZG", "LZG"],
            "applicant": ["Medtronic MiniMed Inc", "Insulet Corporation", "Medtronic MiniMed Inc"],
            "device_name": ["MiniMed 670G System", "Omnipod Insulin Management System", "MiniMed 630G"],
            "decision_date": pd.to_datetime(["2019-01-01", "2021-01-01", "2018-01-01"]),
        }
    )
    links = el.link_maude_to_510k(maude, k510)
    l = links.set_index("mdr_report_key")
    assert l.loc["1", "k_number"] == "K1"  # brand similarity breaks the Medtronic tie
    assert pd.isna(l.loc["2", "k_number"])  # K2 cleared after the report
    assert pd.isna(l.loc["3", "k_number"])  # different product code block


# ---------------------------------------------------------- flatten/join
def test_flatten_maude_and_recall_join():
    rec = {
        "mdr_report_key": "9",
        "date_received": "20230102",
        "event_type": "Injury",
        "device": [{"brand_name": "X", "device_report_product_code": "LZG", "manufacturer_d_name": "ACME", "openfda": {"device_class": "2"}}],
        "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": "pump occluded"}, {"text_type_code": "Additional Manufacturer Narrative", "text": "analysis pending"}],
        "patient": [{"sequence_number_outcome": ["Hospitalization"]}],
    }
    row = flatten_maude(rec)
    assert row["narrative"] == "pump occluded" and row["manufacturer_narrative"] == "analysis pending"
    assert row["device_class"] == "2" and row["patient_outcomes"] == ["Hospitalization"]

    recalls = pd.DataFrame({"res_event_number": ["E1"], "event_date_initiated": [pd.Timestamp("2022-01-01")], "k_numbers": [["K1", "K9"]], "pma_numbers": [[]], "product_code": ["LZG"]})
    enf = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "classification": ["Class II", "Class I"],
            "recall_initiation_date": [pd.Timestamp("2022-01-05"), pd.Timestamp("2022-01-03")],
            "center_classification_date": [pd.NaT, pd.NaT],
        }
    )
    joined = join_recall_class(recalls, enf)
    assert joined["recall_class"].iloc[0] == "Class I"
    ex = explode_recall_submissions(joined)
    assert set(ex["submission_number"]) == {"K1", "K9"}


# ------------------------------------------------------------- survival
def _synthetic_cohort(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2005-01-01") + pd.to_timedelta(rng.integers(0, 365 * 12, n), unit="D")
    depth = rng.integers(0, 8, n)
    early_reports = rng.poisson(1.5, n)
    lam = 0.004 * np.exp(0.25 * depth + 0.15 * early_reports)
    t_recall = rng.exponential(1 / lam)  # months
    devices = pd.DataFrame({"k_number": [f"K{i:06d}" for i in range(n)], "decision_date": dates, "predicate_depth": depth, "n_reports": early_reports})
    recalls = pd.DataFrame(
        {
            "submission_number": devices["k_number"],
            "recall_date": dates + pd.to_timedelta(t_recall * 30.4375, unit="D"),
            "recall_class": rng.choice(["Class I", "Class II"], n),
        }
    )
    return devices, recalls


def test_survival_table_and_cox_recover_hazard():
    devices, recalls = _synthetic_cohort()
    tab = st.build_survival_table(devices, recalls, study_end="2022-12-31", landmark_months=12)
    assert {"duration_months", "event"} <= set(tab.columns)
    assert tab["duration_months"].min() > 0
    assert 0 < tab["event"].mean() < 1
    res = st.fit_cox(tab, ["predicate_depth", "n_reports"], penalizer=0.0)
    assert res.summary.loc["predicate_depth", "hr"] > 1.1
    assert res.concordance_train > 0.6
    ev = st.temporal_split_evaluate(tab, ["predicate_depth", "n_reports"], split_date="2012-01-01", penalizer=0.0)
    assert ev["c_test"] > 0.55


def test_maude_landmark_features_and_tfidf():
    devices = pd.DataFrame({"k_number": ["K1", "K2"], "decision_date": pd.to_datetime(["2020-01-01", "2020-01-01"])})
    maude = pd.DataFrame(
        {
            "k_number": ["K1", "K1", "K1", "K2"],
            "date_received": pd.to_datetime(["2020-02-01", "2020-06-01", "2021-06-01", "2020-03-01"]),
            "event_type": ["Injury", "Malfunction", "Death", "Malfunction"],
            "product_problems": [["Occlusion"], ["Occlusion", "Leak"], ["Failure"], []],
            "narrative": ["pump occluded and alarm", "insulin leak at cannula", "late report", "battery error"],
        }
    )
    feats = st.maude_landmark_features(maude, devices, landmark_months=12)
    f = feats.set_index("k_number")
    assert f.loc["K1", "n_reports"] == 2  # third report is after the landmark
    assert f.loc["K1", "n_distinct_problems"] == 2
    assert "late report" not in f.loc["K1", "narrative"]
    X, vec, svd = st.tfidf_features(list(feats["narrative"]) + ["pump alarm occluded", "battery error again"], n_components=2)
    assert X.shape == (4, 2)


def test_concordance_index_perfect_and_random():
    t = np.array([1, 2, 3, 4.0])
    e = np.array([1, 1, 1, 1])
    assert st.concordance_index(t, e, -t) == 1.0
    assert st.concordance_index(t, e, t) == 0.0
