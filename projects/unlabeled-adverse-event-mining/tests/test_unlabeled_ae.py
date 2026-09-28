"""Synthetic tests for unlabeled_ae (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlabeled_ae import label_client as lc  # noqa: E402
from unlabeled_ae import label_lag as ll  # noqa: E402
from unlabeled_ae import signal_scan as ss  # noqa: E402
from unlabeled_ae import term_extractor as te  # noqa: E402

LABEL_REC = {
    "set_id": "abc-123",
    "id": "v7-guid",
    "version": "7",
    "effective_time": "20230415",
    "openfda": {"generic_name": ["ATORVASTATIN CALCIUM"], "brand_name": ["LIPITOR"], "rxcui": ["83367"], "application_number": ["NDA020702"], "product_type": ["HUMAN PRESCRIPTION DRUG"]},
    "boxed_warning": ["WARNING: HEPATIC FAILURE. Cases of hepatic failure have been reported."],
    "warnings_and_cautions": ["5 WARNINGS AND PRECAUTIONS 5.1 Myopathy and Rhabdomyolysis Cases of myopathy and rhabdomyolysis have been reported. 5.2 Liver Enzyme Abnormalities Persistent elevations in liver enzymes can occur."],
    "adverse_reactions": [
        "6 ADVERSE REACTIONS 6.1 Clinical Trials Experience The most common adverse reactions were nasopharyngitis, arthralgia, diarrhea, pain in extremity, and urinary tract infection. No cases of pancreatitis were observed. "
        "6.2 Postmarketing Experience The following adverse reactions have been identified during postapproval use: myalgia, fatigue, tendon rupture, memory loss, and depression."
    ],
}


def test_parse_label_and_subsections():
    doc = lc.parse_label(LABEL_REC)
    assert doc.generic_name == "ATORVASTATIN CALCIUM" and doc.version == 7
    assert doc.effective_time == pd.Timestamp("2023-04-15")
    assert "boxed_warning" in doc.sections and "adverse_reactions" in doc.sections
    subs = lc.split_subsections(doc.sections["adverse_reactions"])
    assert any("Postmarketing" in k for k in subs)
    assert "tendon rupture" in doc.postmarketing_text


def test_parse_spl_xml_sections():
    xml = """<?xml version="1.0"?>
    <document xmlns="urn:hl7-org:v3"><component><structuredBody><component>
      <section><code code="34084-4" displayName="ADVERSE REACTIONS SECTION"/><title>6 ADVERSE REACTIONS</title>
        <text><paragraph>Headache and nausea were common.</paragraph></text>
        <component><section><code code="34084-4"/><text>Postmarketing: rhabdomyolysis.</text></section></component>
      </section></component><component>
      <section><code code="34066-1"/><text>WARNING: hepatic failure.</text></section>
    </component></structuredBody></component></document>"""
    secs = lc.parse_spl_xml_sections(xml)
    assert "Headache and nausea" in secs["adverse_reactions"]
    assert "rhabdomyolysis" in secs["adverse_reactions"].lower()
    assert "hepatic failure" in secs["boxed_warning"].lower()


def test_extract_terms_synonyms_negation_plurals():
    d = te.MedDRADictionary.from_seed()
    text = "Patients reported headaches and muscle pain. No cases of rhabdomyolysis were seen. Sleep-walking occurred rarely."
    ments = te.extract_terms(text, d)
    pts = {m.pt: m for m in ments}
    assert "Headache" in pts and "Myalgia" in pts and "Somnambulism" in pts
    assert pts["Rhabdomyolysis"].negated is True
    assert pts["Headache"].negated is False
    assert d.match_string("Diarrhea") == "Diarrhoea"
    assert d.match_string("Rhabdomyolysis") == "Rhabdomyolysis"


def test_extract_from_label_postmarketing_only_flag():
    d = te.MedDRADictionary.from_seed()
    doc = lc.parse_label(LABEL_REC)
    tab = te.extract_from_label(doc, d)
    got = tab.set_index(["section", "pt"])
    assert ("adverse_reactions", "Tendon rupture") in got.index
    assert bool(got.loc[("adverse_reactions", "Tendon rupture"), "postmarketing_only"]) is True
    assert bool(got.loc[("adverse_reactions", "Diarrhoea"), "postmarketing_only"]) is False
    # pancreatitis is only mentioned negated -> not affirmed
    assert got.loc[("adverse_reactions", "Pancreatitis"), "n_affirmed"] == 0
    labeled = te.labeled_terms(tab)
    assert "Rhabdomyolysis" in labeled and "Pancreatitis" not in labeled


def _synthetic_reports(seed: int = 0, n_quarters: int = 24, emerge_at: int = 10):
    """Background reports plus a drug-event association switched on at quarter `emerge_at`."""
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    quarters = pd.period_range("2015Q1", periods=n_quarters, freq="Q")
    drugs = ["DRUGA", "DRUGB", "DRUGC", "DRUGD"]
    pts = ["Headache", "Nausea", "Rash", "Rhabdomyolysis", "Dizziness"]
    for qi, q in enumerate(quarters):
        for _ in range(400):
            rid += 1
            drug = drugs[rng.integers(4)]
            # baseline: uniform events; after emergence, DRUGA strongly linked to Rhabdomyolysis
            if drug == "DRUGA" and qi >= emerge_at and rng.random() < 0.35:
                pt = "Rhabdomyolysis"
            else:
                pt = pts[rng.integers(5)] if rng.random() < 0.9 else pts[rng.integers(3)]
            rows.append({"report_id": rid, "quarter": q, "drug": drug, "pt": pt})
    return pd.DataFrame(rows), quarters


def test_time_scan_dates_emergence():
    reports, quarters = _synthetic_reports()
    tables = ss.cumulative_pair_tables(reports, pairs=[("DRUGA", "Rhabdomyolysis"), ("DRUGB", "Rhabdomyolysis")])
    assert len(tables) == 2 * len(quarters)
    # cells are consistent: a+b+c+d == cumulative distinct reports
    last = tables[tables["quarter"] == quarters[-1]].iloc[0]
    assert last[["a", "b", "c", "d"]].sum() == reports["report_id"].nunique()
    scan = ss.time_scan(tables, criterion="ic", min_n=3, k_sustain=2)
    s = scan.set_index(["drug", "pt"])
    fs = s.loc[("DRUGA", "Rhabdomyolysis"), "first_signal_quarter"]
    assert pd.notna(fs)
    assert quarters[10] <= fs <= quarters[13]
    assert pd.isna(s.loc[("DRUGB", "Rhabdomyolysis"), "first_signal_quarter"])
    assert s.loc[("DRUGA", "Rhabdomyolysis"), "signal_current"]


def test_disproportionality_values():
    est = ss.disproportionality([50], [950], [100], [8900])
    assert est["ror"].iloc[0] == pytest.approx((50 * 8900) / (950 * 100))
    assert est["ic025"].iloc[0] > 0


def test_flag_and_rank_unlabeled():
    d = te.MedDRADictionary.from_seed()
    scan = pd.DataFrame(
        {
            "drug": ["DRUGA", "DRUGA", "DRUGA", "DRUGZ"],
            "pt": ["Rhabdomyolysis", "Diarrhea", "Tendon rupture", "Rash"],
            "first_signal_quarter": [pd.Period("2017Q3", "Q")] * 4,
            "n_current": [40, 30, 12, 5],
            "ic_current": [2.0, 1.0, 1.5, 0.8],
            "ic025_current": [1.5, 0.6, 0.9, 0.2],
            "ror_current": [4.0, 2.0, 3.0, 1.5],
            "ror025_current": [3.0, 1.5, 1.8, 1.1],
            "signal_current": [True, True, True, True],
            "n_last_4q": [10, 2, 6, 1],
        }
    )
    labeled = {"DRUGA": {"Rhabdomyolysis", "Diarrhoea"}}
    flagged = ss.flag_unlabeled(scan, labeled, dictionary=d)
    m = flagged.set_index("pt")["label_match"]
    assert m["Rhabdomyolysis"] == "exact" and m["Diarrhea"] == "synonym" and m["Tendon rupture"] == "none" and m["Rash"] == "no_label"
    ranked = ss.rank_unlabeled(flagged)
    assert list(ranked["pt"]) == ["Tendon rupture"]


def test_label_lag_and_km_and_fda_eval():
    signals = pd.DataFrame(
        {
            "drug": ["DRUGA", "DRUGA", "DRUGA"],
            "pt": ["Tendon rupture", "Myalgia", "Rash"],
            "first_signal_quarter": [pd.Period("2016Q1", "Q"), pd.Period("2018Q1", "Q"), pd.NaT],
            "n_current": [12, 40, 3],
            "ic025_current": [0.9, 1.2, -0.1],
            "signal_current": [True, True, False],
            "on_label": [False, True, False],
        }
    )
    version_terms = pd.DataFrame(
        {
            "drug": ["DRUGA", "DRUGA", "DRUGA"],
            "set_id": ["s1", "s1", "s1"],
            "version": [3, 5, 5],
            "effective_time": ["2017-07-01", "2019-01-01", "2017-01-01"],
            "pt": ["Tendon rupture", "Tendon rupture", "Myalgia"],
        }
    )
    first = ll.first_labeled_dates(version_terms)
    assert first.set_index("pt").loc["Tendon rupture", "first_labeled_date"] == pd.Timestamp("2017-07-01")
    lag = ll.compute_label_lag(signals, first, data_cut="2024-12-31")
    l = lag.set_index("pt")
    assert l.loc["Tendon rupture", "category"] == "signal_first"
    assert l.loc["Tendon rupture", "lag_months"] == pytest.approx(18, abs=0.6)
    assert l.loc["Myalgia", "category"] == "label_first"
    assert l.loc["Rash", "category"] == "never_signal"
    km = ll.kaplan_meier([3, 6, 9, 12, np.nan], [1, 0, 1, 0, 1])
    assert km["survival"].iloc[-1] < 1 and (np.diff(km["survival"]) <= 0).all()

    fda = pd.DataFrame({"quarter": ["April - June 2017", "2019Q2"], "product": ["DrugA (Brand)", "DrugA"], "signal_text": ["Tendon rupture", "Rash"]})
    fda_pairs = ll.load_potential_signals(fda, dictionary=te.MedDRADictionary.from_seed())
    assert set(fda_pairs["pt"]) == {"Tendon rupture", "Rash"}
    assert fda_pairs["quarter"].iloc[0] == pd.Period("2017Q2", "Q")
    ev = ll.evaluate_against_fda(signals, fda_pairs)
    assert ev["n_fda_signals"] == 2
    assert ev["detected_rate"] == pytest.approx(0.5)
    assert ev["median_lead_quarters"] == 5  # 2017Q2 - 2016Q1


class _FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers, self.text = payload, status, headers or {}, ""

    def json(self):
        return self._p


class _FakeSession:
    headers = {}

    def get(self, url, params=None, timeout=None):
        if "history.json" in url:
            return _FakeResp({"data": {"history": [{"spl_version": 1, "published_date": "Jan 05, 2015"}, {"spl_version": 2, "published_date": "Mar 01, 2018"}]}})
        return _FakeResp({"results": [LABEL_REC]})


def test_label_client_with_mock_session():
    c = lc.LabelClient(api_key="k", session=_FakeSession(), max_per_minute=100000)
    docs = c.labels_for_generic("atorvastatin")
    assert len(docs) == 1 and docs[0].set_id == "abc-123"
    hist = c.fetch_dailymed_history("abc-123")
    assert list(hist["spl_version"]) == [1, 2]
    assert hist["published_date"].iloc[1] == pd.Timestamp("2018-03-01")
