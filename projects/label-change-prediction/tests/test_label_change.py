"""Synthetic-data tests for label_change (no network)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_change import dailymed_spl as spl  # noqa: E402
from label_change import faers_trajectories as ft  # noqa: E402
from label_change import landmark_model as lm  # noqa: E402
from label_change import openfda_counts as oc  # noqa: E402

SPL_V1 = """<?xml version="1.0"?>
<document xmlns="urn:hl7-org:v3">
 <component><structuredBody>
  <component><section>
   <code code="43685-7" codeSystem="2.16.840.1.113883.6.1"/>
   <title>5 WARNINGS AND PRECAUTIONS</title>
   <text><paragraph>Hepatotoxicity has been reported. Monitor liver enzymes.</paragraph></text>
  </section></component>
 </structuredBody></component>
</document>"""

SPL_V2 = SPL_V1.replace("<component><section>\n   <code code=\"43685-7\"",
                        "<component><section><code code=\"34066-1\" codeSystem=\"2.16.840.1.113883.6.1\"/>"
                        "<title>WARNING: CARDIAC FAILURE</title><text>Congestive heart failure may occur.</text>"
                        "</section></component><component><section>\n   <code code=\"43685-7\"")
SPL_V2 = SPL_V2.replace("Monitor liver enzymes.", "Monitor liver enzymes. Pancreatitis has been observed.")

VOCAB = ["Hepatotoxicity", "Pancreatitis", "Congestive heart failure", "Rash"]


def test_extract_sections_and_new_terms():
    s1 = spl.extract_sections(SPL_V1)
    s2 = spl.extract_sections(SPL_V2)
    assert "warnings_and_precautions" in s1 and "boxed_warning" not in s1
    assert "boxed_warning" in s2
    added = spl.new_terms(s1, s2, VOCAB)
    assert added["boxed_warning"] == {"Congestive heart failure"}
    assert added["warnings_and_precautions"] == {"Pancreatitis"}
    d = spl.diff_sections(s1, s2)
    assert any("Pancreatitis" in s for s in d["warnings_and_precautions"]["added"])


def test_parse_srlc_export_tolerant_columns():
    csv = io.StringIO(
        "Drug Name,Application Number,Section,Approval Date,Summary of Changes\n"
        "DrugX,NDA012345,BOXED WARNING,2019-03-12,Added warning for congestive heart failure\n"
        "DrugX,NDA012345,WARNINGS AND PRECAUTIONS,2020-11-02,Pancreatitis added\n"
    )
    df = spl.parse_srlc_export(csv, vocabulary=VOCAB)
    assert list(df["section"]) == ["boxed_warning", "warnings_and_precautions"]
    assert df["quarter"].tolist() == ["2019Q1", "2020Q4"]
    assert df["terms"].iloc[0] == {"Congestive heart failure"}


def test_quarter_index_roundtrip_and_ic_values():
    assert ft.quarter_index("2004Q1") == 0
    assert ft.quarter_index("2019Q3") == 62
    assert ft.index_to_quarter(62) == "2019Q3"
    ic = ft.bcpnn_ic(np.array([50.0]), np.array([950.0]), np.array([100.0]), np.array([8900.0]))
    e = 1000 * 150 / 10000
    assert ic["ic"][0] == pytest.approx(np.log2(50.5 / (e + 0.5)))
    assert ic["ic025"][0] < ic["ic"][0] < ic["ic975"][0]
    r = ft.ror(np.array([50.0]), np.array([950.0]), np.array([100.0]), np.array([8900.0]), cc=0.0)
    assert r["ror"][0] == pytest.approx(50 * 8900 / (950 * 100))


def _pair_table(rng, n_q=40, rate_pair=0.5, boost_from=None, boost=6.0):
    qs = np.arange(n_q)
    drug = rng.poisson(200, n_q)
    ptot = rng.poisson(100, n_q)  # E[a] = 200*100/20000 = 1 per quarter
    n = rng.poisson(20000, n_q)
    lam = np.full(n_q, rate_pair)
    if boost_from is not None:
        lam[boost_from:] *= boost
    a = rng.poisson(lam)
    return ft.cumulative_2x2(pd.Series(a, qs), pd.Series(drug, qs), pd.Series(ptot, qs), pd.Series(n, qs), qs)


def test_cumulative_2x2_and_landmark_features_respect_cutoff():
    rng = np.random.default_rng(0)
    tab = _pair_table(rng, boost_from=20)
    assert (tab["a"].diff().dropna() >= 0).all()
    f_early = ft.landmark_features(tab, 15)
    f_late = ft.landmark_features(tab, 35)
    assert f_late["ic025"] > f_early["ic025"]
    assert f_early["n_quarters_obs"] == 16
    # the early landmark must not see the later boost
    tab_trunc = tab.loc[tab.index <= 15]
    assert ft.landmark_features(tab_trunc, 15) == f_early


def test_notoriety_ratio_detects_post_event_rise():
    q = np.arange(30)
    pair = pd.Series(np.where(q > 15, 20, 2), q)
    drug = pd.Series(200, q)
    r = ft.notoriety_ratio(pair, drug, event_quarter=15, window=4)
    assert r["ratio"] > 5


def test_landmark_dataset_and_model_recover_signal():
    rng = np.random.default_rng(1)
    tables, events, meta = {}, {}, {}
    n_pairs = 120
    for i in range(n_pairs):
        pid = f"p{i}"
        has_event = i < 40
        boost_from = int(rng.integers(8, 25)) if has_event else None
        tables[pid] = _pair_table(rng, n_q=40, boost_from=boost_from, boost=20.0)
        events[pid] = float(boost_from + 6) if has_event else np.nan
        meta[pid] = {"drug": f"d{i % 15}", "class_warning": 0}
    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    ds = lm.build_landmark_dataset(tables, pd.Series(events), landmarks=range(4, 36, 2), horizon=8,
                                   last_quarter=39, min_reports=1, pair_meta=meta_df)
    assert set(lm.FEATURE_COLS) <= set(ds.columns)
    # no row after the pair's own event
    ev = ds[ds["time_to_event"].notna()]
    assert (ev["time_to_event"] > 0).all()
    assert ds["y"].sum() > 20
    model = lm.fit_pooled_logistic(ds)
    auc = lm.time_dependent_auc(ds, lm.predict_risk(model, ds))["auc"]
    assert auc > 0.7
    cv = lm.grouped_cv_auc(ds, n_splits=3)
    assert cv["auc_pooled"] > 0.65
    lt = lm.lead_times(tables, pd.Series(events))
    assert len(lt) == 40
    assert lt["lead_quarters"].median() > 0
    null = lm.permutation_null_auc(ds, n_perm=3)
    assert null.mean() < auc


def test_openfda_search_clause_and_quarterly():
    s = oc.search_clause(drug="rosiglitazone", pt="Cardiac failure congestive", serious=True, hcp_only=True)
    assert 'generic_name:ROSIGLITAZONE' in s and 'reactionmeddrapt:"Cardiac failure congestive"' in s
    assert "serious:1" in s and "qualification:(1+2+3)" in s
    daily = pd.DataFrame({"date": pd.to_datetime(["2019-01-03", "2019-02-20", "2019-04-01"]), "count": [1, 2, 3]})
    q = oc.daily_to_quarterly(daily)
    assert q[pd.Period("2019Q1")] == 3 and q[pd.Period("2019Q2")] == 3
