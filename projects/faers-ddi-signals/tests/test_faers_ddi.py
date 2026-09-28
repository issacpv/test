"""Synthetic-data tests for faers_ddi (no network access needed)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from faers_ddi.ddi_stats import additive_excess, benjamini_hochberg, hierarchical_shrinkage, interaction_ror, omega, screen, sex_specific_interaction
from faers_ddi.ddi_tables import candidate_pairs, index_reports, pair_event_table, sex_stratified_tables, three_way_counts
from faers_ddi.openfda import OpenFDAClient, age_in_years, flatten_record
from faers_ddi.reference_sets import evaluate_scores, interacting_role_pairs, label_pairs, make_pair, roc_auc


def simulate_reports(n: int = 6000, ior_female: float = 6.0, ior_male: float = 1.0, seed: int = 1) -> List[Dict[str, Any]]:
    """Reports with drugs A, B (independent, p=0.3) and event E whose odds are
    multiplied by an A x B interaction that exists only in women."""
    rng = np.random.default_rng(seed)
    reports = []
    for i in range(n):
        sex = "female" if rng.random() < 0.5 else "male"
        a, b = rng.random() < 0.3, rng.random() < 0.3
        logit = -2.5 + 0.4 * a + 0.4 * b
        if a and b:
            logit += np.log(ior_female if sex == "female" else ior_male)
        p = 1 / (1 + np.exp(-logit))
        reactions = ["E"] if rng.random() < p else []
        reactions += ["NAUSEA"] if rng.random() < 0.2 else []
        drugs = [d for d, flag in (("A", a), ("B", b)) if flag] + ["Z"]
        reports.append({"safetyreportid": str(i), "sex": sex, "drugs": drugs, "suspect": drugs[:1], "interacting": [], "reactions": reactions})
    return reports


def test_three_way_counts_sum_to_total():
    reports = simulate_reports(500)
    drug_idx, event_idx, n = index_reports(reports)
    c = three_way_counts(drug_idx["A"], drug_idx["B"], event_idx["E"], n)
    assert sum(c.values()) == n
    assert c["n111"] + c["n110"] == len(drug_idx["A"] & drug_idx["B"])


def test_candidate_pairs_and_table():
    reports = simulate_reports(800)
    drug_idx, _, _ = index_reports(reports)
    pairs = candidate_pairs(drug_idx, min_coreports=10)
    assert ("A", "B") in [(a, b) for a, b, _ in pairs]
    tab = pair_event_table(reports, [("A", "B")], min_n111=1)
    assert {"n111", "n000", "event"}.issubset(tab.columns)
    assert (tab[["n111", "n110", "n101", "n100", "n011", "n010", "n001", "n000"]].sum(axis=1) == len(reports)).all()


def test_interaction_measures_recover_planted_effect():
    reports = simulate_reports(8000, ior_female=6.0, ior_male=6.0)
    tab = pair_event_table(reports, [("A", "B")], min_n111=1)
    row = tab[tab["event"] == "E"].iloc[0]
    c = {k: float(row[k]) for k in ("n111", "n110", "n101", "n100", "n011", "n010", "n001", "n000")}
    r = interaction_ror(c)
    assert 3.0 < r["ior"] < 12.0
    assert r["ior_lo"] > 1.0 and r["p_lrt"] < 1e-3
    o = omega(c)
    assert o["omega025"] > 0
    for model in ("additive", "multiplicative", "max"):
        assert omega(c, model)["omega"] > 0
    assert additive_excess(c)["reri"] > 0
    # null pair: A with the unrelated drug Z should show no interaction
    tab0 = pair_event_table(reports, [("A", "Z")], min_n111=1)
    row0 = tab0[tab0["event"] == "NAUSEA"].iloc[0]
    c0 = {k: float(row0[k]) for k in c}
    assert interaction_ror(c0)["p_lrt"] > 0.01


def test_sex_specific_interaction_detected():
    reports = simulate_reports(12000, ior_female=6.0, ior_male=1.0)
    st = sex_stratified_tables(reports, [("A", "B")], min_n111=1)
    row = st[st["event"] == "E"].iloc[0]
    keys = ("n111", "n110", "n101", "n100", "n011", "n010", "n001", "n000")
    cf = {k: float(row[f"{k}_f"]) for k in keys}
    cm = {k: float(row[f"{k}_m"]) for k in keys}
    res = sex_specific_interaction(cf, cm)
    assert res["ior_female"] > res["ior_male"]
    assert res["ratio_of_ior"] > 2.0 and res["p_lrt"] < 0.01


def test_hierarchical_shrinkage_pulls_towards_group_mean():
    df = pd.DataFrame(
        {
            "drug_a": ["X"] * 6,
            "drug_b": list("abcdef"),
            "log_ior": [0.2, 1.0, 0.4, 0.9, 0.1, 3.0],
            "se_log_ior": [0.15, 0.15, 0.15, 0.15, 0.15, 1.5],
            "mechanism": ["cyp3a4"] * 6,
        }
    )
    out = hierarchical_shrinkage(df)
    # heterogeneous precise estimates -> tau2 > 0; the noisy outlier is shrunk
    # far more than the precise estimates and lands near the group mean
    assert out.loc[0, "group_tau2"] > 0
    assert out.loc[5, "shrink_b"] > 0.8 > 0.3 > out.loc[0, "shrink_b"]
    assert out.loc[5, "post_mean"] < 1.0
    mu = out.loc[0, "group_mu"]
    assert 0.2 <= out.loc[0, "post_mean"] <= mu
    # a single-pair group is returned unshrunk
    single = hierarchical_shrinkage(df.iloc[:1].assign(mechanism="solo"))
    assert single.loc[0, "post_mean"] == 0.2 and single.loc[0, "shrink_b"] == 0.0


def test_screen_and_bh():
    reports = simulate_reports(3000)
    tab = pair_event_table(reports, [("A", "B"), ("A", "Z")], min_n111=1)
    out = screen(tab)
    assert {"omega", "ior", "reri", "q_ior", "omega_signal"}.issubset(out.columns)
    bh = benjamini_hochberg([0.001, 0.01, 0.2, 0.5])
    assert bh["reject"].tolist() == [True, True, False, False]


def test_reference_sets_and_auc():
    reports = simulate_reports(200)
    reports[0]["interacting"] = ["B"]
    reports[0]["suspect"] = ["A"]
    reports[1]["interacting"] = ["B"]
    reports[1]["suspect"] = ["A"]
    pairs = interacting_role_pairs(reports, min_reports=2)
    assert pairs[make_pair("A", "B")] == 2
    scores = pd.DataFrame({"drug_a": ["A", "A", "C"], "drug_b": ["B", "Z", "D"], "omega": [2.0, 0.1, -0.5]})
    lab = label_pairs(scores, positives=[make_pair("A", "B")], negatives=[make_pair("A", "Z"), make_pair("C", "D")])
    assert lab["label"].tolist() == [1.0, 0.0, 0.0]
    assert roc_auc(lab["omega"], lab["label"]) == 1.0
    ev = evaluate_scores(lab, ["omega"], n_boot=20, k=2)
    assert ev.loc[0, "auroc"] == 1.0


# ------------------------------------------------------------- client tests
class _Resp:
    def __init__(self, payload: Dict[str, Any], status: int = 200, link: str = ""):
        self._payload, self.status_code, self.headers, self.text = payload, status, {"Link": link} if link else {}, json.dumps(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeSession:
    """Emulates openFDA pagination: page 1 carries a Link header, page 2 does not."""

    def __init__(self):
        self.calls: List[str] = []

    def get(self, url: str, params=None, timeout=None):
        self.calls.append(url)
        if "search_after" in url:
            return _Resp({"meta": {"results": {"total": 3}}, "results": [{"safetyreportid": "3", "patient": {}}]})
        if params and params.get("count"):
            return _Resp({"results": [{"term": "NAUSEA", "count": 5}]})
        return _Resp(
            {"meta": {"results": {"total": 3}}, "results": [{"safetyreportid": "1", "patient": {}}, {"safetyreportid": "2", "patient": {}}]},
            link='<https://api.fda.gov/drug/event.json?search=x&limit=2&search_after=abc>; rel="next"',
        )


def test_client_follows_link_header_and_counts():
    sess = _FakeSession()
    client = OpenFDAClient(api_key="k", session=sess, max_per_minute=10000)
    recs = list(client.iter_records("drug/event", "x", limit=2))
    assert [r["safetyreportid"] for r in recs] == ["1", "2", "3"]
    assert any("search_after" in u for u in sess.calls)
    assert client.count("drug/event", "x", "patient.reaction.reactionmeddrapt")[0]["count"] == 5
    assert client.total("drug/event", "x") == 3
    assert list(client.iter_records("drug/event", "x", max_records=1)) and len(list(client.iter_records("drug/event", "x", max_records=1))) == 1


def test_flatten_record_roles_and_age():
    rec = {
        "safetyreportid": "9",
        "patient": {
            "patientsex": "2",
            "patientonsetage": "6",
            "patientonsetageunit": "802",
            "drug": [
                {"drugcharacterization": "1", "openfda": {"generic_name": ["SIMVASTATIN"]}},
                {"drugcharacterization": "3", "medicinalproduct": "Clarithromycin"},
            ],
            "reaction": [{"reactionmeddrapt": "Rhabdomyolysis"}],
        },
    }
    row = flatten_record(rec)
    assert row["sex"] == "female" and row["suspect"] == ["SIMVASTATIN"] and row["interacting"] == ["CLARITHROMYCIN"]
    assert abs(row["age_years"] - 0.5) < 1e-6 and row["reactions"] == ["Rhabdomyolysis"]
    assert age_in_years("70", "801") == 70.0 and age_in_years("x", "801") is None
