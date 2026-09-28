"""Synthetic-data tests for shortage_ae (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortage_ae import openfda_client as oc  # noqa: E402
from shortage_ae import shortage_panel as sp  # noqa: E402
from shortage_ae import staggered_did as sd  # noqa: E402


# ------------------------------------------------------------------ names / episodes
def test_normalize_name_strips_salts_forms_strengths():
    assert sp.normalize_name("Heparin Sodium Injection, USP 5,000 units/mL") == "heparin"
    assert sp.normalize_name("Semaglutide (Ozempic) 0.25 mg/0.5 mL") == "semaglutide"
    assert sp.normalize_name("Amoxicillin and Clavulanate Potassium Tablets") == "amoxicillin"
    assert sp.normalize_name("Norepinephrine Bitartrate in Dextrose 8 mg/250 mL premixed bag") == "norepinephrine"
    assert sp.normalize_name(None) == ""


def test_episodes_from_openfda_collapses_presentations():
    df = pd.DataFrame({
        "generic_name": ["Heparin Sodium Injection", "Heparin Sodium Injection", "Lorazepam Injection"],
        "presentation": ["5,000 units/mL vial", "1,000 units/mL vial", "2 mg/mL vial"],
        "status": ["Resolved", "Resolved", "Current"],
        "initial_posting_date": ["2017-10-15", "2017-11-02", "2023-01-05"],
        "update_date": ["2018-06-01", "2018-08-01", "2024-01-01"],
        "shortage_reason": ["Hurricane", "Hurricane", None],
        "therapeutic_category": ["Hematology", "Hematology", "Anesthesia"],
    })
    ep = sp.episodes_from_openfda(df).set_index("drug")
    assert ep.loc["heparin", "onset"] == pd.Timestamp("2017-10-01")
    assert ep.loc["heparin", "resolution"] == pd.Timestamp("2018-08-01")
    assert ep.loc["heparin", "n_presentations"] == 2
    assert bool(ep.loc["heparin", "injectable"])
    assert pd.isna(ep.loc["lorazepam", "resolution"])


# ------------------------------------------------------------------ client (offline)
class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        pass


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if "count=receivedate" in url:
            return _FakeResp({"results": [{"time": "20200105", "count": 3}, {"time": "20200210", "count": 5}]})
        if "drug/shortages" in url:
            return _FakeResp({"meta": {"results": {"total": 2}},
                              "results": [{"generic_name": "A", "status": "Current"},
                                          {"generic_name": "B", "status": "Resolved"}]})
        return _FakeResp({}, status=404)


def test_build_search_and_count_by_date():
    s = oc.build_faers_search("heparin sodium", reactions=["Wrong drug administered", "Overdose"], serious=True)
    assert 'patient.drug.openfda.generic_name:"HEPARIN SODIUM"' in s
    assert 'reactionmeddrapt:("Wrong drug administered"+Overdose)' in s
    assert "serious:1" in s and "drugcharacterization:1" in s
    client = oc.OpenFDAClient(api_key="k", session=_FakeSession(), min_interval=0.0)
    df = client.count_by_date("drug/event", s, "receivedate", pd.Timestamp("2020-01-01").date(),
                              pd.Timestamp("2020-03-01").date())
    assert list(df["count"]) == [3, 5]
    assert "api_key=k" in client.session.calls[0]
    recs = client.shortage_records()
    assert [r["generic_name"] for r in recs] == ["A", "B"]
    assert client.get("drug/event", {"search": "x"})["results"] == []  # 404 -> empty


def test_flatten_shortage_record_handles_missing_openfda():
    row = oc.flatten_shortage_record({"generic_name": "X", "status": "Current"})
    assert row["route"] == "" and row["pharm_class_epc"] == ""


# ------------------------------------------------------------------ panel + estimators
def _synthetic_panel(n_drugs=60, n_months=72, n_treated=30, effect=1.5, seed=3):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    drugs = [f"d{i}" for i in range(n_drugs)]
    onsets = rng.integers(n_months // 4, (2 * n_months) // 3, size=n_treated)
    episodes = pd.DataFrame({"drug": drugs[:n_treated], "onset": months[onsets], "resolution": pd.NaT})
    counts = {}
    month_fx = np.exp(rng.normal(0, 0.15, n_months))
    for i, d in enumerate(drugs):
        base_all = rng.uniform(150, 400)
        share = rng.uniform(0.03, 0.08)
        n_all = rng.poisson(base_all * month_fx)
        lam = n_all * share
        if i < n_treated:
            post = np.arange(n_months) >= onsets[i]
            lam = lam * np.where(post, effect, 1.0)
        n_err = rng.poisson(lam)
        counts[d] = {"all": pd.Series(n_all, index=months), "error": pd.Series(n_err, index=months)}
    return episodes, counts, months


def test_build_panel_structure():
    episodes, counts, months = _synthetic_panel(n_drugs=6, n_months=24, n_treated=3)
    panel = sp.build_panel(episodes, counts, months)
    assert set(["drug", "month", "t", "n_all", "n_error", "treated", "cohort", "rel_time"]) <= set(panel.columns)
    assert len(panel) == 6 * 24
    tr = panel[panel["ever_treated"] == 1]
    assert (tr.loc[tr["rel_time"] >= 0, "treated"] == 1).all()
    assert (tr.loc[tr["rel_time"] < 0, "treated"] == 0).all()
    assert panel.loc[panel["ever_treated"] == 0, "rel_time"].isna().all()


def test_event_study_and_cs_recover_planted_effect():
    episodes, counts, months = _synthetic_panel()
    panel = sp.build_panel(episodes, counts, months)
    tab, res = sd.event_study_poisson(panel, pre=6, post=6)
    post = tab[(tab["rel_time"] >= 0) & (tab["rel_time"] < 6)]["coef"].mean()
    pre = tab[(tab["rel_time"] <= -2)]["coef"].mean()
    assert post == pytest.approx(np.log(1.5), abs=0.12)
    assert abs(pre) < 0.1
    w = sd.pretrend_wald(res, pre=6)
    assert w["p"] > 0.01
    cs = sd.callaway_santanna_att(panel, horizons=range(0, 7), n_boot=30)
    assert cs["att"].mean() == pytest.approx(np.log(1.5), abs=0.15)
    agg = sd.aggregate_att(cs, range(0, 7))
    assert 1.3 < agg["irr"] < 1.75


def test_placebo_permutation_null_centred():
    episodes, counts, months = _synthetic_panel(n_drugs=30, n_months=48, n_treated=12, effect=1.0)

    panel = sp.build_panel(episodes, counts, months)

    def stat(p):
        cs = sd.callaway_santanna_att(p, horizons=range(0, 4), n_boot=0)
        return cs["att"].mean()

    out = sd.placebo_permutation(panel, stat, n_perm=20, seed=1)
    assert abs(out["observed"]) < 0.15
    assert out["p_value"] > 0.05
    assert len(out["null"]) == 20


def test_find_substitutes_same_class_and_route():
    products = pd.DataFrame({
        "drug": ["heparin", "enoxaparin", "dalteparin", "warfarin", "enoxaparin"],
        "epc": ["Anticoagulant", "Anticoagulant", "Anticoagulant", "Anticoagulant", "Anticoagulant"],
        "route": ["INTRAVENOUS", "SUBCUTANEOUS", "SUBCUTANEOUS", "ORAL", "INTRAVENOUS"],
    })
    assert sp.find_substitutes("heparin", products) == ["enoxaparin"]
    assert sp.find_substitutes("heparin", products, same_route=False) == ["dalteparin", "enoxaparin", "warfarin"]
    assert sp.find_substitutes("nothing", products) == []
