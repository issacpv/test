"""Synthetic-data tests for faers_bias core functions (no network access)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faers_bias import disproportionality as dp  # noqa: E402
from faers_bias import denominators as dn  # noqa: E402
from faers_bias import its  # noqa: E402
from faers_bias.openfda_client import OpenFDAClient, flatten_faers_record  # noqa: E402


# ------------------------------------------------------------- disproportion
def test_ror_and_prr_known_values():
    r = dp.ror(a=50, b=950, c=100, d=8900)
    assert r["ror"] == pytest.approx((50 * 8900) / (950 * 100), rel=1e-9)
    assert r["ror_lo"] < r["ror"] < r["ror_hi"]
    p = dp.prr(a=50, b=950, c=100, d=8900)
    assert p["prr"] == pytest.approx((50 / 1000) / (100 / 9000), rel=1e-9)
    assert p["evans_signal"]
    ic = dp.information_component(a=50, b=950, c=100, d=8900)
    assert ic["ic"] > 0 and ic["ic025"] > 0


def test_zero_cell_continuity_correction():
    r = dp.ror(a=0, b=100, c=10, d=1000)
    assert np.isfinite(r["ror"]) and r["ror"] > 0


def test_sex_interaction_detects_planted_difference():
    # women: ROR ~ 4; men: ROR ~ 1
    tables = {
        "female": {"a": 200, "b": 800, "c": 500, "d": 8000},
        "male": {"a": 60, "b": 940, "c": 480, "d": 7520},
    }
    strat = dp.sex_stratified_ror(tables)
    assert strat.loc["female", "ror"] > strat.loc["male", "ror"]
    t = dp.sex_interaction_test(tables)
    assert t["ratio_of_ror"] > 2
    assert t["p_wald"] < 1e-4 and t["p_lrt"] < 1e-4
    # null: identical tables -> no interaction
    same = {"female": tables["female"], "male": tables["female"]}
    t0 = dp.sex_interaction_test(same)
    assert t0["ratio_of_ror"] == pytest.approx(1.0)
    assert t0["p_lrt"] > 0.5


def test_bias_adjusted_ror_recovers_reporter_and_dsc_effects():
    rng = np.random.default_rng(1)
    n = 40000
    drug = rng.binomial(1, 0.2, n)
    reporter = rng.choice(["physician", "consumer", "lawyer"], size=n, p=[0.4, 0.55, 0.05])
    post = rng.binomial(1, 0.5, n)
    logit = -3.0 + 0.7 * drug + 0.5 * (reporter == "consumer") + 1.2 * (reporter == "lawyer") + 0.1 * post + 0.8 * drug * post
    event = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"drug": drug, "event": event, "reporter": reporter, "post_dsc": post})
    out = dp.bias_adjusted_ror(df)
    assert out.loc["drug", "coef"] == pytest.approx(0.7, abs=0.15)
    assert out.loc["drug:post_dsc", "coef"] == pytest.approx(0.8, abs=0.2)
    assert out.loc["C(reporter)[T.lawyer]", "coef"] > out.loc["C(reporter)[T.consumer]", "coef"]


def test_benjamini_hochberg_monotone():
    p = [0.001, 0.01, 0.02, 0.5, 0.9]
    q = dp.benjamini_hochberg(p)["q"].values
    assert np.all(np.diff(q) >= 0)
    assert q[0] <= 0.005 + 1e-12


# ------------------------------------------------------------------ ITS
def _synthetic_series(step_rr: float, seed: int = 0, n_months: int = 72, event_idx: int = 48):
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    base = np.exp(3.0 + 0.01 * t + 0.15 * np.sin(2 * np.pi * t / 12))
    post = (t >= event_idx).astype(float)
    mu = base * step_rr ** post
    y = rng.negative_binomial(n=20, p=20 / (20 + mu))
    months = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    return pd.DataFrame({"month": months, "count": y}), str(months[event_idx].date())


def test_its_recovers_step_change():
    series, event = _synthetic_series(step_rr=2.0)
    d = its.build_monthly_series(series, event, pre_months=48, post_months=24)
    assert d["post"].sum() == 24
    res = its.fit_its(d, family="nb")
    assert res.level_rr == pytest.approx(2.0, rel=0.3)
    assert res.level_rr_ci[0] > 1.0
    assert res.stimulation_ratio > 1.3
    assert res.excess_reports > 0


def test_its_null_series_is_calibrated():
    series, event = _synthetic_series(step_rr=1.0, seed=3)
    d = its.build_monthly_series(series, event, pre_months=48, post_months=24)
    res = its.fit_its(d, family="poisson")
    assert res.level_rr_ci[0] < 1.0 < res.level_rr_ci[1]


def test_placebo_its_runs():
    series, event = _synthetic_series(step_rr=2.5, n_months=120, event_idx=90)
    placebo = its.placebo_its(series, event, n_placebo=10, pre_months=24, post_months=12, family="poisson")
    assert len(placebo) > 0
    d = its.build_monthly_series(series, event, pre_months=24, post_months=12)
    obs = its.fit_its(d, family="poisson")
    p = its.permutation_pvalue(np.log(obs.level_rr), placebo)
    assert 0 <= p <= 1


# ------------------------------------------------------------ denominators
def test_normalize_drug_name():
    assert dn.normalize_drug_name("Atorvastatin Calcium") == "ATORVASTATIN"
    assert dn.normalize_drug_name("METFORMIN HCL ER") == "METFORMIN"
    assert dn.normalize_drug_name("valsartan/sacubitril") == "SACUBITRIL/VALSARTAN"
    assert dn.normalize_drug_name(None) == ""


def test_partd_loader_and_rrr(tmp_path):
    csv = tmp_path / "partd.csv"
    pd.DataFrame(
        {
            "Brnd_Name": ["Lipitor", "Atorvastatin Calcium"],
            "Gnrc_Name": ["Atorvastatin Calcium", "Atorvastatin Calcium"],
            "Tot_Clms_2022": [1000, 9000],
            "Tot_Benes_2022": [300, 2700],
            "Tot_Dsg_Unts_2022": [30000, 270000],
            "Tot_Clms_2023": [1100, 9900],
            "Tot_Benes_2023": [320, 2880],
            "Tot_Dsg_Unts_2023": [33000, 297000],
        }
    ).to_csv(csv, index=False)
    long = dn.load_partd_spending_by_drug(str(csv))
    assert set(long["year"]) == {2022, 2023}
    assert long.loc[long["year"] == 2023, "total_benes"].iloc[0] == 3200

    faers = pd.DataFrame(
        {"drug_norm": ["ATORVASTATIN"] * 2, "year": [2023, 2023], "sex": ["female", "male"], "reports": [300, 100]}
    )
    denom = pd.DataFrame(
        {"drug_norm": ["ATORVASTATIN"] * 2, "year": [2023, 2023], "sex": ["female", "male"], "exposed_persons": [2e6, 2e6]}
    )
    merged = dn.merge_denominators(faers, denom)
    assert merged["rate"].iloc[0] == pytest.approx(1.5)
    rrr = dn.reporting_rate_ratio(merged)
    assert rrr["rrr"].iloc[0] == pytest.approx(3.0)
    assert rrr["crude_ratio"].iloc[0] == pytest.approx(3.0)
    # doubling female exposure halves the RRR
    denom.loc[denom["sex"] == "female", "exposed_persons"] = 4e6
    rrr2 = dn.reporting_rate_ratio(dn.merge_denominators(faers, denom))
    assert rrr2["rrr"].iloc[0] == pytest.approx(1.5)


def test_meps_aggregation():
    pmed = pd.DataFrame({"DUPERSID": ["a", "a", "b", "c"], "RXDRGNAM": ["atorvastatin", "atorvastatin", "ATORVASTATIN", "metformin"]})
    persons = pd.DataFrame({"DUPERSID": ["a", "b", "c"], "SEX": [2, 1, 2], "AGE23X": [70, 50, 30], "PERWT23F": [1000.0, 2000.0, 500.0]})
    out = dn.aggregate_meps_pmed(pmed, persons, year=2023)
    f = out[(out["drug_norm"] == "ATORVASTATIN") & (out["sex"] == "female")]
    assert f["exposed_persons"].iloc[0] == 1000.0  # person a counted once despite 2 fills
    assert f["age_band"].iloc[0] == "65+"


# ------------------------------------------------------------ client (mock)
class _FakeResponse:
    def __init__(self, payload, headers=None, status=200):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Serves two pages; the first carries a Link header with search_after."""

    def __init__(self):
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "count" in (params or {}):
            return _FakeResponse({"results": [{"term": "2", "count": 60}, {"term": "1", "count": 40}]})
        if "search_after" in url:
            return _FakeResponse({"results": [{"safetyreportid": "3"}]})
        return _FakeResponse(
            {"meta": {"results": {"total": 3}}, "results": [{"safetyreportid": "1"}, {"safetyreportid": "2"}]},
            headers={"Link": '<https://api.fda.gov/drug/event.json?search=x&limit=2&search_after=abc>; rel="next"'},
        )


def test_client_pagination_and_counts():
    sess = _FakeSession()
    client = OpenFDAClient(api_key="k", session=sess, max_per_minute=10000)
    recs = client.fetch_records("drug/event", search="x", limit=2)
    assert [r["safetyreportid"] for r in recs] == ["1", "2", "3"]
    assert any("search_after" in c[0] for c in sess.calls)
    df = client.faers_sex_counts("atorvastatin")
    assert set(df["sex"]) == {"female", "male"}
    assert int(df["count"].sum()) == 100


def test_flatten_record():
    rec = {
        "safetyreportid": "1",
        "receivedate": "20240102",
        "primarysource": {"qualification": "5"},
        "patient": {
            "patientsex": "2",
            "drug": [{"drugcharacterization": "1", "openfda": {"generic_name": ["ATORVASTATIN CALCIUM"]}}],
            "reaction": [{"reactionmeddrapt": "Myalgia"}],
        },
    }
    row = flatten_faers_record(rec)
    assert row["reporter"] == "consumer" and row["sex"] == "female"
    assert row["suspect_generic_names"] == ["ATORVASTATIN CALCIUM"]
    assert row["reactions"] == ["Myalgia"]
