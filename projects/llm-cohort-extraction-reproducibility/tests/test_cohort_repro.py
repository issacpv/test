"""Tests for cohort_repro on a synthetic mini-MIMIC (no real data, no model, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cohort_repro import cohort_spec as cs  # noqa: E402
from cohort_repro import compiler, evaluation, llm_client  # noqa: E402


# ------------------------------------------------------------ fixtures
@pytest.fixture
def mini_mimic():
    """Six ICU stays, four subjects, MIMIC-IV column conventions."""
    patients = pd.DataFrame(
        {"subject_id": [1, 2, 3, 4], "gender": ["M", "F", "M", "F"], "anchor_age": [70, 17, 45, 91], "anchor_year": [2150, 2150, 2150, 2150], "dod": [pd.Timestamp("2150-03-20"), pd.NaT, pd.NaT, pd.Timestamp("2151-01-01")]}
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4],
            "hadm_id": [10, 11, 20, 30, 40],
            "admittime": pd.to_datetime(["2150-03-01", "2150-03-15", "2150-04-01", "2150-05-01", "2150-06-01"]),
            "dischtime": pd.to_datetime(["2150-03-05", "2150-03-20", "2150-04-03", "2150-05-02", "2150-06-10"]),
            "deathtime": [pd.NaT, pd.Timestamp("2150-03-20"), pd.NaT, pd.NaT, pd.NaT],
            "hospital_expire_flag": [0, 1, 0, 0, 0],
            "admission_type": ["EW EMER.", "URGENT", "ELECTIVE", "EW EMER.", "URGENT"],
        }
    )
    icustays = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 3, 4],
            "hadm_id": [10, 11, 11, 20, 30, 40],
            "stay_id": [100, 110, 111, 200, 300, 400],
            "first_careunit": ["MICU", "MICU", "SICU", "MICU", "CCU", "MICU"],
            "intime": pd.to_datetime(["2150-03-01 10:00", "2150-03-15 08:00", "2150-03-18 08:00", "2150-04-01 12:00", "2150-05-01 01:00", "2150-06-01 00:00"]),
            "outtime": pd.to_datetime(["2150-03-03 10:00", "2150-03-17 08:00", "2150-03-20 08:00", "2150-04-01 18:00", "2150-05-02 01:00", "2150-06-05 00:00"]),
            "los": [2.0, 2.0, 2.0, 0.25, 1.0, 4.0],
        }
    )
    diagnoses = pd.DataFrame({"hadm_id": [10, 11, 30, 40], "icd_code": ["A419", "I50.9", "J44", "E11"], "icd_version": [10, 10, 10, 10]})
    labs = pd.DataFrame({"hadm_id": [10, 11, 40], "itemid": [50912, 50912, 50983], "charttime": pd.to_datetime(["2150-03-01 12:00", "2150-03-15 20:00", "2150-06-01 05:00"]), "valuenum": [1.1, 2.3, 140.0]})
    return patients, admissions, icustays, diagnoses, labs


def _defn(criteria, outcome="in_hospital_mortality", unit="icustay"):
    return cs.CohortDefinition(paper_id="p1", database="mimic-iv", unit=unit, criteria=[cs.Criterion(**c) for c in criteria], outcome=cs.Outcome(kind=outcome))


# ------------------------------------------------------------ llm_client
def test_local_url_guard(monkeypatch):
    assert llm_client.is_local_url("http://localhost:8000/v1") and llm_client.is_local_url("http://127.0.0.1:11434/v1")
    assert llm_client.is_local_url("http://192.168.1.20:8000/v1") and not llm_client.is_local_url("https://api.example.com/v1")
    monkeypatch.delenv("COHORT_REPRO_ALLOW_REMOTE", raising=False)
    with pytest.raises(llm_client.RemoteEndpointError):
        llm_client.OpenAICompatibleClient(base_url="https://api.example.com/v1", model="x")
    with pytest.raises(llm_client.RemoteEndpointError):
        llm_client.OpenAICompatibleClient(base_url="https://api.example.com/v1", model="x", allow_remote=True)  # env not set
    c = llm_client.OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="m")
    assert c.base_url == "http://localhost:8000/v1" and llm_client.default_client().base_url.startswith("http://localhost")


def test_openai_compatible_client_posts_to_local_server():
    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class Sess:
        def __init__(self):
            self.payloads = []

        def post(self, url, json=None, headers=None, timeout=None):
            self.payloads.append((url, json))
            return Resp()

    sess = Sess()
    c = llm_client.OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="m", session=sess)
    out = c.complete("sys", "usr", json_mode=True, seed=3)
    assert out == '{"ok": true}' and sess.payloads[0][0].endswith("/chat/completions")
    assert sess.payloads[0][1]["seed"] == 3 and sess.payloads[0][1]["response_format"]["type"] == "json_object"


def test_extract_json_and_chat_json_repair():
    assert llm_client.extract_json('Sure:\n```json\n{"a": 1}\n```')["a"] == 1
    assert llm_client.extract_json('prefix {"a": {"b": [1, 2]}} suffix')["a"]["b"] == [1, 2]
    with pytest.raises(ValueError):
        llm_client.extract_json("no json here")
    client = llm_client.DryRunClient(["not json", '{"database": "nope"}', '{"database": "mimic-iv", "unit": "icustay", "criteria": []}'])
    obj = llm_client.chat_json(client, "s", "u", validator=cs.validate_definition_dict, max_attempts=3)
    assert obj["database"] == "mimic-iv" and len(client.calls) == 3 and "problems" in client.calls[2]["user"]


# ----------------------------------------------------------- cohort_spec
def test_extraction_prompt_parse_and_multiverse():
    prompt = cs.build_extraction_prompt("Adults (>=18) with a first ICU stay of at least 24 h were included.", "MIMIC-IV")
    assert "first_icu_stay_only" in prompt and "Paper text" in prompt
    raw = {
        "database": "mimic-iv",
        "unit": "icustay",
        "criteria": [
            {"kind": "age_min", "params": {"years": 18}, "source": "Adults (>=18)"},
            {"kind": "first_icu_stay_only", "params": {"per": "subject"}, "source": "first ICU stay", "ambiguous": True, "alternatives": [{"per": "hadm"}]},
            {"kind": "min_icu_los_hours", "params": {"hours": 24}, "source": "at least 24 h", "ambiguous": True, "alternatives": [{"hours": 24}, {"hours": 48}]},
        ],
        "outcome": {"kind": "in_hospital_mortality", "source": "died in hospital"},
        "reported_n": 1234,
    }
    assert cs.validate_definition_dict(raw) == []
    d = cs.parse_extraction(raw, "p1", model="test")
    assert d.reported_n == 1234 and len(d.criteria) == 3 and d.extraction_model == "test"
    rt = cs.CohortDefinition.from_dict(json.loads(d.to_json()))
    assert rt.canonical_hash() == d.canonical_hash()
    mv = cs.enumerate_multiverse(d)
    assert len(mv) == 4 and mv[0].canonical_hash() == d.canonical_hash() and len({m.canonical_hash() for m in mv}) == 4
    bad = dict(raw, criteria=[{"kind": "magic", "params": {}}, {"kind": "age_min", "params": {}}])
    probs = cs.validate_definition_dict(bad)
    assert any("unknown kind" in p for p in probs) and any("missing param" in p for p in probs)
    with pytest.raises(ValueError):
        cs.parse_extraction(bad, "p1")


# --------------------------------------------------------------- compiler
def test_pandas_executor_applies_criteria(mini_mimic):
    patients, admissions, icustays, diagnoses, labs = mini_mimic
    ex = compiler.PandasExecutor("mimic-iv", patients, admissions, icustays, diagnoses, labs)
    base = ex.run(_defn([]))
    assert len(base) == 6 and base["outcome"].sum() == 2  # stays 110 and 111 belong to the fatal admission
    adults = ex.run(_defn([{"kind": "age_min", "params": {"years": 18}}]))
    assert set(adults["subject_id"]) == {1, 3, 4}
    first = ex.run(_defn([{"kind": "first_icu_stay_only", "params": {"per": "subject"}}]))
    assert set(first["stay_id"]) == {100, 200, 300, 400}
    first_hadm = ex.run(_defn([{"kind": "first_icu_stay_only", "params": {"per": "hadm"}}]))
    assert set(first_hadm["stay_id"]) == {100, 110, 200, 300, 400}
    los = ex.run(_defn([{"kind": "min_icu_los_hours", "params": {"hours": 24}}]))
    assert 200 not in set(los["stay_id"]) and len(los) == 5
    icd = ex.run(_defn([{"kind": "require_icd_prefix", "params": {"prefixes": ["I50"], "icd_version": 10}}]))
    assert set(icd["hadm_id"]) == {11}
    excl = ex.run(_defn([{"kind": "exclude_icd_prefix", "params": {"prefixes": ["I50", "J44"]}}]))
    assert set(excl["hadm_id"]) == {10, 20, 40}
    lab = ex.run(_defn([{"kind": "require_lab_measured", "params": {"itemids": [50912], "window_hours": 24}}]))
    assert set(lab["stay_id"]) == {100, 110}  # 111's lab is before its intime
    early = ex.run(_defn([{"kind": "exclude_death_within_hours", "params": {"hours": 72}}]))
    assert 111 not in set(early["stay_id"]) and 110 in set(early["stay_id"])
    unit = ex.run(_defn([{"kind": "care_unit_in", "params": {"units": ["MICU"]}}]))
    assert set(unit["first_careunit"] if "first_careunit" in unit else unit["stay_id"]) == {100, 110, 200, 400}
    hadm_unit = ex.run(_defn([], unit="hadm"))
    assert len(hadm_unit) == 5 and hadm_unit["hadm_id"].is_unique
    m30 = ex.run(_defn([], outcome="mortality_30d"))
    assert m30.set_index("stay_id").loc[100, "outcome"] == 1 and m30.set_index("stay_id").loc[400, "outcome"] == 0
    readm = ex.run(_defn([], outcome="readmission_30d"))
    assert readm.set_index("stay_id").loc[100, "outcome"] == 1 and readm.set_index("stay_id").loc[200, "outcome"] == 0


def test_compile_sql_contains_expected_clauses():
    d = _defn(
        [
            {"kind": "age_min", "params": {"years": 18}},
            {"kind": "first_icu_stay_only", "params": {"per": "hadm"}},
            {"kind": "require_icd_prefix", "params": {"prefixes": ["I50.9"], "icd_version": 10}},
            {"kind": "exclude_death_within_hours", "params": {"hours": 6, "from": "hosp_admittime"}},
        ]
    )
    sql = compiler.compile_sql(d)
    assert "age >= 18.0" in sql and "stay_rank_hadm = 1" in sql and "LIKE 'I509%'" in sql and "icd_version = 10" in sql
    assert "date_diff('hour', admittime, deathtime) < 6.0" in sql and "hospital_expire_flag AS INTEGER) AS outcome" in sql
    sql3 = compiler.compile_sql(cs.CohortDefinition(paper_id="p", database="mimic-iii", unit="subject", criteria=[cs.Criterion("age_min", {"years": 16})], outcome=cs.Outcome("icu_los_gt_hours", hours=72)))
    assert "icustay_id" in sql3 and "91.4" in sql3 and "QUALIFY" in sql3 and "icu_los_hours > 72.0" in sql3
    with pytest.raises(ValueError):
        compiler.compile_sql(_defn([{"kind": "age_min", "params": {"years": 18}}], outcome="in_hospital_mortality").__class__(paper_id="x", database="mimic-iv", unit="icustay", criteria=[cs.Criterion("nope", {})]))


def test_sql_matches_pandas_on_duckdb(mini_mimic):
    duckdb = pytest.importorskip("duckdb")
    patients, admissions, icustays, diagnoses, labs = mini_mimic
    con = duckdb.connect()
    for name, df in (("patients", patients), ("admissions", admissions), ("icustays", icustays), ("diagnoses_icd", diagnoses), ("labevents", labs)):
        con.register(name, df)
    ex = compiler.PandasExecutor("mimic-iv", patients, admissions, icustays, diagnoses, labs)
    cases = [
        _defn([{"kind": "age_min", "params": {"years": 18}}, {"kind": "first_icu_stay_only", "params": {}}, {"kind": "min_icu_los_hours", "params": {"hours": 12}}]),
        _defn([{"kind": "require_icd_prefix", "params": {"prefixes": ["I50.9", "J44"], "icd_version": 10}}, {"kind": "exclude_death_within_hours", "params": {"hours": 72}}], outcome="mortality_30d"),
        _defn([{"kind": "require_lab_measured", "params": {"itemids": [50912], "window_hours": 24}}], outcome="icu_mortality"),
        _defn([{"kind": "care_unit_in", "params": {"units": ["MICU"]}}], outcome="readmission_30d", unit="hadm"),
        _defn([{"kind": "first_icu_stay_only", "params": {"per": "hadm"}}], outcome="icu_los_gt_hours", unit="subject"),
    ]
    cases[-1].outcome.hours = 30
    for d in cases:
        via_sql = compiler.run_sql(compiler.compile_sql(d), con)
        via_pd = ex.run(d)
        assert set(via_sql["stay_id"]) == set(via_pd["stay_id"]), d.criteria
        assert int(via_sql["outcome"].sum()) == int(via_pd["outcome"].sum()), d.outcome
        assert len(via_sql) == len(via_pd)


# ------------------------------------------------------------- evaluation
def test_evaluation_metrics():
    e = evaluation.count_error(1100, 1000)
    assert e["within_tol"] and e["rel_error"] == pytest.approx(0.1)
    assert not evaluation.count_error(500, 1000)["within_tol"]
    gold = _defn([{"kind": "age_min", "params": {"years": 18}}, {"kind": "first_icu_stay_only", "params": {"per": "subject"}}])
    pred = _defn([{"kind": "age_min", "params": {"years": 18}}, {"kind": "first_icu_stay_only", "params": {"per": "hadm"}}, {"kind": "min_icu_los_hours", "params": {"hours": 24}}])
    agr = evaluation.criterion_agreement(pred, gold)
    assert agr["exact_precision"] == pytest.approx(1 / 3) and agr["exact_recall"] == pytest.approx(0.5)
    assert agr["kind_recall"] == 1.0 and agr["outcome_match"] == 1.0
    sc = evaluation.self_consistency([gold, gold, pred])
    assert sc["identical_frac"] == pytest.approx(2 / 3) and 0 < sc["mean_jaccard"] < 1
    mv = evaluation.multiverse_summary([900, 1200, 1500], n_reported=1000)
    assert mv["reported_in_range"] and mv["n_variants"] == 3
    rows = pd.DataFrame({"model": ["a", "a", "b"], "paper_id": ["p1", "p2", "p1"], "within_tol": [True, False, True], "abs_rel_error": [0.1, 0.4, 0.05], "executed": [True, True, True], "exact_f1": [0.8, 0.5, 0.9]})
    summ = evaluation.summarize_benchmark(rows)
    assert set(summ["model"]) == {"a", "b"} and summ.set_index("model").loc["a", "within_tol_rate"] == 0.5
    assert np.isnan(evaluation.prevalence_error(0.1, None)["prev_abs_diff_pp"])
