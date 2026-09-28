"""Synthetic-data tests for xspecies_pv (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xspecies_pv import cross_species_signals as cs  # noqa: E402
from xspecies_pv import openfda_animal as oa  # noqa: E402
from xspecies_pv import term_mapping as tm  # noqa: E402


# ------------------------------------------------------------------ flatteners / client
def test_flatten_animal_record_rows_and_dose():
    rec = {
        "report_id": "USA-1", "original_receive_date": "20220310", "serious_ae": "true", "primary_reporter": "Veterinarian",
        "animal": {"species": "Dog", "breed": {"breed_component": "Labrador"}, "gender": "Male",
                   "age": {"min": "5", "unit": "Year"}, "weight": {"min": "30", "max": "30", "unit": "Kilogram"}},
        "drug": [{"brand_name": "Metacam", "route": "Oral", "active_ingredients": [
            {"name": "Meloxicam", "dose": {"numerator": "3", "numerator_unit": "mg", "denominator": "1", "denominator_unit": "dose"}}]}],
        "reaction": [{"veddra_term_name": "Vomiting", "veddra_term_code": "1", "veddra_version": "11"},
                     {"veddra_term_name": "Melaena", "veddra_term_code": "2", "veddra_version": "11"}],
        "outcome": [{"medical_status": "Recovered/Normal", "number_of_animals_affected": "1"}],
    }
    rows = oa.flatten_animal_record(rec)
    assert len(rows) == 2
    assert rows[0]["species"] == "Dog" and rows[0]["active_ingredient"] == "Meloxicam"
    assert rows[0]["dose_value"] == 3.0 and rows[0]["dose_unit"] == "mg"
    assert {r["veddra_term"] for r in rows} == {"Vomiting", "Melaena"}
    assert rows[0]["outcome"] == "Recovered/Normal"
    # missing everything still yields one row
    assert len(oa.flatten_animal_record({})) == 1


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
        if "skip=0" in url:
            return _FakeResp({"meta": {"results": {"total": 3}}, "results": [{"report_id": "a"}, {"report_id": "b"}]})
        if "skip=2" in url:
            return _FakeResp({"meta": {"results": {"total": 3}}, "results": [{"report_id": "c"}]})
        return _FakeResp({}, status=404)


def test_client_pages_through_window_and_species_filter():
    client = oa.OpenFDAAnimalClient(api_key="k", session=_FakeSession(), min_interval=0.0)
    recs = list(client.iter_animal_events(pd.Timestamp("2022-03-01").date(), pd.Timestamp("2022-03-31").date(),
                                          species=["Dog", "Cat"], step_days=31))
    assert [r["report_id"] for r in recs] == ["a", "b", "c"]
    assert "animal.species:(Dog+Cat)" in client.session.calls[0]
    assert "original_receive_date:[20220301+TO+20220331]" in client.session.calls[0]


# ------------------------------------------------------------------ term mapping
def test_normalize_ingredient_and_organ_system_rules():
    assert tm.normalize_ingredient("Meloxicam 1.5 mg/mL") == "meloxicam"
    assert tm.normalize_ingredient("Fluoxetine Hydrochloride") == "fluoxetine"
    assert tm.normalize_ingredient("Levothyroxine sodium (Thyro-Tabs)") == "levothyroxine"
    assert tm.assign_organ_system("Vomiting") == "gastrointestinal"
    assert tm.assign_organ_system("Emesis") == "gastrointestinal"
    assert tm.assign_organ_system("Seizure") == "nervous"
    assert tm.assign_organ_system("Alanine aminotransferase increased") == "hepatic"
    assert tm.assign_organ_system("Lack of expected effectiveness") == "lack_of_efficacy"
    assert tm.assign_organ_system("Injection site swelling") == "injection_application_site"
    assert tm.assign_organ_system("Methaemoglobinaemia") == "haematological"
    assert tm.assign_organ_system("Death") == "death"
    assert tm.assign_organ_system("Zzzz unknown term") == "unclassified"
    # official SOC wins when supplied
    assert tm.assign_organ_system("Weird term", vocab_soc="Cardiac disorders") == "cardiovascular"
    assert tm.soc_to_bucket("Skin and subcutaneous tissue disorders") == "skin_hypersensitivity"


def test_suggest_pt_matches_synonym_and_fuzzy():
    pts = ["Vomiting", "Diarrhoea", "Decreased appetite", "Seizure", "Alanine aminotransferase increased", "Rash"]
    m = tm.suggest_pt_matches("Emesis", pts)
    assert m[0] == ("Vomiting", 1.0)
    m2 = tm.suggest_pt_matches("Diarrhea", pts)
    assert m2[0][0] == "Diarrhoea" and m2[0][1] >= 0.6
    assert tm.suggest_pt_matches("Xyzzy", pts, min_score=0.9) == []


# ------------------------------------------------------------------ signals
def _synthetic_species(rng, n_reports, ingredients, buckets, planted, base_p=0.08, planted_p=0.5):
    rows = []
    for r in range(n_reports):
        ing = ingredients[r % len(ingredients)]
        for b in buckets:
            p = planted_p if (ing, b) in planted else base_p
            if rng.random() < p:
                rows.append({"report_id": f"r{r}", "ingredient": ing, "bucket": b, "quarter": r // (n_reports // 20)})
    return pd.DataFrame(rows)


def test_signal_table_and_concordance_recover_shared_signals():
    rng = np.random.default_rng(0)
    ings = [f"ing{i}" for i in range(12)]
    buckets = ["gastrointestinal", "nervous", "hepatic", "skin_hypersensitivity", "renal_urinary"]
    shared = {("ing0", "gastrointestinal"), ("ing1", "nervous"), ("ing2", "hepatic"), ("ing3", "renal_urinary")}
    dog_only = {("ing4", "nervous")}
    dog = _synthetic_species(rng, 6000, ings, buckets, shared | dog_only)
    human = _synthetic_species(rng, 6000, ings, buckets, shared)
    td, th = cs.signal_table(dog), cs.signal_table(human)
    assert set(td.columns) >= {"ingredient", "bucket", "a", "b", "c", "d", "ic", "ic025", "signal"}
    flagged_dog = set(map(tuple, td.loc[td["signal"], ["ingredient", "bucket"]].values))
    assert shared | dog_only <= flagged_dog
    flagged_h = set(map(tuple, th.loc[th["signal"], ["ingredient", "bucket"]].values))
    assert shared <= flagged_h and ("ing4", "nervous") not in flagged_h
    conc = cs.concordance(td, th, n_boot=50)
    assert conc["rho"] > 0.3 and conc["kappa"] > 0.3
    assert conc["rho_ci"][0] > 0
    null = cs.permutation_null_rho(td, th, n_perm=30)
    assert null["p_value"] < 0.1


def test_lead_lag_detects_earlier_animal_signal():
    rng = np.random.default_rng(1)
    ings = ["ingA", "ingB", "ingC", "ingD"]
    buckets = ["gastrointestinal", "nervous", "hepatic"]
    # animal: signal from the start; human: build a table where the pair only appears late
    dog = _synthetic_species(rng, 4000, ings, buckets, {("ingA", "hepatic")})
    human = _synthetic_species(rng, 4000, ings, buckets, set())
    late = human[(human["ingredient"] == "ingA") & (human["quarter"] >= 12)]
    extra = late.assign(bucket="hepatic").drop_duplicates(["report_id", "bucket"])
    human = pd.concat([human, extra], ignore_index=True)
    ta = cs.cumulative_ic_by_quarter(dog, "ingA", "hepatic")
    th = cs.cumulative_ic_by_quarter(human, "ingA", "hepatic")
    fa, fh = cs.first_sustained_signal(ta), cs.first_sustained_signal(th)
    assert fa is not None and fh is not None and fa < fh
    ll = cs.lead_lag(ta, th)
    assert ll["animal_lead_quarters"] > 0


def test_mg_per_kg_and_allometry():
    assert cs.mg_per_kg(60, "mg", 30, 30, "Kilogram") == pytest.approx(2.0)
    assert cs.mg_per_kg(60, "mg", 66, 66, "lb") == pytest.approx(60 / (66 * 0.4536))
    assert cs.mg_per_kg(2, "mg", None, None, None, dose_denominator_unit="kg") == 2.0
    assert cs.mg_per_kg(2, "tablet", 10, 10, "kg") is None
    assert cs.allometric_dose_ratio(1.0, 1.0, human_kg=70, animal_kg=20) == pytest.approx((20 ** 0.25) / (70 ** 0.25))
