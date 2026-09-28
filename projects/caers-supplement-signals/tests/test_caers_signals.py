"""Synthetic tests for caers_signals (no network)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from caers_signals.normalize import dsld_ingredients_for_products, dsld_search, flatten_caers, flatten_faers_for_supplements, map_ingredients, normalise_product, outcome_flags
from caers_signals.openfda import OpenFDAClient
from caers_signals.signals import cross_system_concordance, monthly_series, poisson_cusum, reference_evaluation, screen, segmented_poisson_its, serious_fraction_model


def test_normalisation_and_lexicon():
    assert normalise_product("Nature Made Vitamin D3 2000 IU, 90 softgels") == "NATURE MADE VITAMIN D3"
    assert map_ingredients("Green Tea Extract Fat Burner") == ["GREEN TEA EXTRACT", "WEIGHT-LOSS BLEND"]
    assert "KRATOM" in map_ingredients("Red Bali Kratom powder") and map_ingredients("plain oatmeal") == []
    assert outcome_flags(["Visited an Emergency Room", "Hospitalization"]) == {"serious": True, "death": False, "hospitalization": True, "medically_attended": True}
    rec = {
        "report_number": "1",
        "date_created": "20230405",
        "outcomes": ["Other Outcome"],
        "reactions": ["Hepatotoxicity", "NAUSEA"],
        "consumer": {"age": "34", "age_unit": "year(s)", "gender": "Female"},
        "products": [{"role": "Suspect", "name_brand": "AG1 Athletic Greens", "industry_code": "54", "industry_name": "Vit/Min/Prot/Unconv Diet(Human/Animal)"}, {"role": "Concomitant", "name_brand": "Coffee", "industry_code": "31"}],
    }
    row = flatten_caers(rec)
    assert row["supplement"] and row["sex"] == "female" and row["age_years"] == 34.0 and not row["serious"]
    assert row["suspect_supplements"] == ["AG1 ATHLETIC GREENS"] and row["reactions"] == ["HEPATOTOXICITY", "NAUSEA"]
    frec = {"safetyreportid": "9", "receivedate": "20230101", "serious": "1", "patient": {"patientsex": "2", "drug": [{"medicinalproduct": "TURMERIC CURCUMIN 500MG"}], "reaction": [{"reactionmeddrapt": "Hepatitis"}]}}
    frow = flatten_faers_for_supplements(frec)
    assert frow["ingredients"] == ["TURMERIC/CURCUMIN"] and frow["supplement"] and frow["serious"]


def simulate(n: int = 5000, seed: int = 0, hepatic_rate: float = 0.35) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ing = rng.choice(["KAVA", "MELATONIN", "MULTIVITAMIN", "PROBIOTIC"], p=[0.15, 0.3, 0.4, 0.15])
        pts = []
        if rng.random() < (hepatic_rate if ing == "KAVA" else 0.03):
            pts.append("HEPATOTOXICITY")
        if rng.random() < 0.2:
            pts.append("NAUSEA")
        if not pts:
            pts.append("HEADACHE")
        month = int(rng.integers(0, 120))
        y, m = 2010 + month // 12, month % 12 + 1
        out.append({"date_created": f"{y}{m:02d}15", "sex": "female" if rng.random() < 0.6 else "male", "age_years": float(rng.uniform(18, 80)), "serious": bool(rng.random() < (0.6 if ing == "KAVA" else 0.3)), "reactions": pts, "ingredients": [str(ing)]})
    return out


def test_screen_recovers_planted_signal():
    reps = simulate()
    tab = screen(reps, min_a=3)
    kava = tab[(tab["exposure"] == "KAVA") & (tab["event"] == "HEPATOTOXICITY")].iloc[0]
    assert kava["ror"] > 5 and kava["ror_signal"] and kava["evans_signal"] and kava["ic_signal"]
    mel = tab[(tab["exposure"] == "MELATONIN") & (tab["event"] == "HEPATOTOXICITY")].iloc[0]
    assert not mel["ror_signal"]
    sf = serious_fraction_model(reps, "KAVA")
    assert sf.loc["exposed", "or"] > 2 and sf.loc["exposed", "p"] < 0.001


def test_time_series_tools():
    reps = simulate(3000, seed=2)
    s = monthly_series(reps)
    assert len(s) >= 110 and s.sum() == 3000
    rng = np.random.default_rng(3)
    base = np.full(72, 20.0)
    y = rng.poisson(np.where(np.arange(72) >= 40, base * 2.0, base))
    its = segmented_poisson_its(y, change_index=40)
    assert its["level_lo"] > 1.3 and its["p_level"] < 0.001
    cs = poisson_cusum(y, expected=20.0)
    assert cs["first_alarm"] >= 40 and cs["n_alarms"] >= 1
    assert poisson_cusum([20] * 30, expected=20.0)["first_alarm"] == -1


def test_concordance_and_reference():
    caers = screen(simulate(4000, seed=4), min_a=3)
    faers = screen(simulate(4000, seed=5, hepatic_rate=0.3), min_a=3)
    c = cross_system_concordance(caers, faers)
    assert c["n_pairs"] >= 6 and c["spearman"] > 0.5 and c["jaccard_signals"] > 0
    ev = reference_evaluation(caers, positives=[("KAVA", "HEPATOTOXICITY")], negatives=[("MELATONIN", "HEPATOTOXICITY"), ("MULTIVITAMIN", "HEPATOTOXICITY"), ("PROBIOTIC", "HEADACHE")], n_boot=50)
    assert ev.set_index("score").loc["ror", "auroc"] == 1.0


class _Resp:
    def __init__(self, payload: Any, status: int = 200, link: str = ""):
        self._payload, self.status_code, self.headers, self.text = payload, status, ({"Link": link} if link else {}), json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls: List[str] = []

    def get(self, url: str, params=None, timeout=None):
        self.calls.append(url)
        if "search-filter" in url:
            return _Resp({"hits": [{"_id": "12345", "_source": {"fullName": "AG1", "brandName": "Athletic Greens", "ingredientRows": [{"name": "Spirulina"}, {"name": "Ashwagandha extract"}]}}]})
        if "search_after" in url:
            return _Resp({"meta": {"results": {"total": 3}}, "results": [{"report_number": "3", "products": []}]})
        if params and params.get("count"):
            return _Resp({"results": [{"term": "54", "count": 1000}]})
        return _Resp({"meta": {"results": {"total": 3}}, "results": [{"report_number": "1", "products": []}, {"report_number": "2", "products": []}]}, link='<https://api.fda.gov/food/event.json?limit=2&search_after=x>; rel="next"')


def test_dsld_and_client_pagination():
    sess = _FakeSession()
    hits = dsld_search("AG1", sess)
    assert hits[0]["dsld_id"] == "12345" and hits[0]["ingredients"] == ["ASHWAGANDHA EXTRACT", "SPIRULINA"]
    cache = dsld_ingredients_for_products(["AG1", "AG1"], sess)
    assert cache["AG1"] == ["ASHWAGANDHA EXTRACT", "SPIRULINA"] and sum("search-filter" in u for u in sess.calls) == 2
    client = OpenFDAClient(api_key="k", session=sess, max_per_minute=10000)
    recs = list(client.iter_records("food/event", "products.industry_code:54", limit=2))
    assert [r["report_number"] for r in recs] == ["1", "2", "3"]
    assert client.count("food/event", None, "products.industry_code")[0]["count"] == 1000
    assert client.date_range("2004-01-01", "2026-06-30", "date_created") == "date_created:[20040101 TO 20260630]"
