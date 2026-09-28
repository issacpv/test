"""Synthetic tests for glp1_sexpv (no network)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from glp1_sexpv.cohort import classify_indication, cohort_summary, flatten_report, is_compounded, match_agent
from glp1_sexpv.denominators import excess_female_reporting, flag_glp1_fills, poisson_rate, rate_ratio, sex_reporting_rates, standardise_by_age, users_from_fills
from glp1_sexpv.sex_signals import adjusted_sex_interaction, aggregate_counts, mh_ror_by_sex, ratio_of_ror, sex_difference_atlas, sex_tables


def test_cohort_matching():
    assert classify_indication("Type 2 diabetes mellitus") == "t2d"
    assert classify_indication("Weight management", None) == "obesity"
    assert classify_indication(None, "WEGOVY") == "obesity" and classify_indication("", "OZEMPIC") == "t2d"
    assert classify_indication("Product used for unknown indication", "ZEPBOUND") == "obesity"
    assert is_compounded("SEMAGLUTIDE COMPOUNDED", True) and is_compounded("semaglutide", False)
    assert not is_compounded("OZEMPIC", True)
    m = match_agent({"openfda": {"brand_name": ["MOUNJARO"], "generic_name": ["TIRZEPATIDE"]}, "drugindication": "Type 2 diabetes", "drugcharacterization": "1"})
    assert m["agent"] == "TIRZEPATIDE" and m["brand"] == "MOUNJARO" and m["indication"] == "t2d" and not m["compounded"]
    assert match_agent({"medicinalproduct": "METFORMIN"}) is None
    rec = {
        "patient": {
            "patientsex": "2",
            "patientonsetage": "45",
            "patientonsetageunit": "801",
            "drug": [{"medicinalproduct": "WEGOVY", "drugcharacterization": "1"}, {"openfda": {"generic_name": ["EMPAGLIFLOZIN"]}}],
            "reaction": [{"reactionmeddrapt": "Alopecia"}],
        }
    }
    row = flatten_report(rec)
    assert row["glp1"] and row["glp1_suspect"] and row["indication"] == "obesity" and row["comparators"] == ["sglt2"] and row["reactions"] == ["ALOPECIA"]
    assert cohort_summary([row])["n_exposed"] == 1


def simulate(n: int = 20000, sex_effect: float = 3.0, confounded: bool = False, seed: int = 0) -> List[Dict[str, Any]]:
    """Exposed/unexposed reports with a planted PT 'X'.

    sex_effect: multiplicative ROR ratio (female/male) among exposed for X.
    confounded: instead of a true sex effect, women are mostly 'obesity'
    indication and 'obesity' indication raises X in exposed and unexposed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        female = rng.random() < 0.5
        exposed = rng.random() < 0.3
        if confounded:
            ind = "obesity" if rng.random() < (0.8 if female else 0.2) else "t2d"
            logit = -3.0 + 1.0 * exposed + 1.5 * (ind == "obesity")
        else:
            ind = "t2d"
            logit = -3.0 + 1.0 * exposed + (np.log(sex_effect) if (exposed and female) else 0.0)
        p = 1 / (1 + np.exp(-logit))
        pts = ["X"] if rng.random() < p else []
        pts += ["NAUSEA"] if rng.random() < (0.3 if exposed else 0.1) else []
        out.append({"sex": "female" if female else "male", "glp1": exposed, "indication": ind, "reactions": pts})
    return out


def test_atlas_recovers_sex_specific_signal():
    reps = simulate(sex_effect=3.0)
    atlas = sex_difference_atlas(reps, lambda r: r["glp1"], min_a=3)
    x = atlas.set_index("event").loc["X"]
    assert x["ratio"] > 1.8 and x["sex_differs"]
    nausea = atlas.set_index("event").loc["NAUSEA"]
    assert nausea["signal_female"] and nausea["signal_male"] and not nausea["sex_differs"]
    t = sex_tables(reps, "X", lambda r: r["glp1"])
    res = ratio_of_ror(t)
    assert res["ror_female"] > res["ror_male"] and res["p_lrt"] < 0.001


def test_indication_adjustment_removes_confounded_sex_difference():
    reps = simulate(confounded=True, seed=3)
    agg = aggregate_counts(reps, "X", lambda r: r["glp1"], strata=("indication",))
    res = adjusted_sex_interaction(agg)
    # crude and adjusted are both estimable; adjusted must be closer to the null (1.0)
    assert abs(np.log(res["ratio_adj"])) < abs(np.log(res["ratio_crude"])) + 0.05
    assert res["ratio_adj_lo"] < 1.0 < res["ratio_adj_hi"] or abs(np.log(res["ratio_adj"])) < 0.35
    mh = mh_ror_by_sex(agg)
    assert mh["female"]["ror_mh"] > 1 and mh["male"]["ror_mh"] > 1
    assert 0.6 < mh["ratio_mh"]["ratio"] < 1.6


def test_denominators():
    r = poisson_rate(20, 100_000)
    assert r["rate_lo"] < r["rate"] < r["rate_hi"] and abs(r["rate"] - 2.0) < 1e-9
    rr = rate_ratio(60, 30_000, 20, 20_000)
    assert 1.5 < rr["rr"] < 2.5 and rr["rr_lo"] > 1
    df = sex_reporting_rates({"female": 650, "male": 350}, {"female": 700_000, "male": 300_000})
    assert df.set_index("sex").loc["female/male", "rate"] < 1.0  # fewer reports per user in women here
    ex = excess_female_reporting(650, 350, 500_000, 500_000)
    assert ex["propensity_ratio"] > 1.2 and ex["p_binomial"] < 1e-6
    rx = pd.DataFrame({"DUPERSID": ["a", "b", "c"], "RXDRGNAM": ["OZEMPIC", "METFORMIN", "Semaglutide"]})
    persons = pd.DataFrame({"DUPERSID": ["a", "b", "c", "d"], "SEX": [2, 1, 1, 2], "PERWT": [1000.0, 2000.0, 1500.0, 500.0]})
    mask = flag_glp1_fills(rx)
    assert mask.tolist() == [True, False, True]
    users = users_from_fills(rx, persons, "DUPERSID", mask, "SEX", "PERWT")
    assert users == {"female": 1000.0, "male": 1500.0}
    rc = pd.DataFrame({"sex": ["female", "female", "male", "male"], "age_band": ["<50", "50+", "<50", "50+"], "n": [10, 10, 5, 5]})
    us = pd.DataFrame({"sex": ["female", "female", "male", "male"], "age_band": ["<50", "50+", "<50", "50+"], "users": [1000, 1000, 500, 500]})
    st = standardise_by_age(rc, us)
    assert np.allclose(st["std_rate"].values, [100.0, 100.0])
