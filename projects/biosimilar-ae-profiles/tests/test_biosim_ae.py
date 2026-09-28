"""Synthetic tests for biosim_ae (no network)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from biosim_ae.labels import label_similarity, labelled_flags, pt_in_label, unlabelled_fraction
from biosim_ae.products import FAMILIES, attributability_summary, attribute_drug_entry, attribute_report
from biosim_ae.profiles import PANELS, calendar_window, comparator_ror, jensen_shannon, launch_curve, panel_comparison, profile_distance_test, ror, stratified_ror, weber_index


def test_attribution_hierarchy():
    assert attribute_drug_entry({"openfda": {"brand_name": ["HYRIMOZ"], "generic_name": ["ADALIMUMAB-ADAZ"]}})["product"] == "HYRIMOZ"
    assert attribute_drug_entry({"medicinalproduct": "Adalimumab-atto"})["product"] == "AMJEVITA"
    assert attribute_drug_entry({"medicinalproduct": "HUMIRA PEN"})["kind"] == "originator"
    assert attribute_drug_entry({"openfda": {"generic_name": ["ADALIMUMAB"]}})["kind"] == "inn_only"
    assert attribute_drug_entry({"medicinalproduct": "XGEVA"})["inn"] == "DENOSUMAB"
    assert attribute_drug_entry({"medicinalproduct": "METFORMIN"}) is None
    # a biosimilar mention wins over an originator mention in another field
    e = {"medicinalproduct": "HUMIRA", "openfda": {"brand_name": ["CYLTEZO"]}}
    assert attribute_drug_entry(e)["product"] == "CYLTEZO"


def test_attribute_report_and_summary():
    rec = {
        "safetyreportid": "1",
        "receivedate": "20230815",
        "patient": {"patientsex": "2", "drug": [{"medicinalproduct": "HADLIMA", "drugcharacterization": "1"}, {"medicinalproduct": "METHOTREXATE"}], "reaction": [{"reactionmeddrapt": "Injection site pain"}]},
    }
    r = attribute_report(rec)
    assert r["inns"] == ["ADALIMUMAB"] and r["products"][0]["role"] == "1" and r["reactions"] == ["INJECTION SITE PAIN"]
    s = attributability_summary([r, {"products": [{"inn": "ADALIMUMAB", "kind": "inn_only"}]}], "ADALIMUMAB")
    assert s == {"originator": 0, "biosimilar": 1, "inn_only": 1, "mixed": 0}
    assert all(b.nonproprietary.startswith(f.inn.lower().split()[0]) for f in FAMILIES.values() for b in f.biosimilars)


def _reports(n: int, p_event: float, event: str, seed: int, month0: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pts = ["HEADACHE"] if rng.random() < 0.3 else []
        pts += [event] if rng.random() < p_event else []
        pts += ["NAUSEA"] if rng.random() < 0.2 else []
        month = month0 + int(rng.integers(0, 24))
        y, m = 2023 + month // 12, month % 12 + 1
        out.append({"receivedate": f"{y}{m:02d}15", "reactions": pts, "qualification": "5" if rng.random() < 0.6 else "1"})
    return out


def test_comparator_ror_detects_enriched_pt():
    bio = _reports(1500, 0.20, "INJECTION SITE PAIN", 1)
    orig = _reports(3000, 0.05, "INJECTION SITE PAIN", 2)
    df = comparator_ror(bio, orig)
    row = df.set_index("event").loc["INJECTION SITE PAIN"]
    assert row["ror"] > 2 and row["signal"]
    assert not df.set_index("event").loc["HEADACHE", "signal"]
    r = ror(10, 90, 5, 95)
    assert r["ror_lo"] < r["ror"] < r["ror_hi"]


def test_profile_distance_and_js():
    p = np.array([0.5, 0.5, 0.0])
    assert jensen_shannon(p, p) == 0.0
    assert abs(jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
    same_a, same_b = _reports(400, 0.1, "X", 3), _reports(400, 0.1, "X", 4)
    res = profile_distance_test(same_a, same_b, n_perm=100)
    assert res["p_perm"] > 0.01
    diff = _reports(400, 0.6, "X", 5)
    res2 = profile_distance_test(same_a, diff, n_perm=100)
    assert res2["p_perm"] < 0.05 and res2["js"] > res["js"]


def test_launch_alignment_and_weber():
    reps = _reports(500, 0.1, "X", 6)
    win = calendar_window(reps, "2023-07", 6)
    assert all(r["receivedate"][:6] in {"202307", "202308", "202309", "202310", "202311", "202312"} for r in win)
    curve = launch_curve(reps, "2023-01", 48)
    assert curve["count"].sum() == len(reps)
    w = weber_index(curve)
    assert 0 <= w["peak_month"] < 24 and w["frac_first_year"] > 0.4


def test_panels_and_stratified():
    bio = _reports(800, 0.3, "DEVICE MALFUNCTION", 7)
    orig = _reports(800, 0.05, "DEVICE MALFUNCTION", 8)
    pc = panel_comparison(bio, orig)
    dev = pc.set_index("panel").loc["device"]
    assert dev["ror"] > 2 and dev["p_fisher"] < 0.001
    assert set(pc["panel"]) == set(PANELS)
    mh = stratified_ror(bio, orig, "DEVICE MALFUNCTION", "qualification")
    assert mh["ror_mh"] > 2 and mh["n_strata"] == 2


def test_label_matching_and_similarity():
    text = "warnings serious infections including tuberculosis hypersensitivity reactions anaphylaxis injection site reactions hypocalcemia"
    assert pt_in_label("TUBERCULOSIS", text) and pt_in_label("ANAPHYLACTIC REACTION", text)
    assert pt_in_label("HYPOCALCAEMIA", text)  # British spelling handled
    assert not pt_in_label("PANCREATITIS", text)
    flags = labelled_flags(["TUBERCULOSIS", "PANCREATITIS"], text)
    assert flags == {"TUBERCULOSIS": True, "PANCREATITIS": False}
    frac = unlabelled_fraction([{"reactions": ["TUBERCULOSIS"]}, {"reactions": ["PANCREATITIS", "TUBERCULOSIS"]}, {"reactions": ["PANCREATITIS"]}], text)
    assert frac["frac_any_unlabelled"] == 2 / 3 and frac["frac_all_unlabelled"] == 1 / 3
    assert label_similarity(text, text) == 1.0
    assert label_similarity(text, text + " new paragraph about the autoinjector device use instructions") < 1.0
