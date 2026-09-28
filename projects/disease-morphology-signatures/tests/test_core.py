"""Synthetic-data tests for disease_morph (no network, no real data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disease_morph.cohort import build_cohort, map_condition
from disease_morph.meta import (contrast_effects, egger_test, hedges_g, random_effects,
                                signature_similarity_test)
from disease_morph.passive import (PassiveParams, ball_and_stick_input_resistance,
                                   electrotonic_summary, input_impedance)
from disease_morph.swc import morphology_from_array, morphometrics, sholl_profile


def _ball_and_stick(r_soma=10.0, d=2.0, length=400.0, n_seg=40, extra_branch=False):
    # soma at the origin; dendrite nodes every `step` so that every segment (including
    # the one from the soma node to the first dendrite node) has length `step`
    rows = [[1, 1, 0, 0, 0, r_soma, -1]]
    step = length / n_seg
    for k in range(1, n_seg + 1):
        rows.append([k + 1, 3, k * step, 0, 0, d / 2, k])
    if extra_branch:
        base = len(rows)
        parent = n_seg // 2 + 1
        for k in range(1, 11):
            rows.append([base + k, 3, rows[parent - 1][2], k * 10.0, 0, d / 2, parent if k == 1 else base + k - 1])
    return morphology_from_array(np.asarray(rows, float))


def test_morphometrics_on_synthetic_tree():
    m = _ball_and_stick(extra_branch=True)
    mm = morphometrics(m, sholl_radii=[50, 100, 200, 300])
    assert mm["total_length"] == pytest.approx(400.0 + 100.0, rel=1e-6)
    assert mm["n_tips"] == 2
    assert mm["n_bifurcations"] == 1
    assert mm["max_branch_order"] == 1
    prof = sholl_profile(m, [50, 250, 450])
    assert list(prof) == [1, 2, 0] or list(prof) == [1, 1, 0]  # branch may or may not cross 250 um shell


def test_hedges_g_and_random_effects_recover_true_effect():
    rng = np.random.default_rng(0)
    effs, vars_ = [], []
    for _ in range(12):
        x = rng.normal(-0.5, 1, 30)
        y = rng.normal(0, 1, 30)
        g, v = hedges_g(x, y)
        effs.append(g)
        vars_.append(v)
    pooled = random_effects(effs, vars_, method="REML")
    assert -0.8 < pooled.estimate < -0.2
    assert pooled.ci_low < pooled.estimate < pooled.ci_high
    assert pooled.k == 12
    dl = random_effects(effs, vars_, method="DL", hksj=False)
    assert abs(dl.estimate - pooled.estimate) < 0.2
    eg = egger_test(effs, vars_)
    assert eg["k"] == 12 and np.isfinite(eg["p"])


def test_cohort_and_contrasts_and_permutation():
    rng = np.random.default_rng(1)
    records = []
    for archive in ["LabA", "LabB", "LabC"]:
        for cond, n in [("Control", 8), ("APP/PS1", 8), ("aged", 8)]:
            for i in range(n):
                records.append({
                    "neuron_name": f"{archive}_{cond}_{i}", "archive": archive, "species": "mouse",
                    "brain_region": ["hippocampus"], "cell_type": ["pyramidal"],
                    "experiment_condition": [cond], "physical_Integrity": "Dendrites Complete",
                    "structural_domains": "Dendrites, Soma", "reference_pmid": ["1"],
                })
    cohort, audit = build_cohort(records, min_per_group=5)
    assert set(cohort["condition_class"]) == {"control", "ad_model", "aging"}
    assert cohort["in_contrast"].all()
    assert map_condition(["Control"]) == "control"
    assert map_condition("pilocarpine model of epilepsy") == "epilepsy"
    assert map_condition("24 months old") == "aging"
    cohort["total_length"] = rng.normal(1000, 100, len(cohort)) - 150 * (cohort["condition_class"] == "ad_model")
    cohort["n_tips"] = rng.normal(30, 5, len(cohort)) - 4 * (cohort["condition_class"] == "ad_model")
    eff = contrast_effects(cohort, "total_length")
    assert set(eff["condition_class"]) == {"ad_model", "aging"}
    ad = eff[eff["condition_class"] == "ad_model"]
    pooled = random_effects(ad["g"], ad["var"])
    assert pooled.estimate < -0.5
    res = signature_similarity_test(cohort, ["total_length", "n_tips"], "ad_model", "aging", n_perm=10)
    assert 0 < res["p_perm"] <= 1 and np.isfinite(res["observed"])


def test_passive_ball_and_stick_matches_rall():
    p = PassiveParams()
    m = _ball_and_stick(r_soma=10.0, d=2.0, length=400.0, n_seg=80)
    num = electrotonic_summary(m, p)["input_resistance_mohm"]
    ana = ball_and_stick_input_resistance(10.0, 2.0, 400.0, p)
    assert num == pytest.approx(ana, rel=0.01)
    res = input_impedance(m, p, freqs_hz=(0.0, 1000.0))
    assert res["z_abs"][1] < res["z_abs"][0]  # impedance falls with frequency


def test_hedges_g_degenerate_inputs():
    g, v = hedges_g([1.0], [1.0, 2.0])
    assert np.isnan(g) and np.isnan(v)
    pooled = random_effects([], [])
    assert pooled.k == 0


def test_contrast_frame_is_empty_when_no_contrasts():
    df = pd.DataFrame({"contrast_id": ["", ""], "condition_class": ["control", "ad_model"], "x": [1.0, 2.0]})
    assert contrast_effects(df, "x").empty
