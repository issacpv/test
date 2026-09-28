"""Synthetic tests for scannability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scannability import iqm, models, participants, simulate


def test_parse_age_sex_group():
    assert participants.parse_age("23") == 23
    assert participants.parse_age("20-25") == 22.5
    assert participants.parse_age("89+") == 89
    assert participants.parse_age("18 months") == pytest.approx(1.5)
    assert np.isnan(participants.parse_age("n/a"))
    assert participants.parse_sex("female") == "F" and participants.parse_sex("1") == "M" and participants.parse_sex(None) == "U"
    assert participants.coarse_group("Healthy Control") == "control"
    assert participants.coarse_group("SCHZ") == "psychiatric"
    assert participants.coarse_group("stroke_left") == "neurological"
    assert participants.coarse_group("ASD") == "neurodevelopmental"
    assert participants.coarse_group("patient") == "other_clinical"
    assert participants.coarse_group("n/a") == "unknown"


def test_harmonize_and_load(tmp_path):
    ds = tmp_path / "ds000001"
    ds.mkdir()
    pd.DataFrame({"participant_id": ["sub-01", "sub-02"], "Age": ["25", "30-35"], "gender": ["M", "F"],
                  "diagnosis": ["control", "bipolar"]}).to_csv(ds / "participants.tsv", sep="\t", index=False)
    corpus = participants.load_corpus_participants(tmp_path)
    assert list(corpus["age"]) == [25.0, 32.5]
    assert list(corpus["group"]) == ["control", "psychiatric"]
    rev = participants.mapping_review_table(corpus)
    assert set(rev.columns) == {"group_raw", "group", "n"}


def test_rules_and_group_table(tmp_path):
    g = pd.DataFrame({"bids_name": ["sub-01_task-rest_bold", "sub-01_task-rest_run-2_bold", "sub-02_task-rest_bold"],
                      "fd_mean": [0.1, 0.4, 0.15], "fd_perc": [5, 40, 10], "tsnr": [50, 30, 55]})
    g.to_csv(tmp_path / "group_bold.tsv", sep="\t", index=False)
    t = iqm.load_group_table(tmp_path / "group_bold.tsv", "ds000001", "bold")
    pp = iqm.per_participant(t, "bold", agg="worst")
    assert len(pp) == 2 and pp.loc[pp["participant_id"] == "sub-01", "fd_mean"].iloc[0] == 0.4
    rs = iqm.literature_rulesets()
    flagged = iqm.apply_rulesets(pp, {"s": rs["bold_fd_strict"]})
    assert list(flagged["excl_bold_fd_strict"]) == [1, 0]


@pytest.fixture(scope="module")
def corpus():
    df = simulate.simulate_corpus(n_datasets=25, seed=1, mean_n=70)
    sweep = iqm.fd_threshold_sweep((0.2, 0.3, 0.5))
    df = iqm.apply_rulesets(df, sweep)
    df = iqm.apply_rulesets(df, {"t": iqm.literature_rulesets()["t1w_relative10"]})
    return df


def test_glm_recovers_structure(corpus):
    fit = models.fit_exclusion_glm(corpus, "excl_fd_mean_gt_0p3")
    ors = models.odds_ratios(fit)
    neuro = ors.set_index("term").loc["group[T.neurological]", "OR"]
    assert neuro > 1.0
    curve = models.age_curve(fit, corpus, "excl_fd_mean_gt_0p3")
    # U-shape: exclusion at the extremes exceeds exclusion at ~30 y
    mid = curve.iloc[(curve["age"] - 30).abs().idxmin()]["p_excluded"]
    assert curve["p_excluded"].iloc[-1] > mid and curve["p_excluded"].iloc[0] > mid
    si = models.scannability_index(fit, corpus, "excl_fd_mean_gt_0p3")
    assert si.between(0, 1).all()


def test_elasticity_and_representation(corpus):
    el = models.exclusion_elasticity(corpus, ["excl_fd_mean_gt_0p2", "excl_fd_mean_gt_0p3", "excl_fd_mean_gt_0p5"], [0.2, 0.3, 0.5])
    assert (el["elasticity"] <= 0).all()  # looser threshold → fewer exclusions
    rep = models.representation_shift(corpus, "excl_fd_mean_gt_0p3")
    assert rep["pass_rate"] < 1.0 and rep["share_over65_passed"] <= rep["share_over65_acquired"] + 0.02


def test_biology_adjustment_reduces_age_structure(corpus):
    ref = (corpus["group"] == "control") & (corpus["excl_fd_mean_gt_0p3"] == 0)
    adj = models.biology_adjust_iqms(corpus, ["cnr", "cjv"], reference_mask=ref)
    assert "cnr_bioadj" in adj
    # age–CNR correlation should shrink after adjustment (within reference rows)
    r_before = abs(np.corrcoef(adj.loc[ref, "age"], adj.loc[ref, "cnr"])[0, 1])
    r_after = abs(np.corrcoef(adj.loc[ref, "age"], adj.loc[ref, "cnr_bioadj"])[0, 1])
    assert r_after < r_before
