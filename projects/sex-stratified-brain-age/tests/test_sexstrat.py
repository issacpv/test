"""Synthetic tests for sexstrat_brainage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sexstrat_brainage import evaluation, freesurfer, models, simulation


@pytest.fixture(scope="module")
def sim():
    df = simulation.simulate_lifespan_cohort(n=500, seed=3, cfg=simulation.scenario("head_size_only"))
    X = df[simulation.feature_columns(df)]
    return df, X


def test_assemble_features_and_tiv(tmp_path):
    subj = ["s1", "s2", "s3", "s4"]
    lh = pd.DataFrame({"lh.aparc.thickness": subj, "lh_bankssts_thickness": [2.5, 2.6, 2.4, 2.3],
                       "lh_MeanThickness_thickness": [2.5, 2.6, 2.4, 2.3]})
    rh = pd.DataFrame({"rh.aparc.thickness": subj, "rh_bankssts_thickness": [2.4, 2.7, 2.5, 2.2]})
    aseg = pd.DataFrame({"Measure:volume": subj, "Left-Hippocampus": [4000, 4300, 3950, 4100],
                         "EstimatedTotalIntraCranialVol": [1.4e6, 1.6e6, 1.35e6, 1.5e6]})
    for name, tab in (("lh.tsv", lh), ("rh.tsv", rh), ("aseg.tsv", aseg)):
        tab.to_csv(tmp_path / name, sep="\t", index=False)
    feats = freesurfer.assemble_features(freesurfer.read_stats_table(tmp_path / "lh.tsv"),
                                         freesurfer.read_stats_table(tmp_path / "rh.tsv"),
                                         freesurfer.read_stats_table(tmp_path / "aseg.tsv"))
    assert list(feats.index) == subj
    assert "thk_lh_bankssts" in feats and "vol_Left-Hippocampus" in feats and "eTIV" in feats
    for method in ("none", "covariate", "proportion", "residual", "power"):
        out = freesurfer.TIVCorrector(method).fit_transform(feats)
        assert out.shape[0] == 4
        if method == "residual":
            # residualized volume should be uncorrelated with TIV
            assert abs(np.corrcoef(out["vol_Left-Hippocampus"], feats["eTIV"])[0, 1]) < 1e-6


def test_strategies_run_and_reduce_spurious_sex_effect(sim):
    df, X = sim
    X = freesurfer.feature_set(X, "volumes")  # head-size confound lives in the volume family
    age, sex, groups = df["age"].to_numpy(), df["sex"].to_numpy(), df["group"].to_numpy()
    res = {}
    for strat, tiv in (("pooled", "none"), ("pooled_sex", "residual"), ("stratified", "residual"), ("pooled_sexbias", "none")):
        est = models.BrainAgeEstimator(strategy=strat, tiv=tiv, bias="beheshti", inner_folds=3)
        res[f"{strat}/{tiv}"] = models.cross_validated_delta(X, age, sex, groups, est, n_splits=4)
    tab = evaluation.summarize_strategies(res, outcome=None)
    assert (tab["mae_all"] < 8).all()
    # in the head-size-only world, a pooled model ignoring TIV manufactures a sex effect
    naive = tab.set_index("strategy").loc["pooled/none", "spur_beta_sex"]
    aware = tab.set_index("strategy").loc["pooled_sex/residual", "spur_beta_sex"]
    assert naive < 0  # males (larger heads, larger volumes) are predicted younger by the naive model
    assert abs(aware) < abs(naive)
    # corrected delta should not depend on age (bias correction worked)
    assert abs(tab["spur_beta_age"]).max() < 0.15


def test_recovery_and_outcome(sim):
    df, X = sim
    est = models.BrainAgeEstimator(strategy="pooled_sex", tiv="residual", inner_folds=3)
    res = models.cross_validated_delta(X, df["age"].to_numpy(), df["sex"].to_numpy(), df["group"].to_numpy(), est, n_splits=4)
    rec = simulation.recovery_error(res, df)
    assert rec["corr_all"] > 0.5
    res["outcome"] = df["outcome"].to_numpy()
    assoc = evaluation.outcome_association(res, "outcome")
    assert assoc["beta_std"] < 0 and assoc["p"] < 0.05
    bysex = evaluation.association_by_sex(res, "outcome")
    assert set(bysex["sex"]) == {"female", "male", "interaction"}
    gap = evaluation.bootstrap_sex_gap(res, n_boot=50)
    assert gap["ci_low"] <= gap["sex_gap"] <= gap["ci_high"] + 1e-9


def test_icc_and_retest():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 3, 40)
    Y = np.stack([truth + rng.normal(0, 1, 40), truth + rng.normal(0, 1, 40)], axis=1)
    icc = evaluation.icc_2_1(Y)
    assert 0.7 < icc < 1.0
    res = pd.DataFrame({"group": np.repeat(np.arange(40), 2), "session": np.tile([1, 2], 40),
                        "delta": Y.reshape(-1), "sex": np.repeat(rng.integers(0, 2, 40), 2)})
    out = evaluation.retest_reliability(res)
    assert out["icc_all"] == pytest.approx(icc)


def test_subsampled_control(sim):
    df, X = sim
    est = models.BrainAgeEstimator(strategy="stratified", tiv="residual", inner_folds=3)
    ctrl = models.pooled_subsampled_control(X, df["age"].to_numpy(), df["sex"].to_numpy(), df["group"].to_numpy(),
                                            est, frac=0.5, n_splits=3)
    assert len(ctrl) == 250
