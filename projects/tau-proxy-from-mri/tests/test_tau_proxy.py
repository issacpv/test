"""Synthetic-data tests for tau_proxy core functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tau_proxy import decision, discordance, features, labels, models, tables


@pytest.fixture(scope="module")
def cohort():
    tabs = tables.simulate_cohort(n_subjects=240, seed=1)
    df = tables.build_pet_mr_table(tabs["tau"], tabs["mr"], tabs["clinical"], tabs["demographics"])
    comp = labels.tau_composites(df)
    thr = labels.TauThresholds(meta_temporal=1.18, neocortical=1.08)
    df["tau_meta_temporal"] = comp["tau_meta_temporal"]
    df["tau_neocortical"] = comp["tau_neocortical"]
    df["T_pos"] = labels.binary_positivity(df["tau_meta_temporal"], thr)
    df["stage"] = labels.t2_stage(df["tau_meta_temporal"], df["tau_neocortical"], thr)
    return df


def test_parse_oasis_id():
    assert tables.parse_oasis_id("OAS30001_AV1451_d1234") == ("OAS30001", "AV1451", 1234)
    with pytest.raises(ValueError):
        tables.parse_oasis_id("sub-01_ses-1")


def test_apoe_count():
    assert tables.apoe_e4_count("34") == 1
    assert tables.apoe_e4_count("E4/E4") == 2
    assert np.isnan(tables.apoe_e4_count(None))


def test_table_matching(cohort):
    assert cohort["subject"].is_unique
    assert cohort["has_mr"].all()
    assert (cohort["gap_mr"] <= 365).all()
    assert cohort["age"].between(55, 100).all()


def test_labels_and_stage(cohort):
    assert set(cohort["T_pos"].dropna().unique()) <= {0.0, 1.0}
    assert set(cohort["stage"].dropna().unique()) <= {0.0, 1.0, 2.0}
    # neocortical stage implies meta-temporal positivity in the typical (non-atypical) case
    sens = labels.threshold_sensitivity(cohort["tau_meta_temporal"], labels.TauThresholds())
    assert (sens["prevalence"].diff().dropna() <= 0).all()


def test_freesurfer_parsing(tmp_path):
    aseg = tmp_path / "aseg.stats"
    aseg.write_text(
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1520000.0, mm^3\n"
        "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean\n"
        "  1  17  4000  4100.5  Left-Hippocampus  80.1\n"
        "  2  53  4200  4300.2  Right-Hippocampus 81.0\n"
    )
    lh = tmp_path / "lh.aparc.stats"
    lh.write_text("# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd\n"
                  "entorhinal 600 400 1500 3.31 0.7\ninferiortemporal 5000 3000 9000 2.80 0.5\n")
    feats = features.load_freesurfer_session(tmp_path)
    assert feats["eTIV"] == 1520000.0
    assert feats["Left-Hippocampus"] == 4100.5
    assert feats["lh_entorhinal_thickness"] == pytest.approx(3.31)


def test_roi_features_and_combat(cohort):
    X = features.build_roi_features(cohort)
    assert "signature_thickness" in X and "hippocampus_norm" in X and "log_wmh" in X
    assert not X.isna().any().any()
    cb = features.ComBat().fit(X.to_numpy(), cohort["scanner"].to_numpy())
    Xh = cb.transform(X.to_numpy(), cohort["scanner"].to_numpy())
    # batch mean difference in thickness columns should shrink after harmonization
    tcols = [i for i, c in enumerate(X.columns) if c.endswith("_thickness")]
    m = cohort["scanner"].to_numpy() == "TrioTim"
    before = np.abs(X.to_numpy()[m][:, tcols].mean(0) - X.to_numpy()[~m][:, tcols].mean(0)).mean()
    after = np.abs(Xh[m][:, tcols].mean(0) - Xh[~m][:, tcols].mean(0)).mean()
    assert after < before


def test_nested_cv_and_delta(cohort):
    X = features.build_roi_features(cohort)
    C = models.covariate_matrix(cohort)
    y = cohort["T_pos"].to_numpy().astype(int)
    g = cohort["subject"].to_numpy()
    base = models.nested_cv(C, y, g, "logistic", n_outer=3, n_inner=2, n_repeats=1, name="covariates")
    roi = models.nested_cv(pd.concat([C, X], axis=1), y, g, "logistic", n_outer=3, n_inner=2, n_repeats=1, name="cov+roi")
    assert 0.0 <= base.auc() <= 1.0 and roi.auc() > 0.55
    d = models.bootstrap_delta_auc(base.mean_prob(), roi.mean_prob(), y, g, n_boot=100)
    assert d["ci_low"] <= d["delta_auc"] <= d["ci_high"]
    tab = models.summarize([base, roi], baseline="covariates", n_boot=50)
    assert set(tab["model"]) == {"covariates", "cov+roi"}


def test_discordance_and_decision(cohort):
    X = features.build_roi_features(cohort)
    a_neg_cn = (cohort["cdr"] == 0) & (cohort["T_pos"] == 0)
    n = labels.n_status(X["hippocampus_norm"], X["signature_thickness"], a_neg_cn)
    prob = 1 / (1 + np.exp(-(3 * (cohort["tau_meta_temporal"] - 1.18) * 10 - 0.5 * X["hippocampus_norm"].to_numpy() + 1)))
    rep = discordance.discordance_report(prob, cohort["T_pos"], n["N_pos"])
    assert set(rep) >= {"auc_all", "auc_N_neg", "fpr_TnegNpos"}
    nb = decision.net_benefit_curve(prob, cohort["T_pos"].to_numpy().astype(int))
    assert (nb["nb_model"] >= nb["nb_none"] - 1e-9).any()
    tab = decision.policy_table({"model": prob}, cohort["T_pos"].to_numpy().astype(int),
                                comparators={"ptau217": (0.9, 0.7)})
    assert "no_prescreen" in set(tab["arm"])
    row = tab[(tab["arm"] == "model") & (tab["target_sensitivity"] == 0.9)].iloc[0]
    assert row["sensitivity"] >= 0.9 - 1e-9
