"""Synthetic-data tests for amyloid_mri (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from amyloid_mri import decision_curve as dc
from amyloid_mri import models, oasis_tables as ot, roi_features as rf


# ---------------------------------------------------------------------------
# oasis_tables
# ---------------------------------------------------------------------------
def test_parse_oasis_id():
    assert ot.parse_oasis_id("OAS30001_MR_d0129") == ("OAS30001", "MR", 129)
    assert ot.parse_oasis_id("OAS30001_AV45_PUPTIMECOURSE_d2430") == ("OAS30001", "AV45", 2430)
    assert ot.parse_oasis_id("OAS30001_ClinicalData_d0000") == ("OAS30001", "ClinicalData", 0)
    with pytest.raises(ValueError):
        ot.parse_oasis_id("sub-01")


def test_apoe_and_sex_parsing():
    assert ot.apoe_e4_count(34) == 1
    assert ot.apoe_e4_count("44") == 2
    assert ot.apoe_e4_count("E3/E3") == 0
    assert np.isnan(ot.apoe_e4_count(None))
    assert ot.sex_to_male("M") == 1.0 and ot.sex_to_male("f") == 0.0


def _synthetic_tables():
    clinical = pd.DataFrame({
        "ADRC_ADRCCLINICALDATA ID": ["OAS30001_ClinicalData_d0000", "OAS30001_ClinicalData_d0400",
                                     "OAS30002_ClinicalData_d0000"],
        "Subject": ["OAS30001", "OAS30001", "OAS30002"],
        "ageAtEntry": [70.0, 70.0, 65.0], "M/F": ["F", "F", "M"], "apoe": [34, 34, 33],
        "mmse": [30, 29, 27], "cdr": [0.0, 0.0, 0.5],
    })
    pup = pd.DataFrame({
        "PUP_PUPTIMECOURSEDATA ID": ["OAS30001_PIB_PUPTIMECOURSE_d0100", "OAS30001_AV45_PUPTIMECOURSE_d0900",
                                     "OAS30002_AV45_PUPTIMECOURSE_d0010"],
        "tracer": ["PIB", "AV45", "AV45"],
        "Centiloid_fSUVR_TOT_CORTMEAN": [10.0, 40.0, 25.0],
    })
    mr = pd.DataFrame({"MR ID": ["OAS30001_MR_d0129", "OAS30001_MR_d0800", "OAS30002_MR_d0005",
                                 "OAS30003_MR_d0000"],
                       "Scanner": ["3.0T", "3.0T", "1.5T", "3.0T"]})
    return ot.load_clinical(clinical), ot.load_pup(pup), ot.load_mr_sessions(mr)


def test_build_subject_table_matching_and_labels():
    clinical, pup, mr = _synthetic_tables()
    t = ot.build_subject_table(clinical, pup, mr, max_pet_gap=365, max_clinical_gap=180, one_per_subject=False)
    # OAS30003 has no PET -> dropped; OAS30001 has two labelled sessions; OAS30002 one
    assert set(t.subject) == {"OAS30001", "OAS30002"}
    r1 = t[t.mr_id == "OAS30001_MR_d0129"].iloc[0]
    assert r1.tracer == "PIB" and r1.centiloid == 10.0 and r1.amyloid_pos == 0.0
    assert r1.cog_status == "CN" and r1.apoe_e4_count == 1 and r1.sex_male == 0.0
    assert abs(r1.age - (70 + 129 / 365.25)) < 1e-6
    r2 = t[t.mr_id == "OAS30001_MR_d0800"].iloc[0]
    assert r2.tracer == "AV45" and r2.amyloid_pos == 1.0 and r2.pet_gap_days == 100
    r3 = t[t.subject == "OAS30002"].iloc[0]
    assert r3.amyloid_pos == 1.0 and r3.cog_status == "CI"  # 25 CL >= 20.6 (AV45)
    one = ot.build_subject_table(clinical, pup, mr, one_per_subject=True)
    assert len(one) == 2 and one[one.subject == "OAS30001"].mr_id.iloc[0] == "OAS30001_MR_d0129"
    grey = ot.build_subject_table(clinical, pup, mr, one_per_subject=False, grey_zone=(20, 30))
    assert "OAS30002" not in set(grey.subject)


# ---------------------------------------------------------------------------
# roi_features
# ---------------------------------------------------------------------------
def test_parse_stats_files(tmp_path):
    aseg = tmp_path / "aseg.stats"
    aseg.write_text(
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3\n"
        "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange\n"
        "  1  17  4000  4100.5  Left-Hippocampus  80.0 6.0 40 110 70\n"
        "  2  53  3900  3950.0  Right-Hippocampus 80.0 6.0 40 110 70\n")
    d = rf.parse_aseg_stats(aseg)
    assert d["eTIV"] == 1500000.0 and d["Left-Hippocampus"] == 4100.5
    aparc = tmp_path / "lh.aparc.stats"
    aparc.write_text(
        "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
        "precuneus 5000 3000 9000 2.35 0.5 0.1 0.02 40 4\n")
    a = rf.parse_aparc_stats(aparc)
    assert a["lh_precuneus_ThickAvg"] == 2.35


def test_residualize_removes_covariate_effect():
    rng = np.random.default_rng(0)
    n = 300
    age = rng.uniform(55, 90, n)
    df = pd.DataFrame({"age": age, "hipp": 5000 - 30 * age + rng.normal(0, 50, n)})
    res = rf.residualize(df, ["hipp"], ["age"])
    assert abs(np.corrcoef(res["hipp"], age)[0, 1]) < 0.05


def test_combat_reduces_batch_effect_and_transforms_new_data():
    rng = np.random.default_rng(1)
    n, p = 240, 6
    batch = np.repeat(["A", "B", "C"], n // 3)
    X = rng.normal(0, 1, (n, p))
    X[batch == "B"] += 1.5
    X[batch == "C"] *= 2.0
    before = rf.batch_effect_size(X, batch).mean()
    cb = rf.ComBat().fit(X[: n // 2 * 2 : 2], batch[: n // 2 * 2 : 2])  # fit on half the rows
    Xh = cb.transform(X, batch)
    after = rf.batch_effect_size(Xh, batch).mean()
    assert after < 0.2 * before
    with pytest.raises(KeyError):
        cb.transform(X[:2], np.array(["Z", "Z"]))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _synthetic_cohort(n_subjects=160, sessions=2, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subjects), sessions)
    age = np.repeat(rng.uniform(60, 85, n_subjects), sessions)
    y_subj = rng.binomial(1, 0.4, n_subjects)
    y = np.repeat(y_subj, sessions)
    roi = rng.normal(0, 1, (len(y), 8))
    roi[:, 0] -= 1.2 * y  # informative feature
    roi[:, 1] -= 0.02 * age  # age-related feature
    return roi, y, subj, age


def test_nested_cv_grouped_and_calibrated():
    X, y, groups, _ = _synthetic_cohort()
    oof = models.nested_cv_predict(X, y, groups, kind="roi_enet", outer_splits=4, inner_splits=2)
    assert len(oof) == len(y)
    # no subject appears in more than one outer fold
    folds_per_subject = oof.assign(g=groups[oof.row]).groupby("g").fold.nunique()
    assert (folds_per_subject == 1).all()
    r = models.evaluate(oof.y, oof.p_cal, groups[oof.row], n_boot=100)
    assert r["auc"] > 0.75
    assert 0.5 < r["cal_slope"] < 1.8


def test_paired_auc_difference_detects_gain():
    X, y, groups, age = _synthetic_cohort()
    p_ref = -age + np.random.default_rng(2).normal(0, 3, len(y))  # near-useless baseline
    p_new = -X[:, 0]
    r = models.paired_auc_difference(y, p_new, p_ref, groups, n_boot=200)
    assert r["delta_auc"] > 0.1 and r["ci_low"] > 0 and r["p_value"] < 0.05


def test_age_matching_balances_age():
    rng = np.random.default_rng(3)
    age = rng.uniform(60, 90, 400)
    y = (rng.uniform(0, 1, 400) < (age - 55) / 40).astype(int)
    idx = models.age_matched_subset(age, y, caliper=1.5)
    assert len(idx) > 50
    assert abs(age[idx][y[idx] == 1].mean() - age[idx][y[idx] == 0].mean()) < 1.0


# ---------------------------------------------------------------------------
# decision_curve
# ---------------------------------------------------------------------------
def test_net_benefit_and_policies():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.3, 2000)
    p = np.clip(0.3 + 0.35 * (2 * y - 1) + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
    nb = dc.net_benefit(y, p, thresholds=[0.1, 0.3, 0.5])
    assert np.allclose(nb.nb_all.iloc[0], y.mean() - (1 - y.mean()) * 0.1 / 0.9)
    assert (nb.nb_model >= nb.nb_all - 1e-9).all()
    t = dc.threshold_for_sensitivity(y, p, 0.9)
    pol = dc.screening_policy(y, p, t)
    assert pol["sensitivity"] >= 0.9 and pol["pet_avoided_fraction"] > 0.3
    ref = dc.policy_from_sens_spec(0.3, 0.9, 0.7)
    assert abs(ref["pet_per_1000"] - 1000 * (0.3 * 0.9 + 0.7 * 0.3)) < 1e-9
    table = dc.compare_policies(y, {"model": p}, 0.9, cost_pet=4000,
                                prescreen_costs={"model": 0.0, "ptau217": 150.0},
                                comparators={"ptau217": (0.9, 0.7, 150.0)})
    assert list(table.policy) == ["no_prescreen", "model", "ptau217"]
    assert abs(table.loc[0, "pet_per_1000"] - 1000) < 1e-6
    assert (table.net_saving_per_1000.iloc[1:] > 0).all()
