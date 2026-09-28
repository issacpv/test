"""Synthetic-data tests for brainage_transport (no downloads required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainage_transport.freesurfer_stats import (  # noqa: E402
    parse_stats_file, load_aseg_features, load_aparc_features, build_feature_table,
)
from brainage_transport.harmonize import ComBat, batch_variance_explained  # noqa: E402
from brainage_transport.brainage import (  # noqa: E402
    BrainAgeModel, cross_validated_predictions, performance_summary, simulate_features,
)
from brainage_transport.bias_correction import (  # noqa: E402
    BiasCorrector, age_bias_slope, apply_all_methods, longitudinal_change_under_method,
    simulate_regression_to_mean,
)
from brainage_transport.longitudinal import (  # noqa: E402
    simulate_longitudinal_cohort, fit_delta_trajectory_model, within_person_change,
    icc_test_retest, align_clinical_to_scans,
)

ASEG = """# Title Segmentation Statistics
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1234567.000000, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 650000.000000, mm^3
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   4      7433     7433.0  Left-Lateral-Ventricle   27.1  12.3  5  90  85
  2  10      8000     8000.5  Left-Thalamus-Proper     90.0   8.0  50 120  70
  3  17      4200     4210.0  Left-Hippocampus         70.0   7.0  40 100  60
"""

APARC = """# Table of FreeSurfer cortical parcellation anatomical statistics
# Measure Cortex, NumVert, Number of Vertices, 120000, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 80000, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.5, mm
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
bankssts   1500  1000  2500  2.50  0.5  0.1  0.02  10  1.0
precuneus  6000  4000  9000  2.40  0.6  0.1  0.02  40  4.0
"""


def _write_stats(tmp_path: Path) -> Path:
    d = tmp_path / "sub-01" / "stats"
    d.mkdir(parents=True)
    (d / "aseg.stats").write_text(ASEG)
    (d / "lh.aparc.stats").write_text(APARC)
    (d / "rh.aparc.stats").write_text(APARC.replace("2.50", "2.60"))
    return d


def test_parse_stats_file(tmp_path):
    d = _write_stats(tmp_path)
    st = parse_stats_file(d / "aseg.stats")
    assert st.measures["eTIV"] == pytest.approx(1500000.0)
    assert list(st.table["StructName"]) == ["Left-Lateral-Ventricle", "Left-Thalamus-Proper", "Left-Hippocampus"]
    assert st.table["Volume_mm3"].dtype.kind == "f"


def test_feature_loaders_and_table(tmp_path):
    d = _write_stats(tmp_path)
    aseg = load_aseg_features(d / "aseg.stats")
    assert aseg["aseg_Left-Hippocampus"] == pytest.approx(4210.0)
    assert aseg["aseg_Left-Thalamus"] == pytest.approx(8000.5)  # FS<7 alias handled
    assert np.isnan(aseg["aseg_Right-Hippocampus"])            # missing reported as NaN
    ap = load_aparc_features(d / "lh.aparc.stats", "lh")
    assert ap["aparc_lh_precuneus_ThickAvg"] == pytest.approx(2.40)
    assert ap["aparc_lh_MeanThickness"] == pytest.approx(2.5)
    df = build_feature_table({"s1": d, "s2": d, "missing": tmp_path / "nope"}, icv_normalise=True)
    assert df.shape[0] == 3
    assert df.loc["s1", "aseg_Left-Hippocampus"] == pytest.approx(4210.0 / 1500000.0)
    assert df.loc["missing"].isna().all()
    assert df.loc["s1", "aparc_rh_bankssts_ThickAvg"] == pytest.approx(2.60)


def test_combat_removes_batch_effect_and_transports():
    X, age, scanner = simulate_features(n=300, n_features=20, scanner_effects={"A": 0.0, "B": 2.0, "C": -1.5}, seed=1)
    before = batch_variance_explained(X, scanner).mean()
    cb = ComBat().fit(X, scanner, covars=age)
    Xh = cb.transform(X, scanner, covars=age)
    after = batch_variance_explained(Xh, scanner).mean()
    assert after < 0.05 and after < before / 5
    # age signal preserved: correlation of first (age-related) feature with age is unchanged in sign/magnitude
    r_before = np.corrcoef(X[:, 0], age)[0, 1]
    r_after = np.corrcoef(Xh[:, 0], age)[0, 1]
    assert np.sign(r_before) == np.sign(r_after) and abs(r_after) >= abs(r_before) - 0.05
    # transport to new samples from a known batch without refit
    Xn, agen, scn = simulate_features(n=60, n_features=20, scanner_effects={"A": 0.0, "B": 2.0, "C": -1.5}, seed=2)
    Xnh = cb.transform(Xn, scn, covars=agen)
    assert Xnh.shape == Xn.shape
    assert batch_variance_explained(Xnh, scn).mean() < batch_variance_explained(Xn, scn).mean()
    # mean-only variant keeps scale parameters at 1
    cb2 = ComBat(mean_only=True).fit(X, scanner)
    assert np.allclose(cb2.delta_star_, 1.0)
    assert set(cb.batch_effect_table().columns) == {"batch", "feature", "gamma_star", "delta_star"}


def test_brainage_cv_and_extrapolation_flag():
    X, age, _ = simulate_features(n=300, n_features=30, noise=0.5, seed=3)
    cv = cross_validated_predictions(X, age, groups=np.arange(300) // 2, kind="ridge")
    perf = performance_summary(cv["age"], cv["pred"])
    assert perf["r"] > 0.8 and perf["mae"] < 10
    assert perf["r_age_delta"] < 0  # the classic regression-to-the-mean bias
    model = BrainAgeModel(kind="ridge").fit(X, age)
    ext = model.predict_external(X[:5], age=[10.0, 30.0, 50.0, 95.0, 60.0])
    assert list(ext["extrapolated"]) == [True, False, False, True, False]
    gbm = BrainAgeModel(kind="gbm").fit(X, age)
    assert gbm.predict(X).shape == (300,)


def test_bias_corrections():
    sim = simulate_regression_to_mean(n=3000, true_r2=0.6, seed=0)
    raw = age_bias_slope(sim["age"], sim["delta"])
    assert raw["slope"] < -0.1
    ref, tgt = sim.iloc[:1500], sim.iloc[1500:]
    for m in ("beheshti", "delange", "smith"):
        bc = BiasCorrector(method=m).fit(ref["age"], ref["pred"])
        d = bc.corrected_delta(tgt["age"], tgt["pred"])
        assert abs(age_bias_slope(tgt["age"], d)["slope"]) < 0.05
    cole = BiasCorrector(method="cole").fit(ref["age"], ref["pred"])
    d_cole = cole.corrected_delta(tgt["age"], tgt["pred"])
    assert abs(age_bias_slope(tgt["age"], d_cole)["slope"]) < 0.05
    assert d_cole.std() > BiasCorrector(method="beheshti").fit(ref["age"], ref["pred"]).corrected_delta(tgt["age"], tgt["pred"]).std()
    beh = BiasCorrector(method="beheshti").fit(ref["age"], ref["pred"])
    dl = BiasCorrector(method="delange").fit(ref["age"], ref["pred"])
    assert np.allclose(beh.corrected_delta(tgt["age"], tgt["pred"]), dl.corrected_delta(tgt["age"], tgt["pred"]))
    tab = apply_all_methods(ref["age"], ref["pred"], tgt["age"], tgt["pred"])
    assert set(tab.columns) >= {"age", "none", "beheshti", "cole", "smith"}
    # longitudinal implication: same raw change, different corrected change
    a_beh = raw["slope"]
    dd = longitudinal_change_under_method([1.0], [2.0], a_beh, "beheshti")
    dc = longitudinal_change_under_method([1.0], [2.0], 1 + a_beh, "cole")
    assert not np.isclose(dd[0], dc[0])


def test_longitudinal_pipeline():
    df = simulate_longitudinal_cohort(n_subjects=120, visits=3, converter_accel=1.5, seed=1)
    res = fit_delta_trajectory_model(df, group_col="converter", covariates=("age_baseline", "sex"))
    assert res.params["years:converter"] > 0.5
    wp = within_person_change(df)
    assert (wp["n_visits"] >= 2).all()
    conv_ids = df.loc[df["converter"] == 1, "subject"].unique()
    assert wp[wp["subject"].isin(conv_ids)]["slope"].mean() > wp[~wp["subject"].isin(conv_ids)]["slope"].mean()
    rng = np.random.default_rng(0)
    a = rng.normal(0, 5, 80)
    icc = icc_test_retest(a, a + rng.normal(0, 1, 80))
    assert 0.8 < icc["icc21"] <= 1.0
    scans = pd.DataFrame({"subject": ["s1", "s1"], "days": [0, 800], "delta": [1.0, 2.0]})
    clin = pd.DataFrame({"subject": ["s1", "s1"], "days": [30, 1500], "cdr": [0.0, 0.5], "mmse": [30, 27]})
    al = align_clinical_to_scans(scans, clin, tolerance_days=365)
    assert al.loc[0, "cdr"] == 0.0 and np.isnan(al.loc[1, "cdr"])
