"""Synthetic-data tests for lifespan_norm."""
import numpy as np
import pandas as pd
import pytest

from lifespan_norm import centiles as ce, harmonize as hz, normative as nm, site_sensitivity as ss


# ---------------------------------------------------------------------------
# harmonize
# ---------------------------------------------------------------------------
def test_age_bins_and_adapters():
    assert hz.parse_age_bin("22-25") == (24.0, True)
    assert hz.parse_age_bin("36+") == (37.0, True)
    assert hz.parse_age_bin(31) == (31.0, False)
    ya = hz.from_hcp_ya(pd.DataFrame({"Subject": [100307, 100408], "Gender": ["F", "M"], "Age": ["26-30", "36+"]}))
    assert list(ya.columns) == list(hz.SCHEMA) and ya.age.tolist() == [28.0, 37.0] and ya.age_is_binned.all()
    nda = hz.from_nda(pd.DataFrame({"src_subject_id": ["HCA6002236"], "interview_age": [420], "sex": ["F"]}),
                      dataset="HCP-A")
    assert nda.age.iloc[0] == 35.0 and nda.site.iloc[0] == "HCP-A"
    oas = hz.from_oasis3(
        pd.DataFrame({"ADRC_ADRCCLINICALDATA ID": ["OAS30001_ClinicalData_d0000", "OAS30002_ClinicalData_d0100"],
                      "ageAtEntry": [70.0, 80.0], "M/F": ["F", "M"], "cdr": [0.0, 0.5]}),
        pd.DataFrame({"MR ID": ["OAS30001_MR_d0129", "OAS30002_MR_d0090"], "Scanner": ["3.0T", "3.0T"]}))
    assert oas.group.tolist() == ["control", "impaired"]
    assert abs(oas.age.iloc[0] - (70 + 129 / 365.25)) < 1e-9
    bids = hz.from_bids_participants(pd.DataFrame({"participant_id": ["sub-01", "sub-02"], "age": [30, 41],
                                                   "sex": ["M", "F"], "diagnosis": ["CONTROL", "SCHZ"]}),
                                     dataset="ds000030", diagnosis_col="diagnosis")
    assert bids.group.tolist() == ["control", "patient"]
    allp = hz.harmonize([ya, nda, oas, bids])
    assert len(allp) == 7 and set(allp.sex) <= {"F", "M"}
    summ = hz.site_summary(allp)
    assert summ.to_numpy().sum() == (allp.group == "control").sum()


def test_freesurfer_table_and_icv():
    fs = pd.DataFrame({"lh.aparc.thickness": ["s1", "s2"], "lh_bankssts_thickness": [2.5, 2.6],
                       "lh_MeanThickness_thickness": [2.4, 2.5], "BrainSegVolNotVent": [1.0e6, 1.1e6],
                       "eTIV": [1.5e6, 1.6e6]})
    th = hz.read_freesurfer_table(fs, measure="thickness")
    assert list(th.columns) == ["lh_bankssts_thickness", "lh_MeanThickness_thickness"] and list(th.index) == ["s1", "s2"]
    vol = hz.read_freesurfer_table(fs, measure="volume")
    normed = hz.icv_normalize(vol)
    assert "eTIV" not in normed.columns and abs(normed.loc["s1", "BrainSegVolNotVent"] - 1e3 * 1.0e6 / 1.5e6) < 1e-9


# ---------------------------------------------------------------------------
# normative
# ---------------------------------------------------------------------------
def _synthetic(n=1200, sites=("A", "B", "C"), seed=0, site_shift=None):
    rng = np.random.default_rng(seed)
    age = rng.uniform(8, 90, n)
    sex = rng.choice(["F", "M"], n)
    site = rng.choice(list(sites), n)
    mu = 2.8 - 0.006 * (age - 8) - 0.00008 * (age - 40) ** 2 + 0.05 * (sex == "M")
    sigma = 0.08 + 0.001 * age
    y = mu + sigma * rng.normal(0, 1, n)
    if site_shift:
        for s, d in site_shift.items():
            y[site == s] += d
    pheno = pd.DataFrame({"subject_id": [f"s{i}" for i in range(n)], "age": age, "sex": sex, "site": site,
                          "group": "control"})
    return pheno, y


def test_spline_basis_shape():
    B = nm.natural_spline_basis(np.linspace(0, 1, 50), np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert B.shape == (50, 4) and np.isfinite(B).all()


def test_quantile_model_centiles_uniform_and_monotone():
    pheno, y = _synthetic()
    model = nm.QuantileNormativeModel(n_knots=4).fit(pheno, y)
    Q = model.predict_quantiles(pheno)
    assert (np.diff(Q, axis=1) >= 0).all()
    c = model.centile(pheno, y)
    u = ce.uniformity_check(c)
    assert u["ks"] < 0.05 and 0.03 < u["extreme_rate"] < 0.08
    # a value far below the curve gets a very low centile
    low = model.centile(pheno.iloc[[0]], np.array([y[0] - 1.0]))
    assert low[0] < 0.001


def test_gaussian_model_and_site_adaptation():
    pheno, y = _synthetic(site_shift={"C": 0.25})
    ctrl_train = pheno.site != "C"
    model = nm.GaussianNormativeModel(n_knots=4).fit(pheno[ctrl_train], y[ctrl_train.to_numpy()])
    held = (pheno.site == "C").to_numpy()
    u0 = ce.uniformity_check(model.centile(pheno[held], y[held]))
    assert u0["extreme_rate"] > 0.15 and u0["mean_centile"] > 0.7   # shifted site looks abnormal
    idx = np.flatnonzero(held)
    shift, scale = model.adapt_site(pheno.iloc[idx[:30]], y[idx[:30]], site="C")
    assert shift > 1.0
    u1 = ce.uniformity_check(model.centile(pheno.iloc[idx[30:]], y[idx[30:]]))
    assert u1["extreme_rate"] < 0.09 and 0.4 < u1["mean_centile"] < 0.6
    lo, mid, hi = model.predict_quantiles(pheno.iloc[[0]])[0]
    assert lo < mid < hi


# ---------------------------------------------------------------------------
# centiles
# ---------------------------------------------------------------------------
def test_extreme_deviations_and_contrasts():
    rng = np.random.default_rng(1)
    Z = pd.DataFrame(rng.normal(0, 1, (300, 20)), columns=[f"f{i}" for i in range(20)])
    labels = np.r_[np.zeros(200, int), np.ones(100, int)]
    Z.iloc[200:, :5] -= 1.5  # patients deviate negatively in 5 features
    counts = ce.extreme_deviations(Z)
    assert counts.loc[:199, "frac_total"].mean() < 0.08
    assert counts.loc[200:, "n_neg"].mean() > counts.loc[:199, "n_neg"].mean() + 1
    con = ce.group_contrast(Z, labels, 1)
    sig = set(con.loc[con.p_fdr < 0.05, "feature"])
    assert {"f0", "f1", "f2", "f3", "f4"} <= sig and len(sig) <= 7
    auc = ce.auc_with_ci(counts["n_total"], labels, n_boot=100)
    assert auc["auc"] > 0.7 and auc["ci_low"] > 0.6
    assert np.allclose(ce.bh_fdr([0.01, 0.02, 0.03]), [0.03, 0.03, 0.03])
    lo, hi = ce.wilson_ci(5, 100)
    assert lo < 0.05 < hi
    icc = ce.icc_test_retest(Z["f0"], Z["f0"] + rng.normal(0, 0.3, 300))
    assert icc["icc"] > 0.85


# ---------------------------------------------------------------------------
# site_sensitivity
# ---------------------------------------------------------------------------
def test_leave_one_site_out_and_reference_agreement():
    pheno, y = _synthetic(n=900, site_shift={"B": 0.3})
    feats = pd.DataFrame({"thick": y, "thick2": y + np.random.default_rng(2).normal(0, 0.05, len(y))})
    res = ss.leave_one_site_out(pheno, feats, ["thick", "thick2"], adapt_budgets=(0, 25), n_reps=2)
    assert set(res.site) == {"A", "B", "C"} and set(res.adapt_n) == {0, 25}
    b0 = res[(res.site == "B") & (res.adapt_n == 0)].extreme_rate.mean()
    b25 = res[(res.site == "B") & (res.adapt_n == 25)].extreme_rate.mean()
    assert b0 > 0.15 and b25 < 0.1
    summ = ss.summarize_loso(res)
    assert {"ks_median", "extreme_rate_median"} <= set(summ.columns)
    qm = nm.QuantileNormativeModel(n_knots=4).fit(pheno, y)
    gm = nm.GaussianNormativeModel(n_knots=4).fit(pheno, y)
    agree = ss.compare_reference_models(qm.centile(pheno, y), gm.centile(pheno, y))
    assert agree["spearman"] > 0.95 and agree["kappa_extreme"] > 0.5


def test_clinical_detection_in_vs_out():
    pheno, y = _synthetic(n=900, seed=3)
    rng = np.random.default_rng(4)
    is_c = (pheno.site == "C").to_numpy()
    patient = is_c & (rng.uniform(0, 1, len(pheno)) < 0.4)
    pheno.loc[patient, "group"] = "impaired"
    feats = pd.DataFrame({f"f{i}": y + rng.normal(0, 0.05, len(y)) - 0.3 * patient for i in range(4)})
    res = ss.clinical_detection_in_vs_out(pheno, feats, list(feats.columns), clinical_site="C",
                                          positive_group="impaired", adapt_n=20)
    assert list(res.setting) == ["in", "out"] and (res.auc > 0.7).all()
    curve = ss.adaptation_curve(pheno, feats["f0"].to_numpy(), site="A", budgets=(10, 25), n_reps=2)
    assert set(curve.adapt_n) == {0, 10, 25}
