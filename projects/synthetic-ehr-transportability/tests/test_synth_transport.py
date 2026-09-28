"""Synthetic-data tests for synth_transport (no real EHR data, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synth_transport import fidelity, generators, ontology, transport  # noqa: E402


def _toy_table(n: int, rng: np.random.Generator, shift: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Correlated numeric + binary features with a logistic outcome."""
    z = rng.standard_normal((n, 3))
    x1 = z[:, 0]
    x2 = 0.7 * z[:, 0] + 0.7 * z[:, 1] + shift
    x3 = z[:, 2]
    b = (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(float)
    logits = 1.2 * x1 - 0.8 * x2 + 0.5 * b - 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "b": b}), y


# ------------------------------------------------------------- ontology
def test_ontology_spec_is_complete():
    assert ontology.verify_spec() == []
    names = ontology.feature_names()
    assert "lab_creatinine" in names and "miss_creatinine" in names and "cond_diabetes" in names
    assert "n_creatinine" not in ontology.feature_names(["demographics", "labs"])


def test_synthea_loader_on_mini_export(tmp_path):
    pd.DataFrame(
        [
            {"Id": "p1", "BIRTHDATE": "1950-01-01", "DEATHDATE": "2020-03-05", "GENDER": "M"},
            {"Id": "p2", "BIRTHDATE": "1980-06-01", "DEATHDATE": "", "GENDER": "F"},
            {"Id": "p3", "BIRTHDATE": "2015-06-01", "DEATHDATE": "", "GENDER": "F"},  # child: excluded
        ]
    ).to_csv(tmp_path / "patients.csv", index=False)
    pd.DataFrame(
        [
            {"Id": "e0", "START": "2019-12-01T00:00:00Z", "STOP": "2019-12-01T01:00:00Z", "PATIENT": "p1", "ENCOUNTERCLASS": "ambulatory"},
            {"Id": "e1", "START": "2020-03-01T10:00:00Z", "STOP": "2020-03-04T10:00:00Z", "PATIENT": "p1", "ENCOUNTERCLASS": "inpatient"},
            {"Id": "e1e", "START": "2020-03-01T02:00:00Z", "STOP": "2020-03-01T05:00:00Z", "PATIENT": "p1", "ENCOUNTERCLASS": "emergency"},
            {"Id": "e2", "START": "2021-01-01T00:00:00Z", "STOP": "2021-01-03T00:00:00Z", "PATIENT": "p2", "ENCOUNTERCLASS": "inpatient"},
            {"Id": "e3", "START": "2021-01-20T00:00:00Z", "STOP": "2021-01-22T00:00:00Z", "PATIENT": "p2", "ENCOUNTERCLASS": "inpatient"},
            {"Id": "e4", "START": "2021-02-01T00:00:00Z", "STOP": "2021-02-02T00:00:00Z", "PATIENT": "p3", "ENCOUNTERCLASS": "inpatient"},
        ]
    ).to_csv(tmp_path / "encounters.csv", index=False)
    pd.DataFrame(
        [
            {"START": "2010-01-01", "STOP": "", "PATIENT": "p1", "ENCOUNTER": "e0", "CODE": "44054006", "DESCRIPTION": "Diabetes"},
            {"START": "2010-01-01", "STOP": "2015-01-01", "PATIENT": "p2", "ENCOUNTER": "e2", "CODE": "59621000", "DESCRIPTION": "Hypertension (resolved)"},
        ]
    ).to_csv(tmp_path / "conditions.csv", index=False)
    pd.DataFrame(
        [
            {"DATE": "2020-03-01T11:00:00Z", "PATIENT": "p1", "ENCOUNTER": "e1", "CODE": "2160-0", "VALUE": "1.4", "UNITS": "mg/dL", "TYPE": "numeric"},
            {"DATE": "2020-03-02T11:00:00Z", "PATIENT": "p1", "ENCOUNTER": "e1", "CODE": "2160-0", "VALUE": "1.9", "UNITS": "mg/dL", "TYPE": "numeric"},
            {"DATE": "2021-01-01T01:00:00Z", "PATIENT": "p2", "ENCOUNTER": "e2", "CODE": "2951-2", "VALUE": "138", "UNITS": "mEq/L", "TYPE": "numeric"},
        ]
    ).to_csv(tmp_path / "observations.csv", index=False)

    df = ontology.load_synthea_encounters(tmp_path)
    assert list(df["encounter_id"]) == ["e1", "e2", "e3"]
    r1 = df.set_index("encounter_id").loc["e1"]
    assert r1["sex_male"] == 1 and r1["emergency"] == 1 and r1["prior_encounters_365d"] == 2
    assert r1["cond_diabetes"] == 1 and r1["lab_creatinine"] == 1.4 and r1["n_creatinine"] == 2 and r1["miss_creatinine"] == 0
    assert r1["mortality"] == 1
    r2 = df.set_index("encounter_id").loc["e2"]
    assert r2["cond_hypertension"] == 0 and r2["readmit_30d"] == 1 and r2["miss_creatinine"] == 1 and r2["lab_sodium"] == 138
    X, med = ontology.build_feature_matrix(df, blocks=["demographics", "labs"])
    assert not X.isna().any().any() and "lab_creatinine" in X.columns and "n_creatinine" not in X.columns
    X2, _ = ontology.build_feature_matrix(df, blocks=["labs"], impute=med)
    assert X2["lab_potassium"].iloc[0] == 0.0  # all-missing lab imputed with 0 when no median exists


# ----------------------------------------------------------- generators
def test_copula_preserves_dependence_and_independent_destroys_it():
    rng = np.random.default_rng(0)
    real, _ = _toy_table(4000, rng)
    cop = generators.GaussianCopulaGenerator().fit(real)
    syn = cop.sample(4000, rng)
    assert list(syn.columns) == list(real.columns) and set(syn["b"].unique()).issubset({0.0, 1.0})
    assert abs(syn["b"].mean() - real["b"].mean()) < 0.03
    r_real = real["x1"].corr(real["x2"])
    assert abs(syn["x1"].corr(syn["x2"]) - r_real) < 0.08
    ind = generators.IndependentMarginalsGenerator().fit(real).sample(4000, rng)
    assert abs(ind["x1"].corr(ind["x2"])) < 0.08
    boot = generators.BootstrapJitterGenerator().fit(real).sample(100, rng)
    assert boot.shape == (100, 4)
    assert isinstance(generators.make_generator("copula"), generators.GaussianCopulaGenerator)
    with pytest.raises(ValueError):
        generators.make_generator("nope")


# ------------------------------------------------------------- fidelity
def test_fidelity_metrics_order_generators():
    rng = np.random.default_rng(1)
    real, _ = _toy_table(3000, rng)
    hold, _ = _toy_table(1000, rng)
    cop = generators.GaussianCopulaGenerator().fit(real).sample(3000, rng)
    ind = generators.IndependentMarginalsGenerator().fit(real).sample(3000, rng)
    assert fidelity.correlation_distance(real, cop) < fidelity.correlation_distance(real, ind)
    p_hold = fidelity.pmse(real, hold, rng=rng)["ratio"]
    p_cop = fidelity.pmse(real, cop, rng=rng)["ratio"]
    p_ind = fidelity.pmse(real, ind, rng=rng)["ratio"]
    assert p_hold < 3.0 and p_ind > p_cop  # linear pMSE: near-null for holdout; copula better than independent
    assert fidelity.mmd_rbf(real, cop, rng=rng) < fidelity.mmd_rbf(real, ind, rng=rng)
    marg = fidelity.marginal_fidelity(real, cop)
    assert set(marg["metric"]) == {"ks", "tv"} and marg["distance"].max() < 0.1
    d = fidelity.dcr(real, cop, hold, rng=rng)
    assert d["syn_p5"] > 0 and "holdout_p5" in d
    rep = fidelity.fidelity_report(real, cop, hold, rng=rng)
    assert "pmse_ratio" in rep and np.isnan(rep["missingness_gap"])


# ------------------------------------------------------------ transport
def test_tstr_matrix_and_decomposition_identity():
    rng = np.random.default_rng(2)
    real_src, y_src = _toy_table(3000, rng)
    hold_src, yh_src = _toy_table(1500, rng)
    tgt, y_tgt = _toy_table(1500, rng, shift=1.0)
    syn_src = generators.GaussianCopulaGenerator().fit(real_src.assign(y=y_src)).sample(3000, rng)
    y_syn = syn_src.pop("y").to_numpy(dtype=int)
    mat = transport.tstr_matrix({"real_src": (real_src, y_src), "syn_src": (syn_src, y_syn)}, {"src": (hold_src, yh_src), "tgt": (tgt, y_tgt)})
    assert len(mat) == 4 and mat["auroc"].between(0.5, 1).all()
    dec = transport.gap_decomposition(mat, "auroc")
    assert dec["amplification"] == pytest.approx(dec["site_gap_syn"] - dec["site_gap_real"])
    assert dec["synthetic_gap_src"] == pytest.approx(dec["site_gap_real"] - dec["site_gap_syn"] + dec["synthetic_gap_tgt"])
    dec_b = transport.gap_decomposition(mat, "brier")  # lower-is-better metric flips sign internally
    rs = float(mat[(mat.train == "real_src") & (mat.test == "src")]["brier"].iloc[0])
    ss = float(mat[(mat.train == "syn_src") & (mat.test == "src")]["brier"].iloc[0])
    assert dec_b["synthetic_gap_src"] == pytest.approx(ss - rs)
    models = {"real_src": transport.fit_model(real_src, y_src), "syn_src": transport.fit_model(syn_src, y_syn)}
    boot = transport.bootstrap_decomposition(models, {"src": (hold_src, yh_src), "tgt": (tgt, y_tgt)}, n_boot=20, rng=rng)
    assert set(boot["component"]) == {"synthetic_gap_src", "synthetic_gap_tgt", "site_gap_real", "site_gap_syn", "amplification"}
    assert (boot["ci_low"] <= boot["estimate"] + 1e-9).all() and (boot["estimate"] <= boot["ci_high"] + 1e-9).all()
    sub = transport.subgroup_metrics(models["real_src"], hold_src, yh_src, groups=hold_src["b"].to_numpy())
    assert len(sub) == 2 and "gap_auroc" in sub


def test_site_gap_is_zero_between_random_halves():
    rng = np.random.default_rng(3)
    real, y = _toy_table(6000, rng)
    half = np.arange(6000) % 2 == 0
    train, ytr = real[: 3000], y[:3000]
    a, ya = real[3000:][half[3000:]], y[3000:][half[3000:]]
    b, yb = real[3000:][~half[3000:]], y[3000:][~half[3000:]]
    mat = transport.tstr_matrix({"real_src": (train, ytr), "syn_src": (train, ytr)}, {"src": (a, ya), "tgt": (b, yb)})
    dec = transport.gap_decomposition(mat, "auroc")
    assert abs(dec["site_gap_real"]) < 0.05 and dec["amplification"] == pytest.approx(0.0, abs=1e-12)
