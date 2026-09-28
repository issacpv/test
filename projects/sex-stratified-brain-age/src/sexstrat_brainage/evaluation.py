"""Evaluation metrics for the pooled-vs-stratified audit.

All functions take the DataFrame produced by ``models.cross_validated_delta``
(columns ``age, sex, group, pred_age, delta, raw_delta``), optionally joined
with outcome columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def mae_by_sex(res: pd.DataFrame) -> dict[str, float]:
    """MAE overall and per sex, plus the absolute sex gap."""
    err = (res["pred_age"] - res["age"]).abs()
    m_f, m_m = err[res["sex"] == 0].mean(), err[res["sex"] == 1].mean()
    return {"mae_all": float(err.mean()), "mae_female": float(m_f), "mae_male": float(m_m), "sex_gap": float(abs(m_f - m_m))}


def bootstrap_sex_gap(res: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Subject-level bootstrap CI for the sex gap in MAE."""
    rng = np.random.default_rng(seed)
    groups = res["group"].to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    gaps = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        sub = res.iloc[idx]
        if (sub["sex"] == 0).sum() == 0 or (sub["sex"] == 1).sum() == 0:
            continue
        gaps.append(mae_by_sex(sub)["sex_gap"])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"sex_gap": mae_by_sex(res)["sex_gap"], "ci_low": float(lo), "ci_high": float(hi)}


def spurious_sex_effect(res: pd.DataFrame, column: str = "delta") -> dict[str, float]:
    """OLS delta ~ sex + age: the sex coefficient is the 'sex difference in brain age' a pooled analysis would report."""
    d = res[[column, "sex", "age"]].dropna()
    fit = smf.ols(f"{column} ~ sex + age", data=d).fit(cov_type="HC3")
    return {"beta_sex": float(fit.params["sex"]), "se_sex": float(fit.bse["sex"]), "p_sex": float(fit.pvalues["sex"]),
            "beta_age": float(fit.params["age"])}


def outcome_association(res: pd.DataFrame, outcome: str, covariates: tuple[str, ...] = ("age", "sex"),
                        cohort_col: str | None = None) -> dict[str, float]:
    """Standardized association of delta with an outcome, controlling covariates.

    With ``cohort_col`` a random-intercept mixed model (statsmodels MixedLM) is used;
    otherwise OLS with HC3 errors. Returns the standardized beta of delta and its CI.
    """
    d = res.dropna(subset=[outcome, "delta"]).copy()
    d["_y"] = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=1)
    d["_d"] = (d["delta"] - d["delta"].mean()) / d["delta"].std(ddof=1)
    rhs = " + ".join(["_d", *covariates])
    if cohort_col:
        fit = smf.mixedlm(f"_y ~ {rhs}", d, groups=d[cohort_col]).fit(reml=True)
        ci = fit.conf_int().loc["_d"]
    else:
        fit = smf.ols(f"_y ~ {rhs}", d).fit(cov_type="HC3")
        ci = fit.conf_int().loc["_d"]
    return {"beta_std": float(fit.params["_d"]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(fit.pvalues["_d"]), "n": int(len(d))}


def association_by_sex(res: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Delta-outcome association within each sex (age-adjusted) and the interaction test."""
    rows = []
    for s, name in ((0, "female"), (1, "male")):
        sub = res[res["sex"] == s]
        rows.append({"sex": name, **outcome_association(sub, outcome, covariates=("age",))})
    d = res.dropna(subset=[outcome]).copy()
    fit = smf.ols(f"{outcome} ~ delta * sex + age", d).fit(cov_type="HC3")
    rows.append({"sex": "interaction", "beta_std": float(fit.params["delta:sex"]),
                 "ci_low": float(fit.conf_int().loc["delta:sex", 0]), "ci_high": float(fit.conf_int().loc["delta:sex", 1]),
                 "p": float(fit.pvalues["delta:sex"]), "n": int(len(d))})
    return pd.DataFrame(rows)


def icc_2_1(measurements: np.ndarray) -> float:
    """ICC(2,1) (two-way random, absolute agreement) for an (n_subjects × k_sessions) array."""
    Y = np.asarray(measurements, dtype=float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def retest_reliability(res: pd.DataFrame, session_col: str = "session") -> dict[str, float]:
    """ICC(2,1) of delta across sessions for subjects with exactly two sessions, by sex."""
    out = {}
    for label, mask in (("all", np.ones(len(res), bool)), ("female", res["sex"] == 0), ("male", res["sex"] == 1)):
        sub = res[mask]
        wide = sub.pivot_table(index="group", columns=session_col, values="delta")
        wide = wide.dropna()
        out[f"icc_{label}"] = icc_2_1(wide.to_numpy()[:, :2]) if len(wide) >= 3 and wide.shape[1] >= 2 else float("nan")
    return out


def transport_mae(train_res_model, X_test: pd.DataFrame, age_test: np.ndarray, sex_test: np.ndarray) -> dict[str, float]:
    """MAE by sex on an unseen cohort using a fitted ``BrainAgeEstimator``."""
    pred = train_res_model.predict(X_test, age_test, sex_test)
    err = np.abs(pred - np.asarray(age_test))
    sex_test = np.asarray(sex_test)
    return {"mae_all": float(err.mean()), "mae_female": float(err[sex_test == 0].mean()),
            "mae_male": float(err[sex_test == 1].mean())}


def permutation_interaction_p(res: pd.DataFrame, outcome: str, n_perm: int = 500, seed: int = 0) -> float:
    """Permutation p-value for delta × sex interaction, permuting sex within age-decade strata."""
    rng = np.random.default_rng(seed)
    d = res.dropna(subset=[outcome]).copy()

    def stat(df: pd.DataFrame) -> float:
        return float(smf.ols(f"{outcome} ~ delta * sex + age", df).fit().params["delta:sex"])

    obs = abs(stat(d))
    strata = (d["age"] // 10).astype(int)
    count = 0
    for _ in range(n_perm):
        p = d.copy()
        p["sex"] = p.groupby(strata)["sex"].transform(lambda s: rng.permutation(s.to_numpy()))
        if abs(stat(p)) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def summarize_strategies(results: dict[str, pd.DataFrame], outcome: str | None = None) -> pd.DataFrame:
    """One row per strategy label with MAE, sex gap, spurious sex effect and optional outcome association."""
    rows = []
    for label, res in results.items():
        row = {"strategy": label, **mae_by_sex(res), **{f"spur_{k}": v for k, v in spurious_sex_effect(res).items()}}
        if outcome and outcome in res:
            row.update({f"out_{k}": v for k, v in outcome_association(res, outcome).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def add_constant(X: np.ndarray) -> np.ndarray:  # small helper re-exported for notebooks
    return sm.add_constant(X)
