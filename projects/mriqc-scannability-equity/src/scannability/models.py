"""Exclusion models, scannability index, elasticity and representation shift.

The main model is a binomial GLM
    exclusion ~ bs(age, df=4) + sex + group + C(dataset)
with dataset-clustered robust standard errors. A mixed-model cross-check with a
random dataset intercept is available via ``fit_mixed``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import entropy

GROUP_LEVELS = ("control", "psychiatric", "neurological", "neurodevelopmental", "other_clinical", "unknown")


def _prepare(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = df.dropna(subset=["age", outcome]).copy()
    d = d[d["sex"].isin(["F", "M"])]
    d["group"] = pd.Categorical(d["group"], categories=[g for g in GROUP_LEVELS if g in set(d["group"])])
    d["sex"] = pd.Categorical(d["sex"], categories=["F", "M"])
    d["dataset"] = d["dataset"].astype(str)
    return d


def fit_exclusion_glm(df: pd.DataFrame, outcome: str, age_df: int = 4, dataset_effects: bool = True):
    """Fit the binomial GLM with dataset fixed effects and dataset-clustered SEs."""
    d = _prepare(df, outcome)
    rhs = f"bs(age, df={age_df}) + sex + group"
    if dataset_effects and d["dataset"].nunique() > 1:
        rhs += " + C(dataset)"
    model = smf.glm(f"Q('{outcome}') ~ {rhs}", data=d, family=sm.families.Binomial())
    groups = pd.factorize(d["dataset"])[0]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_mixed(df: pd.DataFrame, outcome: str, age_df: int = 4):
    """Random-intercept (dataset) logistic model via BinomialBayesMixedGLM (variational)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = _prepare(df, outcome)
    model = BinomialBayesMixedGLM.from_formula(f"Q('{outcome}') ~ bs(age, df={age_df}) + sex + group",
                                               {"dataset": "0 + C(dataset)"}, d)
    return model.fit_vb()


def odds_ratios(fit, terms: tuple[str, ...] = ("sex[T.M]", "group[T.psychiatric]", "group[T.neurological]",
                                               "group[T.neurodevelopmental]", "group[T.other_clinical]")) -> pd.DataFrame:
    """OR with 95% CI for named coefficients (those absent from the fit are skipped)."""
    rows = []
    ci = fit.conf_int()
    for t in terms:
        if t in fit.params.index:
            rows.append({"term": t, "OR": float(np.exp(fit.params[t])), "ci_low": float(np.exp(ci.loc[t, 0])),
                         "ci_high": float(np.exp(ci.loc[t, 1])), "p": float(fit.pvalues[t])})
    return pd.DataFrame(rows)


def age_curve(fit, df: pd.DataFrame, outcome: str, ages: np.ndarray | None = None,
              sex: str = "F", group: str = "control", dataset: str | None = None) -> pd.DataFrame:
    """Predicted exclusion probability across age at a reference dataset (the 'scannability' curve)."""
    d = _prepare(df, outcome)
    ages = np.linspace(d["age"].min(), d["age"].max(), 50) if ages is None else np.asarray(ages)
    ref_ds = dataset or d["dataset"].mode().iloc[0]
    new = pd.DataFrame({"age": ages, "sex": pd.Categorical([sex] * len(ages), categories=["F", "M"]),
                        "group": pd.Categorical([group] * len(ages), categories=d["group"].cat.categories),
                        "dataset": ref_ds})
    p = fit.predict(new)
    return pd.DataFrame({"age": ages, "p_excluded": np.asarray(p), "scannability": 1 - np.asarray(p)})


def scannability_index(fit, df: pd.DataFrame, outcome: str, dataset: str | None = None) -> pd.Series:
    """Per-participant predicted pass probability at a reference dataset (removes dataset design)."""
    d = _prepare(df, outcome)
    ref_ds = dataset or d["dataset"].mode().iloc[0]
    new = d[["age", "sex", "group"]].copy()
    new["dataset"] = ref_ds
    return pd.Series(1 - np.asarray(fit.predict(new)), index=d.index, name="scannability")


def exclusion_elasticity(df: pd.DataFrame, excl_cols: list[str], thresholds: list[float],
                         by: str = "group") -> pd.DataFrame:
    """Exclusion fraction per group across a threshold sweep and its slope (elasticity).

    ``excl_cols[i]`` must be the exclusion flag produced with ``thresholds[i]``.
    Elasticity is the OLS slope of exclusion fraction on log(threshold).
    """
    rows = []
    for level, sub in df.groupby(by, observed=True):
        fracs = [float(sub[c].mean()) for c in excl_cols]
        lt = np.log(np.asarray(thresholds, dtype=float))
        slope = float(np.polyfit(lt, fracs, 1)[0]) if len(thresholds) > 1 else np.nan
        rows.append({by: level, "n": len(sub), **{f"excl@{t}": f for t, f in zip(thresholds, fracs)},
                     "elasticity": slope, "range": max(fracs) - min(fracs)})
    return pd.DataFrame(rows)


def representation_shift(df: pd.DataFrame, excl_col: str, age_bins: np.ndarray | None = None) -> dict[str, float]:
    """Compare the acquired vs QC-passed corpus: age quantiles, tail shares, group shares, KL divergence."""
    age_bins = np.arange(0, 101, 5) if age_bins is None else age_bins
    acquired = df.dropna(subset=["age"])
    passed = acquired[acquired[excl_col] == 0]
    h_a, _ = np.histogram(acquired["age"], bins=age_bins)
    h_p, _ = np.histogram(passed["age"], bins=age_bins)
    p_a = (h_a + 1e-9) / (h_a.sum() + 1e-9 * len(h_a))
    p_p = (h_p + 1e-9) / (h_p.sum() + 1e-9 * len(h_p))
    out = {
        "n_acquired": int(len(acquired)), "n_passed": int(len(passed)), "pass_rate": float(len(passed) / max(1, len(acquired))),
        "age_iqr_acquired": float(acquired["age"].quantile(0.75) - acquired["age"].quantile(0.25)),
        "age_iqr_passed": float(passed["age"].quantile(0.75) - passed["age"].quantile(0.25)),
        "share_under10_acquired": float((acquired["age"] < 10).mean()), "share_under10_passed": float((passed["age"] < 10).mean()),
        "share_over65_acquired": float((acquired["age"] > 65).mean()), "share_over65_passed": float((passed["age"] > 65).mean()),
        "kl_passed_vs_acquired": float(entropy(p_p, p_a)),
    }
    if "group" in df.columns:
        for g in ("control", "neurological", "psychiatric", "neurodevelopmental"):
            out[f"share_{g}_acquired"] = float((acquired["group"] == g).mean())
            out[f"share_{g}_passed"] = float((passed["group"] == g).mean())
    return out


def biology_adjust_iqms(df: pd.DataFrame, iqm_cols: list[str], age_df: int = 4,
                        reference_mask: pd.Series | None = None) -> pd.DataFrame:
    """Residualize IQMs on an age spline within dataset, fitted on reference rows (QC-passed controls).

    Returns a copy with ``<iqm>_bioadj`` columns (residual + reference mean) so that
    the original rules can be re-applied to biology-adjusted values.
    """
    from patsy import dmatrix

    out = df.copy()
    ref = np.ones(len(df), dtype=bool) if reference_mask is None else np.asarray(reference_mask, dtype=bool)
    for ds, idx in df.groupby("dataset").groups.items():
        sub = df.loc[idx]
        m = ref[df.index.get_indexer(idx)] & sub["age"].notna().to_numpy()
        if m.sum() < age_df + 5:
            for c in iqm_cols:
                out.loc[idx, f"{c}_bioadj"] = sub[c]
            continue
        ages = sub["age"].to_numpy(dtype=float)
        knots_frame = pd.DataFrame({"age": ages})
        X = np.asarray(dmatrix(f"bs(age, df={age_df}, include_intercept=True)", knots_frame, return_type="dataframe"))
        for c in iqm_cols:
            if c not in sub.columns:
                continue
            y = sub[c].to_numpy(dtype=float)
            ok = m & ~np.isnan(y)
            if ok.sum() < age_df + 5:
                out.loc[idx, f"{c}_bioadj"] = y
                continue
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            out.loc[idx, f"{c}_bioadj"] = y - X @ beta + y[ok].mean()
    return out


def permutation_null_or(df: pd.DataFrame, outcome: str, term: str, n_perm: int = 100, seed: int = 0) -> np.ndarray:
    """Null distribution of a coefficient by permuting demographics within dataset."""
    rng = np.random.default_rng(seed)
    d = _prepare(df, outcome)
    out = np.empty(n_perm)
    for i in range(n_perm):
        p = d.copy()
        for col in ("age", "sex", "group"):
            p[col] = p.groupby("dataset")[col].transform(lambda s: s.iloc[rng.permutation(len(s))].to_numpy())
        p["group"] = pd.Categorical(p["group"], categories=d["group"].cat.categories)
        p["sex"] = pd.Categorical(p["sex"], categories=["F", "M"])
        fit = fit_exclusion_glm(p, outcome)
        out[i] = fit.params.get(term, np.nan)
    return out
