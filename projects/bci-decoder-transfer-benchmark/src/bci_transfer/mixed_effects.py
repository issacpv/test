"""Statistics for the benchmark: paired effect sizes, negative-transfer rate, mixed models,
dataset-level meta-regression and Holm correction.

The benchmark's long table has one row per (dataset, subject, method, budget, draw) with an
``accuracy`` column. Subjects are nested in datasets, so the mixed model uses a dataset random
intercept plus a subject-in-dataset variance component.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def paired_effect_size(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of per-subject accuracies ``a`` (method) vs ``b`` (baseline).

    Returns mean difference with 95% t-CI, Cohen's d_z, Wilcoxon signed-rank p and the
    fraction of subjects with ``a > b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean_diff": float(d.mean()) if n else np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "d_z": np.nan, "p_wilcoxon": np.nan, "frac_improved": np.nan}
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    try:
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = np.nan
    return {"n": n, "mean_diff": float(d.mean()), "ci_low": float(d.mean() - t * se), "ci_high": float(d.mean() + t * se),
            "d_z": float(d.mean() / sd) if sd > 0 else np.inf, "p_wilcoxon": p_w, "frac_improved": float(np.mean(d > 0))}


def negative_transfer_rate(acc_transfer: Sequence[float], acc_baseline: Sequence[float], margin: float = 0.0
                           ) -> Dict[str, float]:
    """Fraction of subjects for whom transfer is worse than baseline by more than ``margin``."""
    a = np.asarray(acc_transfer, dtype=float)
    b = np.asarray(acc_baseline, dtype=float)
    harmed = (b - a) > margin
    k, n = int(harmed.sum()), int(len(harmed))
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n if n else np.nan, "ci_low": lo, "ci_high": hi, "n_harmed": k, "n": n}


def fit_transfer_mixed_model(df: pd.DataFrame, outcome: str = "accuracy", method_col: str = "method",
                             budget_col: str = "budget", dataset_col: str = "dataset", subject_col: str = "subject",
                             moderators: Sequence[str] = (), reml: bool = True):
    """Mixed model ``accuracy ~ method * log(budget+1) [+ moderators]`` with dataset random intercept
    and a subject-in-dataset variance component (statsmodels ``MixedLM``).

    Returns the fitted results object. Falls back to an OLS with cluster-robust SEs by dataset
    if the mixed model fails to converge (a warning is attached as ``results.fallback``).
    """
    import statsmodels.formula.api as smf

    data = df.copy()
    data["log_budget"] = np.log(data[budget_col].astype(float) + 1.0)
    data["_subject"] = data[dataset_col].astype(str) + ":" + data[subject_col].astype(str)
    fixed = f"{outcome} ~ C({method_col}) * log_budget"
    if moderators:
        fixed += " + " + " + ".join(moderators)
    try:
        model = smf.mixedlm(fixed, data, groups=data[dataset_col], re_formula="1",
                            vc_formula={"subject": "0 + C(_subject)"})
        res = model.fit(reml=reml, method=["lbfgs", "powell"], maxiter=500)
        res.fallback = None
        return res
    except Exception as exc:  # noqa: BLE001
        ols = smf.ols(fixed, data).fit(cov_type="cluster", cov_kwds={"groups": data[dataset_col].astype("category").cat.codes})
        ols.fallback = f"MixedLM failed ({exc}); cluster-robust OLS used"
        return ols


def meta_regression(effects: Sequence[float], variances: Sequence[float], X: Optional[np.ndarray] = None,
                    names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Random-effects meta-regression with a method-of-moments (DerSimonian-Laird-type) tau^2.

    ``effects`` are dataset-level transfer gains, ``variances`` their sampling variances (e.g.
    squared SE of the mean paired difference), ``X`` an optional (n_datasets, p) moderator
    matrix (no intercept column; one is added). Returns coefficients, SEs, z, p, tau^2, I^2 and
    the fraction of between-dataset variance explained (R^2 relative to the intercept-only tau^2).
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    n = len(y)
    Xm = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), np.asarray(X, dtype=float)])
    p = Xm.shape[1]

    def _wls(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W = np.diag(weights)
        cov = np.linalg.pinv(Xm.T @ W @ Xm)
        beta = cov @ Xm.T @ W @ y
        return beta, cov

    def _tau2(Xd: np.ndarray) -> float:
        w = 1.0 / v
        W = np.diag(w)
        P = W - W @ Xd @ np.linalg.pinv(Xd.T @ W @ Xd) @ Xd.T @ W
        q = float(y @ P @ y)
        df = n - Xd.shape[1]
        denom = float(np.trace(P @ np.diag(v)) - 0)  # trace(P V) with V = diag(v); equals trace(P)*... general form
        denom = float(np.trace(P))  # since P is built from W = V^{-1}, trace(P V) = trace(P @ diag(v))
        denom = float(np.trace(P @ np.diag(v)))
        return max(0.0, (q - df) / denom) if denom > 0 else 0.0

    tau2 = _tau2(Xm)
    tau2_null = _tau2(np.ones((n, 1)))
    w = 1.0 / (v + tau2)
    beta, cov = _wls(w)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    # heterogeneity of the intercept-only model
    w0 = 1.0 / v
    mu0 = np.sum(w0 * y) / np.sum(w0)
    Q = float(np.sum(w0 * (y - mu0) ** 2))
    I2 = float(max(0.0, (Q - (n - 1)) / Q)) if Q > 0 else 0.0
    names = list(names or [f"x{i}" for i in range(p - 1)])
    return {"coef": dict(zip(["intercept"] + names, beta.tolist())), "se": dict(zip(["intercept"] + names, se.tolist())),
            "z": dict(zip(["intercept"] + names, z.tolist())), "p": dict(zip(["intercept"] + names, pv.tolist())),
            "tau2": float(tau2), "tau2_null": float(tau2_null),
            "R2_between": float(1 - tau2 / tau2_null) if tau2_null > 0 else np.nan, "Q": Q, "I2": I2, "n": n}


def per_subject_summary(df: pd.DataFrame, budget: int, method_col: str = "method", baseline: str = "scratch") -> pd.DataFrame:
    """Per-(dataset, subject) mean accuracy at one budget for every method, wide format."""
    sub = df[df["budget"] == budget]
    wide = sub.groupby(["dataset", "subject", method_col])["accuracy"].mean().unstack(method_col)
    if baseline in wide.columns:
        for col in wide.columns:
            if col != baseline:
                wide[f"gain_{col}"] = wide[col] - wide[baseline]
    return wide.reset_index()


__all__ = ["holm", "wilson_ci", "paired_effect_size", "negative_transfer_rate", "fit_transfer_mixed_model",
           "meta_regression", "per_subject_summary"]
