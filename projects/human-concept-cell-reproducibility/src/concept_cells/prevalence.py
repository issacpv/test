"""Hierarchical prevalence of concept cells, cluster bootstrap, heterogeneity, agreement.

Input is a long "units table": one row per (unit, criterion) or a wide table with one
boolean column per criterion, plus grouping columns (``dataset``, ``session``, ``patient``,
``region``, optionally ``electrode``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def cluster_bootstrap_ci(flags: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> Dict[str, float]:
    """Proportion with a cluster (e.g. session) bootstrap CI.

    Resamples clusters with replacement and recomputes the pooled proportion, which respects
    the non-independence of units recorded in the same session / patient.
    """
    flags = np.asarray(flags, dtype=float)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    members = {c: flags[clusters == c] for c in ids}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        vals = np.concatenate([members[c] for c in pick])
        boots[b] = vals.mean() if vals.size else np.nan
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(flags.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_units": int(flags.size),
            "n_clusters": int(len(ids))}


def prevalence_table(units: pd.DataFrame, criteria: Sequence[str], by: Sequence[str] = ("dataset", "region"),
                     cluster: str = "session", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Prevalence per group x criterion with Wilson and cluster-bootstrap CIs."""
    rows: List[Dict[str, object]] = []
    for keys, grp in units.groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for crit in criteria:
            flags = grp[crit].to_numpy(dtype=bool)
            k, n = int(flags.sum()), int(flags.size)
            w_lo, w_hi = wilson_ci(k, n)
            cb = cluster_bootstrap_ci(flags, grp[cluster].to_numpy(), n_boot=n_boot, seed=seed)
            rows.append({**dict(zip(by, keys)), "criterion": crit, "n_units": n, "n_selected": k,
                         "prevalence": k / n if n else np.nan, "wilson_low": w_lo, "wilson_high": w_hi,
                         "boot_low": cb["ci_low"], "boot_high": cb["ci_high"], "n_clusters": cb["n_clusters"]})
    return pd.DataFrame(rows)


def cochran_q(proportions: Iterable[float], ns: Iterable[int]) -> Dict[str, float]:
    """Cochran's Q and I^2 for heterogeneity of proportions across studies (logit scale)."""
    p = np.asarray(list(proportions), dtype=float)
    n = np.asarray(list(ns), dtype=float)
    p = np.clip(p, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))  # continuity for 0/1
    logit = np.log(p / (1 - p))
    var = 1.0 / (n * p * (1 - p))
    w = 1.0 / var
    pooled = np.sum(w * logit) / np.sum(w)
    q = float(np.sum(w * (logit - pooled) ** 2))
    df = len(p) - 1
    i2 = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
    return {"Q": q, "df": df, "p": float(stats.chi2.sf(q, df)) if df > 0 else float("nan"), "I2": i2,
            "pooled_prevalence": float(1 / (1 + np.exp(-pooled)))}


def criterion_agreement(units: pd.DataFrame, criteria: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Cohen's kappa and Jaccard index between criterion selection flags."""
    k = len(criteria)
    kappa = np.full((k, k), np.nan)
    jacc = np.full((k, k), np.nan)
    for i, a in enumerate(criteria):
        fa = units[a].to_numpy(dtype=bool)
        for j, b in enumerate(criteria):
            fb = units[b].to_numpy(dtype=bool)
            po = np.mean(fa == fb)
            pe = fa.mean() * fb.mean() + (1 - fa.mean()) * (1 - fb.mean())
            kappa[i, j] = (po - pe) / (1 - pe) if pe < 1 else 1.0
            union = np.sum(fa | fb)
            jacc[i, j] = np.sum(fa & fb) / union if union else 1.0
    return (pd.DataFrame(kappa, index=criteria, columns=criteria), pd.DataFrame(jacc, index=criteria, columns=criteria))


def mixed_logistic_prevalence(units: pd.DataFrame, flag: str, fixed: Sequence[str] = ("region",),
                              groups: str = "session", covariates: Sequence[str] = ()) -> Optional[object]:
    """Logistic GLM of selection status with cluster-robust SEs by ``groups`` (session/patient).

    A full random-effects logistic model (``statsmodels.BinomialBayesMixedGLM``) is the
    intended production model; this cluster-robust GLM is the fast, dependency-light version
    that gives the same fixed-effect contrasts with valid SEs under within-cluster correlation.
    Returns the fitted statsmodels results object (or None if statsmodels is missing).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:  # pragma: no cover
        return None
    df = units.copy()
    df["_y"] = df[flag].astype(int)
    terms = " + ".join([f"C({f})" for f in fixed] + list(covariates)) or "1"
    model = smf.glm(f"_y ~ {terms}", data=df, family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[groups].astype("category").cat.codes.to_numpy()})


def beta_binomial_loglik(params: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    a, b = np.exp(params)
    from scipy.special import betaln

    return float(np.sum(betaln(k + a, n - k + b) - betaln(a, b)))


def beta_binomial_prevalence(k: Sequence[int], n: Sequence[int]) -> Dict[str, float]:
    """Beta-binomial ML fit across sessions: mean prevalence and over-dispersion rho = 1/(a+b+1)."""
    from scipy.optimize import minimize

    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    res = minimize(lambda p: -beta_binomial_loglik(p, k_arr, n_arr), x0=np.log([1.0, 1.0]), method="Nelder-Mead")
    a, b = np.exp(res.x)
    return {"mean": float(a / (a + b)), "rho": float(1.0 / (a + b + 1.0)), "a": float(a), "b": float(b),
            "converged": bool(res.success)}


__all__ = ["wilson_ci", "cluster_bootstrap_ci", "prevalence_table", "cochran_q", "criterion_agreement",
           "mixed_logistic_prevalence", "beta_binomial_prevalence"]
