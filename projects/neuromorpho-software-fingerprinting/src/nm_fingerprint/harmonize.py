"""Harmonisation of morphometric feature tables and tests of residual batch structure.

* :func:`combat` - parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic, 2007) with optional
  biological covariates that are protected from removal.
* :func:`kbet_rejection_rate` - a kBET-style local batch-mixing test (Büttner et al., 2019):
  fraction of sampled neighbourhoods whose batch composition differs from the global one.
* :func:`effect_size_preservation` - compares a biological contrast (Cohen's d per feature)
  before and after harmonisation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _design(batch: np.ndarray, covars: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    levels, idx = np.unique(batch, return_inverse=True)
    B = np.zeros((len(batch), len(levels)))
    B[np.arange(len(batch)), idx] = 1.0
    D = B if covars is None else np.column_stack([B, np.asarray(covars, dtype=float)])
    return D, idx, len(levels)


def _postmean(g_hat: np.ndarray, g_bar: float, n: int, d_star: np.ndarray, t2: float) -> np.ndarray:
    return (n * t2 * g_hat + d_star * g_bar) / (n * t2 + d_star)


def _postvar(sum2: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def combat(X: np.ndarray, batch: Sequence, covars: Optional[np.ndarray] = None, parametric: bool = True,
           max_iter: int = 100, tol: float = 1e-5, mean_only: bool = False) -> Dict[str, object]:
    """ComBat-adjust a (n_samples, n_features) matrix for ``batch``, protecting ``covars``.

    Returns ``{"X_adj", "gamma_star", "delta_star", "levels"}``. Features are standardised using the
    batch-and-covariate regression fit, batch location/scale parameters are shrunk with parametric
    EB priors, and the data are re-scaled to the pooled variance and grand mean (plus covariate effects).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    batch = np.asarray(batch)
    D, idx, n_batch = _design(batch, covars)
    n_per = np.bincount(idx, minlength=n_batch).astype(float)
    if np.any(n_per < 2):
        raise ValueError("every batch needs at least 2 samples")
    # OLS fit per feature
    beta = np.linalg.lstsq(D, X, rcond=None)[0]  # (n_design, p)
    grand_mean = (n_per / n) @ beta[:n_batch]
    resid = X - D @ beta
    var_pooled = (resid ** 2).sum(axis=0) / n
    var_pooled = np.maximum(var_pooled, 1e-12)
    stand_mean = np.outer(np.ones(n), grand_mean)
    if covars is not None:
        stand_mean += np.asarray(covars, dtype=float) @ beta[n_batch:]
    S = (X - stand_mean) / np.sqrt(var_pooled)

    gamma_hat = np.vstack([S[idx == b].mean(axis=0) for b in range(n_batch)])
    delta_hat = np.vstack([S[idx == b].var(axis=0, ddof=1) for b in range(n_batch)])
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(delta_hat)
    for b in range(n_batch):
        g_bar, t2 = float(gamma_hat[b].mean()), float(gamma_hat[b].var(ddof=1)) if p > 1 else 1.0
        m, v = float(delta_hat[b].mean()), float(delta_hat[b].var(ddof=1)) if p > 1 else 1.0
        a_prior = (2 * v + m ** 2) / v if v > 0 else 2.0
        b_prior = (m * v + m ** 3) / v if v > 0 else 1.0
        if mean_only or not parametric:
            g_new = _postmean(gamma_hat[b], g_bar, int(n_per[b]), delta_hat[b], t2) if parametric else gamma_hat[b]
            d_new = np.ones(p) if mean_only else delta_hat[b]
        else:
            g_new, d_new = gamma_hat[b].copy(), delta_hat[b].copy()
            Sb = S[idx == b]
            for _ in range(max_iter):
                g_next = _postmean(gamma_hat[b], g_bar, int(n_per[b]), d_new, t2)
                sum2 = ((Sb - g_next) ** 2).sum(axis=0)
                d_next = _postvar(sum2, int(n_per[b]), a_prior, b_prior)
                change = max(np.max(np.abs(g_next - g_new) / (np.abs(g_new) + 1e-12)), np.max(np.abs(d_next - d_new) / (d_new + 1e-12)))
                g_new, d_new = g_next, d_next
                if change < tol:
                    break
        gamma_star[b], delta_star[b] = g_new, np.maximum(d_new, 1e-12)
    S_adj = S.copy()
    for b in range(n_batch):
        S_adj[idx == b] = (S[idx == b] - gamma_star[b]) / np.sqrt(delta_star[b])
    X_adj = S_adj * np.sqrt(var_pooled) + stand_mean
    return {"X_adj": X_adj, "gamma_star": gamma_star, "delta_star": delta_star, "levels": np.unique(batch)}


def kbet_rejection_rate(X: np.ndarray, batch: Sequence, k: int = 25, n_samples: int = 300, alpha: float = 0.05,
                        seed: int = 0) -> Dict[str, float]:
    """Fraction of k-nearest-neighbourhoods whose batch composition rejects the global composition (chi-square).

    0 means batches are locally well mixed; values near 1 mean neighbourhoods are batch-pure.
    Features are z-scored before computing Euclidean neighbourhoods.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num((X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-12))
    batch = np.asarray(batch)
    levels, idx = np.unique(batch, return_inverse=True)
    global_p = np.bincount(idx, minlength=len(levels)) / len(idx)
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    pick = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    _, neigh = nn.kneighbors(X[pick])
    rejections = 0
    for row in neigh:
        counts = np.bincount(idx[row[1:]], minlength=len(levels))
        expected = global_p * k
        ok = expected > 0
        chi2 = np.sum((counts[ok] - expected[ok]) ** 2 / expected[ok])
        pval = stats.chi2.sf(chi2, df=max(int(ok.sum()) - 1, 1))
        rejections += int(pval < alpha)
    return {"rejection_rate": rejections / len(pick), "k": k, "n_sampled": int(len(pick)), "n_batches": int(len(levels))}


def effect_size_preservation(X_before: np.ndarray, X_after: np.ndarray, contrast: Sequence, feature_names: Optional[Sequence[str]] = None
                             ) -> pd.DataFrame:
    """Cohen's d for a binary biological contrast per feature, before vs after harmonisation."""
    c = np.asarray(contrast)
    levels = np.unique(c)
    if len(levels) != 2:
        raise ValueError("contrast must be binary")

    def d(X: np.ndarray) -> np.ndarray:
        a, b = X[c == levels[0]], X[c == levels[1]]
        sp = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1)) / (len(a) + len(b) - 2))
        return (a.mean(axis=0) - b.mean(axis=0)) / (sp + 1e-12)

    d0, d1 = d(np.asarray(X_before, dtype=float)), d(np.asarray(X_after, dtype=float))
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(len(d0))]
    return pd.DataFrame({"feature": names, "d_before": d0, "d_after": d1, "abs_change": np.abs(d1 - d0)})


__all__ = ["combat", "kbet_rejection_rate", "effect_size_preservation"]
