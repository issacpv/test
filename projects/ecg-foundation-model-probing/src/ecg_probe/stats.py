"""Statistics: permutation nulls, bootstrap CIs, and a DeLong AUROC test."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def permutation_pvalue(observed: float, null_samples: np.ndarray, two_sided: bool = True) -> float:
    """Empirical p-value of ``observed`` against a null sample array."""
    null = np.asarray(null_samples, float)
    if two_sided:
        count = np.sum(np.abs(null) >= abs(observed))
    else:
        count = np.sum(null >= observed)
    return float((count + 1) / (len(null) + 1))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = T
    return out


def delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """DeLong test for the difference of two correlated AUROCs (same samples).

    Returns AUROC_a, AUROC_b, the difference and a two-sided z-test p-value.
    """
    y_true = np.asarray(y_true)
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "delta": float("nan"), "p": float("nan")}

    def structural(scores):
        s = np.asarray(scores, float)
        pos_s, neg_s = s[pos], s[neg]
        tx = _compute_midrank(pos_s)
        ty = _compute_midrank(neg_s)
        tz = _compute_midrank(s)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[pos] - tx) / n
        v10 = 1.0 - (tz[neg] - ty) / m
        return auc, v01, v10

    auc_a, v01a, v10a = structural(scores_a)
    auc_b, v01b, v10b = structural(scores_b)
    s01 = np.cov(np.stack([v01a, v01b]))
    s10 = np.cov(np.stack([v10a, v10b]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta": auc_a - auc_b, "p": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var)
    from scipy.stats import norm

    p = 2 * (1 - norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "delta": float(auc_a - auc_b), "p": float(p)}
