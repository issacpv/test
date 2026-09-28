"""Centile scoring, extreme-deviation counts, uniformity checks, group contrasts and reliability."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

_EPS = 1e-6


def centile_to_z(c: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(np.asarray(c, float), _EPS, 1 - _EPS))


def z_to_centile(z: np.ndarray) -> np.ndarray:
    return norm.cdf(np.asarray(z, float))


def extreme_deviations(Z: pd.DataFrame, threshold: float = 1.96) -> pd.DataFrame:
    """Per-subject counts of extreme negative / positive / total deviations across features.

    ``Z`` is subjects × features (z-scores). Missing values are ignored in counts and reported in
    ``n_valid`` so that counts can be expressed as fractions.
    """
    Z = Z.apply(pd.to_numeric, errors="coerce")
    neg = (Z < -threshold).sum(axis=1)
    pos = (Z > threshold).sum(axis=1)
    n_valid = Z.notna().sum(axis=1)
    return pd.DataFrame({"n_neg": neg, "n_pos": pos, "n_total": neg + pos, "n_valid": n_valid,
                         "frac_total": (neg + pos) / n_valid.replace(0, np.nan)}, index=Z.index)


def uniformity_check(centiles: np.ndarray, threshold_z: float = 1.96) -> dict[str, float]:
    """KS statistic vs U(0,1), mean centile and the fraction beyond ±threshold (expected ≈ 5%)."""
    c = np.asarray(centiles, float)
    c = c[np.isfinite(c)]
    if len(c) == 0:
        return {"n": 0, "ks": np.nan, "ks_p": np.nan, "mean_centile": np.nan, "extreme_rate": np.nan}
    ks = stats.kstest(c, "uniform")
    z = centile_to_z(c)
    return {"n": int(len(c)), "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mean_centile": float(c.mean()), "extreme_rate": float((np.abs(z) > threshold_z).mean()),
            "expected_extreme_rate": float(2 * (1 - norm.cdf(threshold_z)))}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def group_contrast(Z: pd.DataFrame, labels: Sequence, positive_label, threshold: float = 1.96) -> pd.DataFrame:
    """Per-feature contrast of z between a clinical group and controls: Mann-Whitney U, Cliff's delta,
    extreme-deviation rates in each group, BH-FDR across features."""
    labels = np.asarray(labels)
    m = labels == positive_label
    rows = []
    for f in Z.columns:
        a, b = Z[f].to_numpy(float)[m], Z[f].to_numpy(float)[~m]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        delta = 2 * u.statistic / (len(a) * len(b)) - 1
        rows.append({"feature": f, "n_pos": len(a), "n_ctrl": len(b), "median_z_pos": float(np.median(a)),
                     "median_z_ctrl": float(np.median(b)), "cliffs_delta": float(delta), "p": float(u.pvalue),
                     "extreme_rate_pos": float((np.abs(a) > threshold).mean()),
                     "extreme_rate_ctrl": float((np.abs(b) > threshold).mean())})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr"] = bh_fdr(df["p"])
    return df


def auc_with_ci(score: np.ndarray, label: np.ndarray, n_boot: int = 1000, random_state: int = 0) -> dict[str, float]:
    """AUC of a scalar score (e.g. extreme-deviation count) for a binary label, with bootstrap CI."""
    from sklearn.metrics import roc_auc_score

    s, y = np.asarray(score, float), np.asarray(label, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"auc": float(auc), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(y))}


def icc_test_retest(x1: np.ndarray, x2: np.ndarray) -> dict[str, float]:
    """ICC(3,1) (two-way mixed, consistency) for two sessions plus the within-subject SD.

    Returns icc, within_sd (e.g. in centile points when centiles are passed) and n.
    """
    a, b = np.asarray(x1, float), np.asarray(x2, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return {"icc": np.nan, "within_sd": np.nan, "n": n}
    Y = np.column_stack([a, b])
    grand = Y.mean()
    ms_rows = 2 * ((Y.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((Y.mean(0) - grand) ** 2).sum() / 1
    ss_tot = ((Y - grand) ** 2).sum()
    ms_err = (ss_tot - ms_rows * (n - 1) - ms_cols * 1) / ((n - 1) * 1)
    icc = (ms_rows - ms_err) / (ms_rows + ms_err)
    return {"icc": float(icc), "within_sd": float(np.sqrt(max(ms_err, 0))), "n": int(n)}


def subject_deviation_profile(Z: pd.DataFrame, subject: str, top: int = 10) -> pd.DataFrame:
    """The ``top`` most extreme features for one subject (for case reports / QC)."""
    z = Z.loc[subject].astype(float).dropna()
    return z.reindex(z.abs().sort_values(ascending=False).index[:top]).rename("z").to_frame()
