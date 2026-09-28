"""Subgroup performance, calibration and error-structure metrics.

Beyond the mean-accuracy gap reported by Li et al. (2022) and the 2025-2026
ABCD benchmarks, this module quantifies *how* errors differ between groups:
systematic bias (mean residual), calibration slope, residual variance and
the residual-vs-truth slope (regression-to-the-majority-mean). Uncertainty is
obtained by family-cluster bootstrap; significance of gaps by permuting
subgroup labels at the family level.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Basic metrics
# --------------------------------------------------------------------------- #
def calibration(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    """Calibration-in-the-large and slope from OLS of ``y`` on ``y_hat``.

    A perfectly calibrated predictor has slope 1 and intercept 0. Slopes < 1
    indicate over-dispersed predictions (typical of over-fit models);
    group-specific intercepts indicate systematic over/under-prediction.
    """
    if len(y) < 3 or np.std(y_hat) == 0:
        return {"cal_slope": np.nan, "cal_intercept": np.nan}
    slope, intercept, *_ = stats.linregress(y_hat, y)
    return {"cal_slope": float(slope), "cal_intercept": float(intercept)}


def basic_metrics(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    """Pearson r, R^2 (1 - SSE/SST), MAE, RMSE, bias, residual SD, calibration."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(y_hat)
    y, y_hat = y[ok], y_hat[ok]
    n = len(y)
    out: Dict[str, float] = {"n": float(n)}
    if n < 3:
        out.update({k: np.nan for k in ("r", "r2", "mae", "rmse", "bias", "resid_sd", "cal_slope", "cal_intercept")})
        return out
    resid = y_hat - y
    r = float(np.corrcoef(y, y_hat)[0, 1]) if np.std(y_hat) > 0 and np.std(y) > 0 else np.nan
    sst = float(((y - y.mean()) ** 2).sum())
    out["r"] = r
    out["r2"] = float(1 - (resid**2).sum() / sst) if sst > 0 else np.nan
    out["mae"] = float(np.abs(resid).mean())
    out["rmse"] = float(np.sqrt((resid**2).mean()))
    out["bias"] = float(resid.mean())
    out["resid_sd"] = float(resid.std(ddof=1))
    out.update(calibration(y, y_hat))
    return out


def subgroup_performance(
    y: np.ndarray, y_hat: np.ndarray, groups: Sequence, include_all: bool = True
) -> pd.DataFrame:
    """Per-subgroup metrics table (rows = groups, plus ``ALL``)."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    rows = {}
    for g in pd.unique(groups):
        m = groups == g
        rows[g] = basic_metrics(y[m], y_hat[m])
    if include_all:
        rows["ALL"] = basic_metrics(y, y_hat)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Error structure
# --------------------------------------------------------------------------- #
def error_structure(
    y: np.ndarray, y_hat: np.ndarray, groups: Sequence, reference: object
) -> pd.DataFrame:
    """Compare the *structure* of residuals in each group against a reference group.

    Columns
    -------
    resid_vs_y_slope
        Slope of residual on true score; negative values mean predictions
        shrink toward a mean (if the majority mean differs from the minority
        mean this produces group-specific bias even with equal r).
    mean_resid_diff, welch_p
        Difference in mean residual vs reference (Welch t-test).
    levene_p
        Test of unequal residual variance vs reference.
    frac_overpredicted
        Share of subjects whose predicted score exceeds their true score.
    ks_p
        Kolmogorov-Smirnov test comparing residual distributions.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    resid = y_hat - y
    ref_mask = groups == reference
    if ref_mask.sum() == 0:
        raise ValueError(f"reference group {reference!r} not found")
    rows = {}
    for g in pd.unique(groups):
        m = groups == g
        rg = resid[m]
        row: Dict[str, float] = {"n": float(m.sum())}
        if m.sum() >= 3 and np.std(y[m]) > 0:
            row["resid_vs_y_slope"] = float(stats.linregress(y[m], rg).slope)
        else:
            row["resid_vs_y_slope"] = np.nan
        row["mean_resid"] = float(rg.mean())
        row["frac_overpredicted"] = float((rg > 0).mean())
        if g == reference or m.sum() < 3:
            row.update({"mean_resid_diff": 0.0, "welch_p": np.nan, "levene_p": np.nan, "ks_p": np.nan})
        else:
            row["mean_resid_diff"] = float(rg.mean() - resid[ref_mask].mean())
            row["welch_p"] = float(stats.ttest_ind(rg, resid[ref_mask], equal_var=False).pvalue)
            row["levene_p"] = float(stats.levene(rg, resid[ref_mask]).pvalue)
            row["ks_p"] = float(stats.ks_2samp(rg, resid[ref_mask]).pvalue)
        rows[g] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Gaps with uncertainty
# --------------------------------------------------------------------------- #
def performance_gap(
    y: np.ndarray, y_hat: np.ndarray, groups: Sequence, reference: object, metric: str = "r"
) -> pd.Series:
    """``metric(group) - metric(reference)`` for every group."""
    perf = subgroup_performance(y, y_hat, groups, include_all=False)
    return perf[metric] - perf.loc[reference, metric]


def _resample_clusters(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of a cluster (family) bootstrap resample."""
    uniq, inverse = np.unique(clusters, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(uniq))]
    draw = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([members[i] for i in draw])


def cluster_bootstrap_gap(
    y: np.ndarray,
    y_hat: np.ndarray,
    groups: Sequence,
    clusters: Sequence,
    reference: object,
    metric: str = "r",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Family-cluster bootstrap percentile CIs for subgroup gaps.

    Resampling whole families preserves the within-family dependence of
    out-of-fold predictions.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    point = performance_gap(y, y_hat, groups, reference, metric)
    boots = []
    for _ in range(n_boot):
        idx = _resample_clusters(clusters, rng)
        try:
            g = performance_gap(y[idx], y_hat[idx], groups[idx], reference, metric)
        except KeyError:  # reference group absent in resample (tiny groups)
            continue
        boots.append(g.reindex(point.index))
    B = pd.concat(boots, axis=1).T
    lo = B.quantile(alpha / 2)
    hi = B.quantile(1 - alpha / 2)
    return pd.DataFrame({"gap": point, "ci_low": lo, "ci_high": hi, "boot_sd": B.std(ddof=1)})


def permutation_gap_test(
    y: np.ndarray,
    y_hat: np.ndarray,
    groups: Sequence,
    clusters: Sequence,
    reference: object,
    metric: str = "r",
    n_perm: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Permute subgroup labels across *families* to obtain a null for each gap.

    Family members share demographic labels, so labels are permuted at the
    family level (each family keeps one label, drawn from the observed
    family-label distribution), preserving the group-size distribution and
    the family structure. Two-sided p-values use the (k + 1)/(n + 1) rule.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(clusters, return_inverse=True)
    fam_label = np.empty(len(uniq), dtype=object)
    for i in range(len(uniq)):
        vals, counts = np.unique(groups[inverse == i], return_counts=True)
        fam_label[i] = vals[np.argmax(counts)]
    observed = performance_gap(y, y_hat, groups, reference, metric)
    exceed = pd.Series(0.0, index=observed.index)
    valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(fam_label)
        g_perm = perm[inverse]
        try:
            null = performance_gap(y, y_hat, g_perm, reference, metric).reindex(observed.index)
        except KeyError:
            continue
        valid += 1
        exceed += (np.abs(null) >= np.abs(observed)).astype(float)
    p = (exceed + 1) / (valid + 1)
    return pd.DataFrame({"gap": observed, "p_perm": p, "n_perm_valid": valid})


def accuracy_at_matched_n(
    y: np.ndarray,
    y_hat: np.ndarray,
    groups: Sequence,
    n: int,
    n_draws: int = 200,
    metric: str = "r",
    seed: int = 0,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> pd.DataFrame:
    """Evaluate each group on random subsets of size ``n`` (equalises the
    sampling variance of the metric, which otherwise favours the large group
    when comparing e.g. minimum/maximum r across groups).
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    rows = {}
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < n:
            rows[g] = {"mean": np.nan, "sd": np.nan, "n_avail": float(len(idx))}
            continue
        vals = []
        for _ in range(n_draws):
            s = rng.choice(idx, size=n, replace=False)
            if metric_fn is not None:
                vals.append(metric_fn(y[s], y_hat[s]))
            else:
                vals.append(basic_metrics(y[s], y_hat[s])[metric])
        vals = np.asarray(vals, dtype=float)
        rows[g] = {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)), "n_avail": float(len(idx))}
    return pd.DataFrame(rows).T
