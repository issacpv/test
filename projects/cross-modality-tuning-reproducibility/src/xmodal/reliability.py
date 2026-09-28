"""Reliability ceilings: split-half and test-retest reliability, Spearman-Brown, correction for attenuation."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def spearman_brown(r_half: float, k: float = 2.0) -> float:
    """Reliability of the full-length measure from the half-length correlation."""
    if not np.isfinite(r_half):
        return np.nan
    return float(k * r_half / (1.0 + (k - 1.0) * r_half))


def split_half_reliability(trial_resp: np.ndarray, conditions: np.ndarray, n_splits: int = 50,
                           rng: Optional[np.random.Generator] = None, method: str = "pearson") -> Dict[str, float]:
    """Reliability of a tuning curve: correlate condition means from two random halves of the trials,
    average over splits, and apply Spearman-Brown."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(trial_resp, dtype=float)
    c = np.asarray(conditions)
    levels = pd.unique(c)
    rs = []
    for _ in range(n_splits):
        m1 = np.zeros(len(x), dtype=bool)
        for lv in levels:
            idx = np.flatnonzero(c == lv)
            pick = rng.choice(idx, size=len(idx) // 2, replace=False)
            m1[pick] = True
        a = np.array([x[(c == lv) & m1].mean() for lv in levels])
        b = np.array([x[(c == lv) & ~m1].mean() for lv in levels])
        if a.std() > 0 and b.std() > 0:
            r = pearsonr(a, b)[0] if method == "pearson" else spearmanr(a, b)[0]
            rs.append(r)
    r_half = float(np.mean(rs)) if rs else np.nan
    return {"r_half": r_half, "reliability": spearman_brown(r_half), "n_splits": len(rs)}


def test_retest_reliability(metric_a: np.ndarray, metric_b: np.ndarray, method: str = "spearman") -> float:
    """Correlation of a per-cell metric across two sessions/blocks (matched cells)."""
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0] if method == "spearman" else pearsonr(a[ok], b[ok])[0])


def disattenuated_correlation(r_xy: float, rel_x: float, rel_y: float) -> float:
    """Correction for attenuation: r_xy / sqrt(rel_x * rel_y), capped at 1."""
    denom = np.sqrt(max(rel_x, 0.0) * max(rel_y, 0.0))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(min(r_xy / denom, 1.0))


def reliability_ceiling(rel_x: float, rel_y: float) -> float:
    """Maximum expected raw correlation between two measures given their reliabilities."""
    return float(np.sqrt(max(rel_x, 0.0) * max(rel_y, 0.0)))


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], data: np.ndarray, n_boot: int = 1000,
                 rng: Optional[np.random.Generator] = None, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap of a statistic over rows of ``data`` (cells/units)."""
    rng = np.random.default_rng(0) if rng is None else rng
    data = np.asarray(data)
    point = float(stat_fn(data))
    vals = np.array([stat_fn(data[rng.integers(0, len(data), len(data))]) for _ in range(n_boot)])
    return point, float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def reliability_normalised_selectivity(selectivity: np.ndarray, reliability: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Selectivity divided by sqrt(reliability): a candidate modality-invariant metric (H6).

    Cells with reliability below ``floor`` are set to NaN rather than inflated."""
    s = np.asarray(selectivity, dtype=float)
    r = np.asarray(reliability, dtype=float)
    out = s / np.sqrt(np.clip(r, floor, None))
    out[r < floor] = np.nan
    return out
