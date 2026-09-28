"""Null models for drift statistics.

* time-shuffle: permute the block assignment of trials (within condition), which
  removes temporal ordering while keeping tuning, rate and trial-count structure;
* Poisson rate-matched: regenerate counts from each unit's condition mean pooled
  over blocks (no drift by construction), giving the decline expected from
  finite counts alone;
* circular shift: shift each spike train by a random offset, destroying stimulus
  locking while preserving autocorrelation (used for PSTH significance).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def permutation_pvalue(observed: float, null: np.ndarray, alternative: str = "greater") -> float:
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    n = len(null)
    if n == 0 or not np.isfinite(observed):
        return np.nan
    if alternative == "greater":
        return float((1 + np.sum(null >= observed)) / (n + 1))
    if alternative == "less":
        return float((1 + np.sum(null <= observed)) / (n + 1))
    return float((1 + np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean()))) / (n + 1))


def _summarize(observed: float, null: np.ndarray, alternative: str) -> Dict[str, float]:
    null = np.asarray(null, float)
    sd = np.nanstd(null)
    return {"observed": float(observed), "null_mean": float(np.nanmean(null)), "null_sd": float(sd),
            "z": float((observed - np.nanmean(null)) / sd) if sd > 0 else np.nan,
            "p_value": permutation_pvalue(observed, null, alternative), "n_perm": int(len(null))}


def time_shuffle_null(X: np.ndarray, cond: np.ndarray, block: np.ndarray,
                      metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
                      n_perm: int = 200, seed: int = 0, alternative: str = "less") -> Dict[str, float]:
    """Null distribution of ``metric_fn(X, cond, block)`` under block-label permutation within condition.

    ``alternative='less'`` is appropriate for similarity-type metrics (observed lag-1
    PV correlation lower than the null means drift); use ``'greater'`` for drift indices.
    """
    rng = np.random.default_rng(seed)
    observed = metric_fn(X, cond, block)
    null = np.empty(n_perm)
    for i in range(n_perm):
        b = block.copy()
        for c in np.unique(cond):
            idx = np.where(cond == c)[0]
            b[idx] = block[rng.permutation(idx)]
        null[i] = metric_fn(X, cond, b)
    return _summarize(observed, null, alternative)


def poisson_rate_matched_null(X: np.ndarray, cond: np.ndarray, block: np.ndarray,
                              metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
                              n_perm: int = 200, seed: int = 0, alternative: str = "less") -> Dict[str, float]:
    """Null from Poisson surrogates with per-(unit, condition) means pooled over blocks.

    ``X`` must contain non-negative counts (or rates times a constant window). The
    surrogate has identical tuning and trial structure but no drift, so the
    difference ``observed - null_mean`` is drift beyond finite-count noise.
    """
    rng = np.random.default_rng(seed)
    observed = metric_fn(X, cond, block)
    conds = np.unique(cond)
    means = np.stack([X[cond == c].mean(axis=0) for c in conds])  # (n_cond, n_units)
    lookup = {c: i for i, c in enumerate(conds)}
    lam = np.stack([means[lookup[c]] for c in cond])
    null = np.empty(n_perm)
    for i in range(n_perm):
        Xs = rng.poisson(np.clip(lam, 0, None)).astype(float)
        null[i] = metric_fn(Xs, cond, block)
    return _summarize(observed, null, alternative)


def circular_shift_spike_trains(spike_times: Dict[int, np.ndarray], t_start: float, t_end: float,
                                rng: Optional[np.random.Generator] = None) -> Dict[int, np.ndarray]:
    """Shift each unit's spike train by an independent random offset, wrapping within [t_start, t_end)."""
    rng = np.random.default_rng(0) if rng is None else rng
    span = t_end - t_start
    out = {}
    for u, st in spike_times.items():
        if len(st) == 0:
            out[u] = st.copy()
            continue
        shift = rng.random() * span
        out[u] = np.sort(t_start + np.mod(st - t_start + shift, span))
    return out


def lag1_pv_metric(X: np.ndarray, cond: np.ndarray, block: np.ndarray) -> float:
    """Convenience metric: lag-1 population-vector correlation (higher = more stable)."""
    from .drift_metrics import drift_rate_from_matrix, population_vector_correlation

    return drift_rate_from_matrix(population_vector_correlation(X, cond, block))["lag1"]


def decoder_drift_metric(X: np.ndarray, cond: np.ndarray, block: np.ndarray) -> float:
    """Convenience metric: decoder drift index (higher = more drift)."""
    from .drift_metrics import decoder_cross_time, decoder_drift_index

    return decoder_drift_index(decoder_cross_time(X, cond, block))
