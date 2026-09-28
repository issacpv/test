"""Null models and resampling helpers.

* :func:`motion_matched_permutation` permutes the phenotype *within motion strata*: the
  permuted phenotype keeps its relationship with motion but loses any relationship with the
  brain beyond motion. Prediction accuracy on such a phenotype is the accuracy obtainable from
  motion alone (a "placebo phenotype" / negative-control outcome).
* :func:`permutation_pvalue` and :func:`bootstrap_ci` are generic helpers.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def motion_matched_permutation(
    y: np.ndarray,
    motion: np.ndarray,
    n_bins: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Permute ``y`` within quantile bins of ``motion``."""
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, float)
    motion = np.asarray(motion, float)
    edges = np.quantile(motion, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, motion, side="right") - 1, 0, n_bins - 1)
    out = y.copy()
    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        if len(idx) > 1:
            out[idx] = y[rng.permutation(idx)]
    return out


def permutation_pvalue(observed: float, null: np.ndarray, alternative: str = "greater") -> float:
    """p-value with the +1 correction (Phipson and Smyth, 2010)."""
    null = np.asarray(null, float)
    if alternative == "greater":
        k = (null >= observed).sum()
    elif alternative == "less":
        k = (null <= observed).sum()
    else:
        k = (np.abs(null) >= abs(observed)).sum()
    return float((k + 1) / (len(null) + 1))


def bootstrap_ci(
    func: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    groups: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI of ``func(*arrays)``; resamples groups (families) if given."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(arrays[0])
    stats = np.empty(n_boot)
    if groups is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            stats[b] = func(*(a[idx] for a in arrays))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for b in range(n_boot):
            gs = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in gs])
            stats[b] = func(*(a[idx] for a in arrays))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), stats
