"""Functional connectivity metrics and model-empirical similarity."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def fc_from_timeseries(x: np.ndarray) -> np.ndarray:
    """Pearson FC of a (T, N) time-series array."""
    return np.corrcoef(np.asarray(x, float).T)


def upper(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(M), k=1)
    return np.asarray(M)[iu]


def fc_similarity(fc_model: np.ndarray, fc_emp: np.ndarray, method: str = "pearson") -> float:
    """Similarity between two FC matrices on the upper triangle (pearson or spearman)."""
    a, b = upper(fc_model), upper(fc_emp)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    r = stats.pearsonr(a[ok], b[ok])[0] if method == "pearson" else stats.spearmanr(a[ok], b[ok])[0]
    return float(r)


def homotopic_fc(fc: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """FC values of homotopic (left-right) region pairs."""
    return np.asarray([fc[i, j] for i, j in pairs])


def homotopic_similarity(fc_model: np.ndarray, fc_emp: np.ndarray, pairs: np.ndarray) -> float:
    a, b = homotopic_fc(fc_model, pairs), homotopic_fc(fc_emp, pairs)
    return float(stats.pearsonr(a, b)[0]) if len(a) > 2 else float("nan")


def sc_fc_coupling(W: np.ndarray, fc: np.ndarray, log: bool = True) -> float:
    """Spearman correlation between structural weights (nonzero edges) and FC."""
    Ws = 0.5 * (W + W.T)
    a, b = upper(Ws), upper(fc)
    m = a > 0
    if m.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(np.log(a[m]) if log else a[m], b[m])[0])


def fcd(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Functional connectivity dynamics: correlation between sliding-window FC vectors.

    Returns the (n_windows x n_windows) FCD matrix for (T, N) time series.
    """
    T = x.shape[0]
    starts = range(0, T - window + 1, step)
    vecs = np.array([upper(np.corrcoef(x[s:s + window].T)) for s in starts])
    return np.corrcoef(vecs)


def fcd_distribution(F: np.ndarray) -> np.ndarray:
    return upper(F)


def fcd_ks(F_model: np.ndarray, F_emp: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the upper-triangular FCD distributions."""
    return float(stats.ks_2samp(fcd_distribution(F_model), fcd_distribution(F_emp)).statistic)


def fit_gain_over_null(score_obs: float, null_scores: Sequence[float]) -> Dict[str, float]:
    """Observed fit vs. distribution of fits obtained with null connectomes."""
    null = np.asarray(null_scores, float)
    null = null[np.isfinite(null)]
    if not len(null):
        return {"gain": float("nan"), "z": float("nan"), "p": float("nan")}
    p = float((np.sum(null >= score_obs) + 1) / (len(null) + 1))
    return {"gain": float(score_obs - null.mean()), "z": float((score_obs - null.mean()) / max(null.std(), 1e-12)),
            "p": p, "null_mean": float(null.mean())}


def region_overlap(names_a: Sequence[str], names_b: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Indices into a and b of the regions present in both (for fMRI vs. widefield comparisons)."""
    common = [n for n in names_a if n in set(names_b)]
    ia = np.array([list(names_a).index(n) for n in common])
    ib = np.array([list(names_b).index(n) for n in common])
    return ia, ib
