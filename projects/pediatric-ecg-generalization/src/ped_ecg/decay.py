"""AUROC-as-a-function-of-age generalization decay curve, with nulls.

Given per-record predicted scores, binary labels and ages for a pediatric test
set, estimate the smooth AUROC(age) curve of an adult-trained model, with a
bootstrap confidence band and an age-permutation null, plus a confident-error
analysis by age band.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _windowed_auroc(age: np.ndarray, y: np.ndarray, s: np.ndarray, centers: np.ndarray, bw: float):
    """AUROC in a sliding age window (bandwidth bw years) centered at each center."""
    out = np.full(len(centers), np.nan)
    for i, c in enumerate(centers):
        m = np.abs(age - c) <= bw
        if m.sum() >= 20 and len(np.unique(y[m])) == 2:
            out[i] = roc_auc_score(y[m], s[m])
    return out


def auroc_vs_age(
    age: np.ndarray, y: np.ndarray, scores: np.ndarray,
    centers: np.ndarray | None = None, bw: float = 2.0,
    n_boot: int = 500, seed: int = 0,
) -> dict:
    """Sliding-window AUROC(age) with a percentile bootstrap band.

    Returns dict with centers, auroc, ci_low, ci_high.
    """
    age, y, scores = np.asarray(age, float), np.asarray(y), np.asarray(scores, float)
    if centers is None:
        centers = np.linspace(np.nanmin(age), np.nanmax(age), 15)
    point = _windowed_auroc(age, y, scores, centers, bw)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, len(centers)), np.nan)
    n = len(age)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _windowed_auroc(age[idx], y[idx], scores[idx], centers, bw)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return {"centers": centers, "auroc": point, "ci_low": lo, "ci_high": hi}


def age_permutation_null(
    age: np.ndarray, y: np.ndarray, scores: np.ndarray, bw: float = 2.0,
    n_perm: int = 500, seed: int = 0,
) -> dict:
    """Null distribution of the age-slope of AUROC when ages are shuffled.

    Tests H0: AUROC does not depend on age. Returns observed slope and p-value.
    """
    age, y, scores = np.asarray(age, float), np.asarray(y), np.asarray(scores, float)
    centers = np.linspace(np.nanmin(age), np.nanmax(age), 12)

    def _slope(a):
        au = _windowed_auroc(a, y, scores, centers, bw)
        m = np.isfinite(au)
        if m.sum() < 4:
            return np.nan
        return float(np.polyfit(centers[m], au[m], 1)[0])

    obs = _slope(age)
    rng = np.random.default_rng(seed)
    null = np.array([_slope(rng.permutation(age)) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return {"slope": obs, "p": np.nan}
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"slope": obs, "p": float(p)}


def confident_error_rate(y: np.ndarray, scores: np.ndarray, age: np.ndarray,
                         bands: dict[str, tuple[float, float]], thresh: float = 0.8) -> dict:
    """Fraction of high-confidence (score>thresh) predictions that are wrong, by age band.

    Flags age ranges where the adult model is confidently wrong on age-normal variants.
    """
    y, scores, age = np.asarray(y), np.asarray(scores, float), np.asarray(age, float)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (age >= lo) & (age < hi) & (scores > thresh)
        if m.sum() >= 5:
            out[name] = float(np.mean(y[m] == 0))  # confident-positive but true-negative
        else:
            out[name] = float("nan")
    return out
