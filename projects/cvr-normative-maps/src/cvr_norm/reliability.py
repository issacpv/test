"""Test-retest reliability metrics for CVR maps."""
from __future__ import annotations

import numpy as np


def icc_2_1(data: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement. ``data`` = (subjects, sessions)."""
    Y = np.asarray(data, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((Y.mean(0) - grand) ** 2) / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand
    ms_e = np.sum(resid**2) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def icc_bootstrap_ci(data: np.ndarray, n_boot: int = 1000, rng: np.random.Generator | None = None,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """ICC(2,1) with a subject-level percentile bootstrap CI."""
    rng = np.random.default_rng() if rng is None else rng
    Y = np.asarray(data, float)
    est = icc_2_1(Y)
    boots = np.array([icc_2_1(Y[rng.integers(0, Y.shape[0], Y.shape[0])]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def within_subject_cov(data: np.ndarray) -> float:
    """Mean within-subject coefficient of variation (%) across sessions."""
    Y = np.asarray(data, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = Y.std(axis=1, ddof=1) / np.abs(Y.mean(axis=1))
    return float(100 * np.nanmean(cv))


def minimal_detectable_change(data: np.ndarray, confidence: float = 0.95) -> float:
    """MDC = z × √2 × SEM with SEM = SD_between × √(1 - ICC)."""
    from scipy.stats import norm

    Y = np.asarray(data, float)
    icc = max(icc_2_1(Y), 0.0)
    sd = Y.mean(axis=1).std(ddof=1)
    sem = sd * np.sqrt(1 - icc)
    return float(norm.ppf(0.5 + confidence / 2) * np.sqrt(2) * sem)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else float("nan")


def retest_summary(data: np.ndarray, n_boot: int = 500, rng: np.random.Generator | None = None) -> dict[str, float]:
    """ICC (with CI), within-subject CoV and MDC95 for a (subjects × sessions) matrix."""
    est, lo, hi = icc_bootstrap_ci(data, n_boot, rng)
    return {"icc": est, "icc_ci_low": lo, "icc_ci_high": hi, "cov_within_pct": within_subject_cov(data),
            "mdc95": minimal_detectable_change(data)}
