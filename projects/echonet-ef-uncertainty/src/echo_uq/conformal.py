"""Conformal prediction intervals for EF regression, with and without covariate shift.

* :func:`split_conformal_quantile` - standard split conformal (Vovk et al.; Lei et al., 2018).
* Locally adaptive scores ``|y - f(x)| / sigma(x)`` via the ``sigma`` arguments.
* :func:`weighted_conformal_quantile` - weighted conformal under covariate shift
  (Tibshirani, Foygel Barber, Candes & Ramdas, 2019): calibration scores are
  weighted by ``w(x) = p_test(x) / p_cal(x)`` and the test point carries its
  own weight with an atom at +inf.
* :func:`density_ratio_weights` - likelihood ratios from a logistic domain classifier.
* Evaluation: coverage with Clopper-Pearson CIs, width, conditional coverage,
  and whether width flags hard cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def conformity_scores(y: np.ndarray, pred: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    """Absolute residuals, optionally normalised by a heuristic scale ``sigma(x) > 0``."""
    r = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    if sigma is not None:
        r = r / np.clip(np.asarray(sigma, float), 1e-6, None)
    return r


def split_conformal_quantile(scores_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores_cal, float))
    n = s.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def predict_interval(pred: np.ndarray, q: np.ndarray | float, sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Interval ``pred -/+ q * sigma`` (``sigma`` = 1 for non-adaptive scores)."""
    pred = np.asarray(pred, float)
    scale = np.ones_like(pred) if sigma is None else np.clip(np.asarray(sigma, float), 1e-6, None)
    half = np.asarray(q, float) * scale
    return pred - half, pred + half


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Smallest v such that the normalised weight of {values <= v} >= level (weights may include an inf atom)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    idx = np.searchsorted(cw, level, side="left")
    return float(v[min(idx, v.size - 1)])


def weighted_conformal_quantile(scores_cal: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray | float, alpha: float = 0.1) -> np.ndarray:
    """Per-test-point quantile under covariate shift (Tibshirani et al., 2019, Eq. 6).

    ``w_cal[i] = p_test(x_i)/p_cal(x_i)`` for calibration points, ``w_test`` the
    same ratio at each test point.  Returns an array of quantiles (one per test point).
    """
    s = np.asarray(scores_cal, float)
    w_cal = np.asarray(w_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))
    vals = np.append(s, np.inf)
    out = np.empty(w_test.size)
    for j, wt in enumerate(w_test):
        w = np.append(w_cal, wt)
        out[j] = weighted_quantile(vals, w, 1 - alpha)
    return out


def density_ratio_weights(X_cal: np.ndarray, X_test: np.ndarray, clip: float = 50.0, seed: int = 0,
                          C: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``w(x) = p_test(x)/p_cal(x)`` with logistic regression on standardised features.

    Returns ``(w_cal, w_test)``; both clipped to ``[1/clip, clip]``.  Use
    :func:`echo_uq.shift.effective_sample_size` to check that the calibration
    set still carries enough weight.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([X_cal, X_test]).astype(float)
    y = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sc.transform(X), y)
    prior = len(X_test) / len(X_cal)

    def ratio(Z: np.ndarray) -> np.ndarray:
        p = lr.predict_proba(sc.transform(Z))[:, 1]
        return np.clip((p / np.clip(1 - p, 1e-6, None)) / prior, 1.0 / clip, clip)

    return ratio(np.asarray(X_cal, float)), ratio(np.asarray(X_test, float))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@dataclass
class IntervalMetrics:
    coverage: float
    coverage_low: float
    coverage_high: float
    mean_width: float
    median_width: float
    n: int


def coverage_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact interval for a coverage proportion."""
    if n == 0:
        return float("nan"), float("nan")
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def evaluate_intervals(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> IntervalMetrics:
    """Empirical coverage (with exact CI) and interval widths."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    covered = (y >= lo) & (y <= hi)
    k, n = int(covered.sum()), int(covered.size)
    cl, ch = coverage_ci(k, n)
    w = hi - lo
    finite = np.isfinite(w)
    return IntervalMetrics(k / n if n else float("nan"), cl, ch,
                           float(w[finite].mean()) if finite.any() else float("inf"),
                           float(np.median(w[finite])) if finite.any() else float("inf"), n)


def conditional_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, groups: np.ndarray) -> dict[str, IntervalMetrics]:
    """Coverage/width per level of ``groups`` (dataset, view, quality label, EF tertile ...)."""
    groups = np.asarray(groups)
    return {str(g): evaluate_intervals(y[groups == g], lo[groups == g], hi[groups == g]) for g in np.unique(groups)}


def coverage_by_bins(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, feature: np.ndarray, n_bins: int = 5) -> list[tuple[float, float, float]]:
    """Coverage in quantile bins of a continuous feature -> [(bin_low, bin_high, coverage)]; the min is a worst-slab proxy."""
    f = np.asarray(feature, float)
    edges = np.quantile(f, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (f >= edges[i]) & (f <= edges[i + 1]) if i == n_bins - 1 else (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out.append((float(edges[i]), float(edges[i + 1]), evaluate_intervals(y[m], lo[m], hi[m]).coverage))
    return out


def width_flags_hard_cases(widths: np.ndarray, hard: np.ndarray) -> float:
    """AUROC of interval width for detecting hard cases (1 = hard); NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    hard = np.asarray(hard).astype(int)
    if hard.sum() in (0, hard.size):
        return float("nan")
    return float(roc_auc_score(hard, np.asarray(widths, float)))


def abstention_rate(widths: np.ndarray, reference_widths: np.ndarray, pct: float = 95.0) -> float:
    """Fraction of test intervals wider than the ``pct``-th percentile of in-distribution widths."""
    thr = np.percentile(np.asarray(reference_widths, float), pct)
    return float(np.mean(np.asarray(widths, float) > thr))


def bland_altman(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Bias and 95 % limits of agreement between predicted and reference EF."""
    d = np.asarray(pred, float) - np.asarray(y, float)
    return {"bias": float(d.mean()), "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
            "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean()))}
