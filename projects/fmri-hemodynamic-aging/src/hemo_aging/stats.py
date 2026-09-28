"""Reliability, partial association, attenuation and mediation helpers."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def icc_2_1(data: np.ndarray) -> float:
    """ICC(2,1) (two-way random effects, absolute agreement, single measure).

    ``data`` is ``(n_subjects, n_raters_or_sessions)`` with no missing values.
    """
    Y = np.asarray(data, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((Y.mean(0) - grand) ** 2) / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand
    ms_e = np.sum(resid**2) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def _residualize(v: np.ndarray, covars: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(v.shape[0]), covars])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_corr(x: np.ndarray, y: np.ndarray, covars: np.ndarray) -> tuple[float, float]:
    """Pearson correlation of ``x`` and ``y`` after regressing out ``covars`` (n, p). Returns (r, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covars = np.atleast_2d(np.asarray(covars, float))
    if covars.shape[0] != x.shape[0]:
        covars = covars.T
    rx, ry = _residualize(x, covars), _residualize(y, covars)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = x.size - covars.shape[1] - 2
    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    p = float(2 * sps.t.sf(abs(t), dof))
    return r, p


def age_slope(y: np.ndarray, age: np.ndarray, covars: np.ndarray | None = None) -> float:
    """Standardized age slope of ``y`` (per SD of age, in SD of y) adjusting for covariates."""
    y, age = np.asarray(y, float), np.asarray(age, float)
    if covars is not None:
        c = np.atleast_2d(np.asarray(covars, float))
        c = c.T if c.shape[0] != y.shape[0] else c
        y, age = _residualize(y, c), _residualize(age, c)
    return float(np.polyfit((age - age.mean()) / age.std(), (y - y.mean()) / y.std(), 1)[0])


def attenuation_ratio(effect_before: np.ndarray, effect_after: np.ndarray) -> float:
    """Fraction of the effect removed by correction: 1 - <|after|> / <|before|> (edge-wise mean)."""
    b = np.mean(np.abs(np.asarray(effect_before, float)))
    a = np.mean(np.abs(np.asarray(effect_after, float)))
    return float(1 - a / b) if b > 0 else float("nan")


def mediation_bootstrap(x: np.ndarray, m: np.ndarray, y: np.ndarray, covars: np.ndarray | None = None,
                        n_boot: int = 2000, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Simple mediation x -> m -> y with percentile bootstrap CI of the indirect effect a*b.

    Returns total (c), direct (c'), indirect (a*b), its 95% CI and the proportion mediated.
    """
    rng = np.random.default_rng() if rng is None else rng
    x, m, y = (np.asarray(v, float) for v in (x, m, y))
    n = x.size
    C = np.zeros((n, 0)) if covars is None else np.atleast_2d(np.asarray(covars, float))
    if C.shape[0] != n:
        C = C.T

    def fit(idx: np.ndarray) -> tuple[float, float, float, float]:
        X1 = np.column_stack([np.ones(idx.size), x[idx], C[idx]])
        a = np.linalg.lstsq(X1, m[idx], rcond=None)[0][1]
        c = np.linalg.lstsq(X1, y[idx], rcond=None)[0][1]
        X2 = np.column_stack([np.ones(idx.size), x[idx], m[idx], C[idx]])
        coef = np.linalg.lstsq(X2, y[idx], rcond=None)[0]
        return a, coef[2], c, coef[1]

    a, b, c, c_prime = fit(np.arange(n))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ab = fit(idx)
        boots[i] = ab[0] * ab[1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"a": float(a), "b": float(b), "total": float(c), "direct": float(c_prime),
            "indirect": float(a * b), "indirect_ci_low": float(lo), "indirect_ci_high": float(hi),
            "proportion_mediated": float(a * b / c) if c != 0 else float("nan")}
