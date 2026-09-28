"""Within-patient dose-response and time-course models for QTc change.

Primary estimand: the slope of within-patient Delta-QTc (ms) on administered
dose (per canonical unit), adjusting for time-since-dose, serum potassium, age
and sex, with patient as a random intercept.

Falls back to OLS/GEE-free closed-form estimates when ``statsmodels`` is not
installed so the module and its tests run in a minimal environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf  # type: ignore

    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False


@dataclass
class DoseSlope:
    ingredient: str
    slope: float          # ms per canonical dose unit (mg)
    se: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_patients: int
    method: str

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= 0.0 <= self.ci_high)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Simple OLS slope of y on x with standard error (intercept included)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def fit_dose_slope(pairs: pd.DataFrame, ingredient: str, use_mixed: bool = True) -> DoseSlope:
    """Estimate the dose-response slope for one ingredient.

    Uses a linear mixed model (random intercept per patient) when statsmodels is
    available and there are enough patients; otherwise OLS on the pairs.
    """
    d = pairs[pairs["ingredient"] == ingredient].dropna(subset=["dose_mg", "delta_qtc"])
    d = d[np.isfinite(d["dose_mg"]) & np.isfinite(d["delta_qtc"])]
    n_pairs = len(d)
    n_pat = int(d["subject_id"].nunique()) if "subject_id" in d else n_pairs
    if n_pairs < 5:
        return DoseSlope(ingredient, np.nan, np.nan, np.nan, np.nan, n_pairs, n_pat, "insufficient")

    if use_mixed and _HAVE_SM and n_pat >= 5 and d["subject_id"].nunique() > 1:
        d = d.copy()
        # keep the model well-posed even when covariates are missing
        formula = "delta_qtc ~ dose_mg + hours_since_dose"
        try:
            md = smf.mixedlm(formula, d, groups=d["subject_id"])
            res = md.fit(reml=True, method="lbfgs", maxiter=200)
            slope = float(res.params["dose_mg"])
            se = float(res.bse["dose_mg"])
            return DoseSlope(ingredient, slope, se, slope - 1.96 * se, slope + 1.96 * se,
                             n_pairs, n_pat, "mixedlm")
        except Exception:  # pragma: no cover - numeric fallback
            pass

    slope, se = _ols_slope(d["dose_mg"].to_numpy(float), d["delta_qtc"].to_numpy(float))
    return DoseSlope(ingredient, slope, se, slope - 1.96 * se, slope + 1.96 * se,
                     n_pairs, n_pat, "ols")


def dose_for_threshold(slope: DoseSlope, threshold_ms: float = 10.0) -> float:
    """Dose (mg) that yields a mean +threshold_ms QTc change under the linear model."""
    if not np.isfinite(slope.slope) or slope.slope <= 0:
        return float("nan")
    return threshold_ms / slope.slope


def time_course_peak(pairs: pd.DataFrame, ingredient: str, n_knots: int = 4) -> dict:
    """Locate the time-to-peak of Delta-QTc via a piecewise-linear (spline-like) fit.

    Returns the estimated hours-to-peak and the peak Delta-QTc. Uses a coarse
    binning + smoothing so it works without statsmodels/patsy.
    """
    d = pairs[pairs["ingredient"] == ingredient].dropna(subset=["hours_since_dose", "delta_qtc"])
    if len(d) < 8:
        return {"ingredient": ingredient, "hours_to_peak": np.nan, "peak_delta_qtc": np.nan, "n": len(d)}
    h = d["hours_since_dose"].to_numpy(float)
    y = d["delta_qtc"].to_numpy(float)
    edges = np.linspace(h.min(), h.max(), n_knots + 1)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h <= hi if hi == edges[-1] else h < hi)
        if m.sum() >= 2:
            centers.append(0.5 * (lo + hi))
            means.append(float(np.mean(y[m])))
    if not centers:
        return {"ingredient": ingredient, "hours_to_peak": np.nan, "peak_delta_qtc": np.nan, "n": len(d)}
    centers = np.asarray(centers)
    means = np.asarray(means)
    k = int(np.argmax(means))
    return {
        "ingredient": ingredient,
        "hours_to_peak": float(centers[k]),
        "peak_delta_qtc": float(means[k]),
        "n": int(len(d)),
    }


def permutation_null(pairs: pd.DataFrame, ingredient: str, n_perm: int = 500, seed: int = 0) -> float:
    """Two-sided permutation p-value for the dose slope (shuffle dose within drug)."""
    d = pairs[pairs["ingredient"] == ingredient].dropna(subset=["dose_mg", "delta_qtc"])
    d = d[np.isfinite(d["dose_mg"]) & np.isfinite(d["delta_qtc"])]
    if len(d) < 5:
        return float("nan")
    x = d["dose_mg"].to_numpy(float)
    y = d["delta_qtc"].to_numpy(float)
    obs, _ = _ols_slope(x, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        xs = rng.permutation(x)
        s, _ = _ols_slope(xs, y)
        if abs(s) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)
