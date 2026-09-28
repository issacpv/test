"""Dose-response models and random-effects pooling across datasets.

``fit_dose_response`` regresses the per-subject behavioural effect on the
individual E-field dose (plus covariates) with statsmodels OLS; ``gain_shape_test``
implements the nested comparison of H2 (global gain vs. residual target
shape); ``dersimonian_laird`` pools per-dataset slopes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class SlopeEstimate:
    slope: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    r2: float


def fit_dose_response(
    effect: np.ndarray,
    dose: np.ndarray,
    covariates: pd.DataFrame | None = None,
    standardize_dose: bool = True,
) -> SlopeEstimate:
    """OLS of behavioural effect on individual dose.

    Parameters
    ----------
    effect : (n,) behavioural effect per subject (positive = intended direction).
    dose : (n,) dose metric per subject (e.g. ROI mean |E| at the applied current).
    covariates : optional DataFrame of adjustment variables (age, sex, ...).
    standardize_dose : if True, the slope is per SD of dose (comparable across datasets).
    """
    y = np.asarray(effect, float)
    d = np.asarray(dose, float)
    if standardize_dose:
        d = (d - d.mean()) / d.std(ddof=1)
    X = pd.DataFrame({"dose": d})
    if covariates is not None:
        X = pd.concat([X, covariates.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, X).fit()
    ci = res.conf_int().loc["dose"]
    return SlopeEstimate(
        slope=float(res.params["dose"]),
        se=float(res.bse["dose"]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        p_value=float(res.pvalues["dose"]),
        n=int(res.nobs),
        r2=float(res.rsquared),
    )


def gain_shape_test(effect: np.ndarray, roi_dose: np.ndarray, gain: np.ndarray) -> dict[str, float]:
    """Nested-model test: does ROI dose add to the global gain? (H2)

    Fits ``effect ~ gain`` and ``effect ~ gain + resid(roi_dose | gain)`` and
    returns the F-test p-value for the residual-shape term plus both R^2.
    """
    y = np.asarray(effect, float)
    g = np.asarray(gain, float)
    r = np.asarray(roi_dose, float)
    # residualise ROI dose on gain so the added term is pure "shape"
    Xg = sm.add_constant(g)
    shape = r - sm.OLS(r, Xg).fit().predict(Xg)
    m0 = sm.OLS(y, Xg).fit()
    m1 = sm.OLS(y, sm.add_constant(np.column_stack([g, shape]))).fit()
    f, p, _ = m1.compare_f_test(m0)
    return {"r2_gain": float(m0.rsquared), "r2_gain_shape": float(m1.rsquared), "f": float(f), "p_shape": float(p)}


def dersimonian_laird(estimates: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """DerSimonian-Laird random-effects meta-analysis of per-dataset slopes.

    Returns pooled estimate, SE, 95% CI, z-test p, tau^2, I^2 and the 95%
    prediction interval (t-based, k-2 df).
    """
    y = np.asarray(estimates, float)
    v = np.asarray(variances, float)
    k = len(y)
    if k < 2:
        raise ValueError("need at least two studies")
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_re = 1 / (v + tau2)
    pooled = np.sum(w_re * y) / np.sum(w_re)
    se = np.sqrt(1 / np.sum(w_re))
    z = pooled / se
    p = 2 * stats.norm.sf(abs(z))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    if k > 2:
        t = stats.t.ppf(0.975, k - 2)
        pi_half = t * np.sqrt(tau2 + se**2)
    else:
        pi_half = float("nan")
    return {
        "pooled": float(pooled),
        "se": float(se),
        "ci_low": float(pooled - 1.96 * se),
        "ci_high": float(pooled + 1.96 * se),
        "p": float(p),
        "tau2": float(tau2),
        "i2": float(i2),
        "q": float(q),
        "pi_low": float(pooled - pi_half),
        "pi_high": float(pooled + pi_half),
        "k": k,
    }


def attenuation_corrected_slope(slope: float, reliability: float) -> float:
    """Disattenuate a slope for unreliability of the behavioural difference score."""
    if not (0 < reliability <= 1):
        raise ValueError("reliability must be in (0, 1]")
    return slope / np.sqrt(reliability)
