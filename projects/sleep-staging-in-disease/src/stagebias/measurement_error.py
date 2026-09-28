"""Outcome models with error-prone exposures and two standard corrections: regression calibration and SIMEX.

The exposure (e.g., N3% of TST) is measured without error by human scoring (``x_gold``, available on a
validation subset) and with error by an automated stager (``x_err``, available on everyone). The functions
are estimator-agnostic: ``fit_fn(x, df) -> float`` returns the coefficient of interest (e.g., log HR per unit
exposure) from any model.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

FitFn = Callable[[np.ndarray, pd.DataFrame], float]


# ----------------------------------------------------------------------------------------------------------
# Outcome models
# ----------------------------------------------------------------------------------------------------------

def cox_log_hr(x: np.ndarray, df: pd.DataFrame, time_col: str = "time", event_col: str = "event",
               covariates: Sequence[str] = ()) -> float:
    """Log hazard ratio per unit of ``x`` from a Cox model with optional covariates (statsmodels PHReg)."""
    X = np.column_stack([np.asarray(x, dtype=float)] + [df[c].to_numpy(dtype=float) for c in covariates])
    model = PHReg(df[time_col].to_numpy(dtype=float), X, status=df[event_col].to_numpy(dtype=int), ties="efron")
    res = model.fit(disp=0)
    return float(res.params[0])


def logistic_log_or(x: np.ndarray, df: pd.DataFrame, outcome_col: str = "outcome",
                    covariates: Sequence[str] = ()) -> float:
    """Log odds ratio per unit of ``x`` from a logistic regression with optional covariates."""
    X = np.column_stack([np.asarray(x, dtype=float)] + [df[c].to_numpy(dtype=float) for c in covariates])
    X = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df[outcome_col].to_numpy(dtype=int), X).fit(disp=0, maxiter=200)
    return float(res.params[1])


def swap_in_comparison(fit_fn: FitFn, df: pd.DataFrame, exposures: Dict[str, str], n_boot: int = 500,
                       rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Fit the same outcome model with each exposure version and bootstrap the *paired* differences.

    ``exposures`` maps a label (e.g., 'human', 'argmax', 'expected') to a column of ``df``. The first label
    is the reference; the table reports each coefficient, its bootstrap CI, and the CI of (coef - reference).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    labels = list(exposures)
    point = {k: fit_fn(df[v].to_numpy(), df) for k, v in exposures.items()}
    boots = {k: np.empty(n_boot) for k in labels}
    n = len(df)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        dfb = df.iloc[idx].reset_index(drop=True)
        for k, v in exposures.items():
            try:
                boots[k][b] = fit_fn(dfb[v].to_numpy(), dfb)
            except Exception:  # singular bootstrap sample
                boots[k][b] = np.nan
    rows = []
    ref = labels[0]
    for k in labels:
        diff = boots[k] - boots[ref]
        rows.append({"exposure": k, "coef": point[k], "ci_low": np.nanpercentile(boots[k], 2.5),
                     "ci_high": np.nanpercentile(boots[k], 97.5), "diff_vs_ref": point[k] - point[ref],
                     "diff_ci_low": np.nanpercentile(diff, 2.5), "diff_ci_high": np.nanpercentile(diff, 97.5)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Regression calibration
# ----------------------------------------------------------------------------------------------------------

def regression_calibration(fit_fn: FitFn, df: pd.DataFrame, x_err_col: str, x_gold_col: str,
                           calib_covariates: Sequence[str] = ()) -> Dict[str, float]:
    """Regression calibration (Rosner, Spiegelman & Willett, 1990).

    Fit E[X_gold | X_err, Z] on the validation subset (rows where ``x_gold_col`` is not NaN), predict the
    calibrated exposure for everyone, and refit the outcome model on it. Returns the naive and corrected
    coefficients, the calibration slope, and the reliability ratio (attenuation factor) lambda.
    """
    val = df[~df[x_gold_col].isna()]
    Z = sm.add_constant(val[[x_err_col, *calib_covariates]].astype(float), has_constant="add")
    calib = sm.OLS(val[x_gold_col].astype(float), Z).fit()
    Zall = sm.add_constant(df[[x_err_col, *calib_covariates]].astype(float), has_constant="add")
    x_cal = calib.predict(Zall).to_numpy()
    naive = fit_fn(df[x_err_col].to_numpy(dtype=float), df)
    corrected = fit_fn(x_cal, df)
    return {"naive": naive, "corrected": corrected, "calibration_slope": float(calib.params[x_err_col]),
            "reliability_ratio": float(calib.params[x_err_col]), "n_validation": int(len(val)),
            "calibration_r2": float(calib.rsquared)}


# ----------------------------------------------------------------------------------------------------------
# SIMEX
# ----------------------------------------------------------------------------------------------------------

def estimate_error_variance(x_err: np.ndarray, x_gold: np.ndarray) -> float:
    """Classical-error variance estimate var(X_err - X_gold) from a validation subset (NaNs dropped)."""
    d = np.asarray(x_err, dtype=float) - np.asarray(x_gold, dtype=float)
    d = d[~np.isnan(d)]
    return float(d.var(ddof=1))


def simex(fit_fn: FitFn, df: pd.DataFrame, x_err_col: str, sigma_u2: float,
          lambdas: Sequence[float] = (0.5, 1.0, 1.5, 2.0), n_sim: int = 50, extrapolation: str = "quadratic",
          rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """Simulation-extrapolation (Cook & Stefanski, 1994) for a classical additive error of variance sigma_u2.

    For each lambda, add N(0, lambda * sigma_u2) noise to the error-prone exposure ``n_sim`` times, average
    the coefficient, then extrapolate the curve theta(lambda) back to lambda = -1 (no error).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = df[x_err_col].to_numpy(dtype=float)
    lam = np.array([0.0, *lambdas], dtype=float)
    theta = np.empty_like(lam)
    theta[0] = fit_fn(x, df)
    for i, l in enumerate(lam[1:], start=1):
        vals = []
        for _ in range(n_sim):
            xs = x + rng.normal(0.0, np.sqrt(l * sigma_u2), size=len(x))
            vals.append(fit_fn(xs, df))
        theta[i] = float(np.mean(vals))
    if extrapolation == "quadratic":
        coef = np.polyfit(lam, theta, 2)
    elif extrapolation == "linear":
        coef = np.polyfit(lam, theta, 1)
    else:
        raise ValueError("extrapolation must be 'quadratic' or 'linear'")
    corrected = float(np.polyval(coef, -1.0))
    return {"naive": float(theta[0]), "corrected": corrected, "lambdas": lam, "theta": theta,
            "extrapolation": extrapolation}


# ----------------------------------------------------------------------------------------------------------
# Simulation helpers (used by the tests and for the 'classical-error null' of hypothesis H4)
# ----------------------------------------------------------------------------------------------------------

def simulate_cohort(n: int, beta: float, sigma_u: float, rng: np.random.Generator,
                    differential: float = 0.0, validation_fraction: float = 0.2) -> pd.DataFrame:
    """Simulate a cohort with a true exposure, a binary outcome and an error-prone exposure.

    ``differential`` adds outcome-dependent error (error mean shifts by ``differential`` in cases), which is
    the situation where regression calibration with a non-differential model is no longer sufficient.
    """
    x = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(-1.0 + beta * x)))
    y = rng.binomial(1, p)
    u = rng.normal(0.0, sigma_u, n) + differential * y
    x_err = x + u
    x_gold = x.copy()
    hide = rng.random(n) > validation_fraction
    x_gold[hide] = np.nan
    return pd.DataFrame({"x_true": x, "x_err": x_err, "x_gold": x_gold, "outcome": y})


def classical_error_null(fit_fn: FitFn, df: pd.DataFrame, x_gold_col: str, sigma_u2: float, n_draws: int = 200,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Distribution of the coefficient when the gold exposure is corrupted by *classical* error of variance
    sigma_u2 (the null for 'is the observed automated-vs-human shift larger than classical error explains?')."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = df[x_gold_col].to_numpy(dtype=float)
    out = np.empty(n_draws)
    for i in range(n_draws):
        out[i] = fit_fn(x + rng.normal(0.0, np.sqrt(sigma_u2), len(x)), df)
    return out
