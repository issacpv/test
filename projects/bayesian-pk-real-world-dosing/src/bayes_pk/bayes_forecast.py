"""MAP-Bayesian individual estimation and forecasting, as implemented in MIPD software.

Given a :class:`~bayes_pk.pk_models.ModelSpec`, a dosing history and observed levels, the posterior mode of the
individual random effects ``eta`` is found by minimising

    -2 log p(eta | y) = sum_i [ (y_i - f_i(eta))^2 / var_i + log var_i ] + sum_j eta_j^2 / omega_j

with ``var_i = (sigma_prop * f_i)^2 + sigma_add^2``.  Laplace standard errors come from the numerical Hessian.
Forecasts of the next level and AUC over a window are computed at the MAP estimate; a Monte-Carlo draw from the
Laplace approximation gives an uncertainty band for AUC.  ``model_average`` combines several models with
weights proportional to ``exp(-OFV/2)`` (an Akaike-type weighting; see Uster et al., 2021, Clin Pharmacol Ther,
for the model-averaging idea).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from .pk_models import ModelSpec, as_dose_frame, auc_interval, concentration


@dataclass
class MapResult:
    """Result of a MAP fit."""

    spec_name: str
    eta: np.ndarray
    params: Dict[str, float]
    ofv: float
    se_eta: np.ndarray
    cov_eta: np.ndarray
    n_levels: int
    success: bool


def _residual_var(spec: ModelSpec, pred: np.ndarray) -> np.ndarray:
    return (spec.sigma_prop * pred) ** 2 + spec.sigma_add**2


def neg2_log_posterior(
    eta: np.ndarray,
    spec: ModelSpec,
    covariates: Mapping[str, float],
    doses: pd.DataFrame,
    level_times: np.ndarray,
    level_values: np.ndarray,
) -> float:
    """-2 log posterior (up to a constant) of ``eta`` given the observed levels."""
    params = spec.individual_params(covariates, eta)
    pred = concentration(spec, params, doses, level_times)
    var = _residual_var(spec, pred)
    ll = np.sum((level_values - pred) ** 2 / var + np.log(var))
    omega = np.array([spec.omega[n] for n in spec.eta_names])
    prior = np.sum(np.asarray(eta) ** 2 / omega)
    return float(ll + prior)


def _numerical_hessian(f, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        for j in range(i, n):
            ej = np.zeros(n)
            ej[j] = h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h**2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h**2)
    return H


def map_estimate(
    spec: ModelSpec,
    covariates: Mapping[str, float],
    doses: pd.DataFrame,
    level_times: Sequence[float],
    level_values: Sequence[float],
) -> MapResult:
    """Posterior-mode estimate of the individual random effects.

    With no levels the estimate is the prior mode (``eta = 0``), i.e. the *a priori* individual.
    """
    doses = as_dose_frame(doses)
    lt = np.asarray(level_times, dtype=float)
    lv = np.asarray(level_values, dtype=float)
    n_eta = len(spec.eta_names)
    omega = np.array([spec.omega[n] for n in spec.eta_names])
    if lt.size == 0:
        eta = np.zeros(n_eta)
        return MapResult(spec.name, eta, spec.individual_params(covariates, eta), 0.0, np.sqrt(omega), np.diag(omega), 0, True)

    def obj(e: np.ndarray) -> float:
        return neg2_log_posterior(e, spec, covariates, doses, lt, lv)

    bounds = [(-4.0, 4.0)] * n_eta
    res = optimize.minimize(obj, np.zeros(n_eta), method="L-BFGS-B", bounds=bounds)
    eta = res.x
    H = _numerical_hessian(obj, eta)
    try:
        cov = 2.0 * np.linalg.inv(H)  # Hessian of -2 log post -> covariance = 2 H^-1
        cov = 0.5 * (cov + cov.T)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.diag(omega)
        se = np.sqrt(omega)
    return MapResult(spec.name, eta, spec.individual_params(covariates, eta), float(res.fun), se, cov, int(lt.size), bool(res.success))


def predict_levels(spec: ModelSpec, result: MapResult, doses: pd.DataFrame, times: Sequence[float]) -> np.ndarray:
    """Model prediction at ``times`` for the individual in ``result``."""
    return concentration(spec, result.params, as_dose_frame(doses), np.asarray(times, dtype=float))


def forecast_next_level(
    spec: ModelSpec,
    covariates: Mapping[str, float],
    doses: pd.DataFrame,
    levels: pd.DataFrame,
    n_used: int,
) -> Dict[str, float]:
    """Fit on the first ``n_used`` levels (sorted by time) and forecast level ``n_used + 1``.

    ``levels`` needs columns ``time_h`` and ``value``. Returns observed, predicted and the fit's OFV. Doses after
    the forecast time are irrelevant for the prediction but harmless (the model is causal).
    """
    lv = levels.sort_values("time_h").reset_index(drop=True)
    if n_used >= len(lv):
        raise ValueError("n_used must be smaller than the number of levels")
    fit = map_estimate(spec, covariates, doses, lv["time_h"].to_numpy()[:n_used], lv["value"].to_numpy()[:n_used])
    t_next = float(lv.loc[n_used, "time_h"])
    pred = float(predict_levels(spec, fit, doses, [t_next])[0])
    return {"time_h": t_next, "observed": float(lv.loc[n_used, "value"]), "predicted": pred, "ofv": fit.ofv, "n_used": n_used}


def auc_with_uncertainty(
    spec: ModelSpec,
    covariates: Mapping[str, float],
    result: MapResult,
    doses: pd.DataFrame,
    t0: float,
    t1: float,
    n_draws: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """AUC over ``[t0, t1]`` at the MAP estimate plus a Laplace Monte-Carlo SD and 90% interval."""
    rng = np.random.default_rng() if rng is None else rng
    auc_map = auc_interval(spec, result.params, doses, t0, t1)
    if n_draws <= 0 or result.n_levels == 0:
        return {"auc": auc_map, "sd": float("nan"), "q05": float("nan"), "q95": float("nan")}
    draws = rng.multivariate_normal(result.eta, result.cov_eta, size=n_draws)
    aucs = np.array([auc_interval(spec, spec.individual_params(covariates, e), doses, t0, t1, n_grid=801) for e in draws])
    return {"auc": auc_map, "sd": float(aucs.std(ddof=1)), "q05": float(np.quantile(aucs, 0.05)), "q95": float(np.quantile(aucs, 0.95))}


def model_average(
    specs: Sequence[ModelSpec],
    covariates: Mapping[str, float],
    doses: pd.DataFrame,
    level_times: Sequence[float],
    level_values: Sequence[float],
    predict_times: Sequence[float],
) -> Dict[str, object]:
    """Fit each model and return OFV-weighted averaged predictions at ``predict_times``.

    Weights are ``exp(-0.5 * (OFV_m - min OFV))`` normalised to 1; with no levels all models get equal weight.
    """
    fits: List[MapResult] = [map_estimate(s, covariates, doses, level_times, level_values) for s in specs]
    ofv = np.array([f.ofv for f in fits])
    if len(level_times) == 0:
        w = np.full(len(specs), 1.0 / len(specs))
    else:
        w = np.exp(-0.5 * (ofv - ofv.min()))
        w = w / w.sum()
    preds = np.vstack([predict_levels(s, f, doses, predict_times) for s, f in zip(specs, fits)])
    return {"weights": dict(zip([s.name for s in specs], w)), "prediction": (w[:, None] * preds).sum(axis=0), "per_model": preds, "fits": fits}
