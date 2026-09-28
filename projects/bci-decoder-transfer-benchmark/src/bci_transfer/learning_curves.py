"""Calibration-budget protocol and learning-curve estimands.

The central estimand of the benchmark is not accuracy at full calibration but the number
of *target* calibration trials a method needs to reach a given fraction of the subject's
asymptotic accuracy. This module implements

* :func:`stratified_draw` - draw ``k`` calibration trials with balanced classes;
* :func:`calibration_curve` - accuracy at each budget for a user-supplied ``fit_predict``;
* :func:`fit_learning_curve` - inverse-power fit ``acc(k) = a - b (k+1)^(-c)``;
* :func:`trials_to_fraction` and :func:`calibration_savings` - the headline numbers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .covariance import balanced_accuracy

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], np.ndarray]
"""``fit_predict(X_cal, y_cal, X_test, source) -> y_pred``; ``X_cal`` may be empty for budget 0."""


def stratified_draw(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``k`` trials with class counts as equal as possible (all of a class if too few)."""
    y = np.asarray(y)
    if k <= 0:
        return np.zeros(0, dtype=int)
    classes = np.unique(y)
    per_class = np.full(len(classes), k // len(classes))
    per_class[: k % len(classes)] += 1
    chosen = []
    for cls, m in zip(rng.permutation(classes), per_class):
        idx = np.flatnonzero(y == cls)
        chosen.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    return np.concatenate(chosen)


def calibration_curve(fit_predict: FitPredict, X_cal_pool: np.ndarray, y_cal_pool: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray, budgets: Sequence[int] = (0, 5, 10, 20, 40, 80), n_draws: int = 10,
                      source: Any = None, seed: int = 0, metric: Callable[[np.ndarray, np.ndarray], float] = balanced_accuracy
                      ) -> pd.DataFrame:
    """Accuracy on ``X_test`` after calibrating with ``k`` trials drawn from the calibration pool.

    Returns a long DataFrame with columns ``budget, draw, accuracy, n_cal``. Budgets larger
    than the pool are clipped to the pool size (and run once).
    """
    rng = np.random.default_rng(seed)
    rows = []
    n_pool = len(y_cal_pool)
    for k in budgets:
        k_eff = min(int(k), n_pool)
        draws = 1 if (k_eff == 0 or k_eff == n_pool) else n_draws
        for d in range(draws):
            idx = stratified_draw(y_cal_pool, k_eff, rng) if k_eff < n_pool else np.arange(n_pool)
            y_pred = fit_predict(X_cal_pool[idx], y_cal_pool[idx], X_test, source)
            rows.append({"budget": int(k), "draw": d, "accuracy": float(metric(y_test, y_pred)), "n_cal": int(k_eff)})
    return pd.DataFrame(rows)


def _power_curve(k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.power(k + 1.0, -c)


def fit_learning_curve(budgets: Sequence[float], acc: Sequence[float], chance: float = 0.5) -> Dict[str, float]:
    """Fit ``acc(k) = a - b (k+1)^(-c)`` by non-linear least squares.

    ``a`` is the asymptote, ``a - b`` the zero-shot accuracy (k = 0), ``c`` the learning speed.
    Returns nan parameters when the fit fails (fewer than 3 distinct budgets or no convergence).
    """
    k = np.asarray(budgets, dtype=float)
    y = np.asarray(acc, dtype=float)
    ok = np.isfinite(k) & np.isfinite(y)
    k, y = k[ok], y[ok]
    if len(np.unique(k)) < 3:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "rmse": np.nan}
    a0 = float(np.max(y))
    b0 = float(max(a0 - y[np.argmin(k)], 1e-3))
    try:
        popt, _ = curve_fit(_power_curve, k, y, p0=[a0, b0, 0.5], bounds=([chance * 0.5, 0.0, 0.01], [1.0, 1.0, 5.0]),
                            maxfev=20000)
    except (RuntimeError, ValueError):
        return {"a": np.nan, "b": np.nan, "c": np.nan, "rmse": np.nan}
    rmse = float(np.sqrt(np.mean((_power_curve(k, *popt) - y) ** 2)))
    return {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]), "rmse": rmse}


def trials_to_fraction(params: Dict[str, float], fraction: float = 0.9, chance: float = 0.5) -> float:
    """Calibration trials needed to reach ``chance + fraction * (a - chance)`` on the fitted curve.

    Returns 0 if the zero-shot accuracy already exceeds the target, ``inf`` if the curve never
    reaches it (a <= chance), nan for a failed fit.
    """
    a, b, c = params.get("a", np.nan), params.get("b", np.nan), params.get("c", np.nan)
    if not np.all(np.isfinite([a, b, c])):
        return float("nan")
    if a <= chance:
        return float("inf")
    target_gap = (1.0 - fraction) * (a - chance)  # allowed distance below the asymptote
    if b <= target_gap:
        return 0.0
    return float((b / target_gap) ** (1.0 / c) - 1.0)


def calibration_savings(params_transfer: Dict[str, float], params_scratch: Dict[str, float], fraction: float = 0.9,
                        chance: float = 0.5) -> Dict[str, float]:
    """Trials saved by transfer to reach the *scratch* curve's target accuracy.

    The target is defined on the from-scratch curve (``chance + fraction*(a_scratch - chance)``)
    so that both methods are held to the same absolute accuracy.
    """
    a_s = params_scratch.get("a", np.nan)
    if not np.isfinite(a_s) or a_s <= chance:
        return {"k_scratch": np.nan, "k_transfer": np.nan, "saved": np.nan, "ratio": np.nan}
    target = chance + fraction * (a_s - chance)
    k_s = trials_to_fraction(params_scratch, fraction, chance)
    # trials for the transfer curve to reach the same absolute target
    a_t, b_t, c_t = params_transfer.get("a", np.nan), params_transfer.get("b", np.nan), params_transfer.get("c", np.nan)
    if not np.all(np.isfinite([a_t, b_t, c_t])):
        k_t = float("nan")
    elif a_t <= target:
        k_t = float("inf")
    elif a_t - b_t >= target:
        k_t = 0.0
    else:
        k_t = float((b_t / (a_t - target)) ** (1.0 / c_t) - 1.0)
    saved = k_s - k_t if np.isfinite(k_s) and np.isfinite(k_t) else np.nan
    ratio = k_t / k_s if np.isfinite(k_s) and np.isfinite(k_t) and k_s > 0 else np.nan
    return {"k_scratch": float(k_s), "k_transfer": float(k_t), "saved": float(saved), "ratio": float(ratio)}


def auc_learning_curve(budgets: Sequence[float], acc: Sequence[float], chance: float = 0.5) -> float:
    """Area between the accuracy curve and chance over ``log(k+1)``, normalised by the log range."""
    k = np.log(np.asarray(budgets, dtype=float) + 1.0)
    y = np.asarray(acc, dtype=float) - chance
    order = np.argsort(k)
    k, y = k[order], y[order]
    if k[-1] == k[0]:
        return float(y.mean())
    return float(np.trapz(y, k) / (k[-1] - k[0]))


def summarise_curve(df: pd.DataFrame, chance: float = 0.5, fraction: float = 0.9) -> Dict[str, float]:
    """Mean accuracy per budget -> fitted curve -> trials-to-fraction and AUC."""
    m = df.groupby("budget")["accuracy"].mean()
    params = fit_learning_curve(m.index.to_numpy(), m.to_numpy(), chance)
    return {**params, "k_to_fraction": trials_to_fraction(params, fraction, chance),
            "auc": auc_learning_curve(m.index.to_numpy(), m.to_numpy(), chance),
            "zero_shot": float(m.get(0, np.nan)), "max_budget_acc": float(m.iloc[-1])}


__all__ = ["stratified_draw", "calibration_curve", "fit_learning_curve", "trials_to_fraction", "calibration_savings",
           "auc_learning_curve", "summarise_curve"]
