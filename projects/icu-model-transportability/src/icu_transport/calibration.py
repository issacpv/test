"""Calibration metrics, decision curves and few-shot site recalibration.

Implements the calibration hierarchy of Van Calster et al. (2016/2019): calibration-in-the-large
(intercept), weak calibration (slope), moderate calibration (binned / smoothed curve, ECE, ICI),
plus net benefit (Vickers & Elkin, 2006). Recalibration methods: intercept-only (label-shift
correction), Platt (logistic on the logit), temperature scaling (Guo et al., 2017), isotonic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------------------
# core metrics
# ----------------------------------------------------------------------------------------------


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                               strategy: Literal["quantile", "uniform"] = "quantile") -> float:
    """ECE = sum_b (n_b / n) |mean(y_b) - mean(p_b)|. Quantile bins are the default because ICU
    outcome probabilities are heavily right-skewed and uniform bins leave most bins empty."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _irls_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, fit_slope: bool = True,
                   max_iter: int = 50, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic regression y ~ a + b*x (+ offset). Returns (coef, standard errors)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x]) if fit_slope else np.ones((n, 1))
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = _sigmoid(eta)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta + off
    mu = _sigmoid(eta)
    H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration intercept (in-the-large, fitted with slope fixed at 1) and slope (Cox, 1958).

    Returns intercept, slope, their standard errors and 95% CIs. Perfect calibration:
    intercept = 0, slope = 1. Slope < 1 indicates over-fitting / over-confident predictions.
    """
    y = np.asarray(y, dtype=float)
    lp = _logit(p)
    (a,), (se_a,) = _irls_logistic(lp, y, offset=lp, fit_slope=False)
    (c, b), (_, se_b) = _irls_logistic(lp, y, fit_slope=True)
    return {"intercept": float(a), "intercept_se": float(se_a),
            "intercept_ci": (float(a - 1.96 * se_a), float(a + 1.96 * se_a)),
            "slope": float(b), "slope_se": float(se_b),
            "slope_ci": (float(b - 1.96 * se_b), float(b + 1.96 * se_b)),
            "slope_model_intercept": float(c)}


def integrated_calibration_index(y: np.ndarray, p: np.ndarray, bandwidth: float = 0.05) -> float:
    """ICI (Austin & Steyerberg, 2019): mean |smoothed observed - predicted| over the sample.

    The smoothed observed rate at p_i is a Nadaraya-Watson kernel estimate (Gaussian kernel on
    the probability scale) instead of loess, to keep the dependency footprint small."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    d = (p[:, None] - p[None, :]) / bandwidth
    K = np.exp(-0.5 * d ** 2)
    smooth = (K @ y) / K.sum(axis=1)
    return float(np.mean(np.abs(smooth - p)))


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Murphy (1973) decomposition: Brier = reliability - resolution + uncertainty (binned)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    base = y.mean()
    rel = res = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
            res += m.sum() * (y[m].mean() - base) ** 2
    n = len(y)
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": float(rel / n),
            "resolution": float(res / n), "uncertainty": float(base * (1 - base))}


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Net benefit of "treat if p >= t" vs treat-all and treat-none (Vickers & Elkin, 2006)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    t = np.linspace(0.01, 0.5, 50) if thresholds is None else np.asarray(thresholds, dtype=float)
    n = len(y)
    prev = y.mean()
    nb = np.empty_like(t)
    for i, th in enumerate(t):
        treat = p >= th
        tp = np.sum(treat & (y == 1)) / n
        fp = np.sum(treat & (y == 0)) / n
        nb[i] = tp - fp * th / (1 - th)
    nb_all = prev - (1 - prev) * t / (1 - t)
    return {"threshold": t, "net_benefit": nb, "treat_all": nb_all, "treat_none": np.zeros_like(t)}


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """All scalar calibration metrics in one dict (for tables)."""
    si = calibration_slope_intercept(y, p)
    bd = brier_decomposition(y, p, n_bins)
    return {"intercept": si["intercept"], "slope": si["slope"],
            "ece": expected_calibration_error(y, p, n_bins),
            "ici": integrated_calibration_index(y, p) if len(y) <= 20000 else float("nan"),
            "brier": bd["brier"], "reliability": bd["reliability"], "resolution": bd["resolution"],
            "prevalence": float(np.mean(y)), "mean_pred": float(np.mean(p))}


# ----------------------------------------------------------------------------------------------
# recalibration
# ----------------------------------------------------------------------------------------------

Method = Literal["intercept", "platt", "temperature", "isotonic"]


@dataclass
class Recalibrator:
    method: Method
    params: dict
    transform: Callable[[np.ndarray], np.ndarray]

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.transform(np.asarray(p, dtype=float))


def fit_recalibrator(p: np.ndarray, y: np.ndarray, method: Method = "intercept") -> Recalibrator:
    """Fit a post-hoc recalibration map on a (small) labelled target sample.

    * intercept:   logit(p') = logit(p) + a           (1 parameter; corrects label shift only)
    * platt:       logit(p') = a + b * logit(p)       (2 parameters; intercept + slope)
    * temperature: logit(p') = logit(p) / T           (1 parameter; slope only, Guo et al., 2017)
    * isotonic:    monotone step function              (non-parametric; needs n >~ 1000)
    """
    y = np.asarray(y, dtype=float)
    lp = _logit(p)
    if method == "intercept":
        (a,), _ = _irls_logistic(lp, y, offset=lp, fit_slope=False)
        return Recalibrator(method, {"a": float(a)}, lambda q: _sigmoid(_logit(q) + a))
    if method == "platt":
        (a, b), _ = _irls_logistic(lp, y, fit_slope=True)
        return Recalibrator(method, {"a": float(a), "b": float(b)}, lambda q: _sigmoid(a + b * _logit(q)))
    if method == "temperature":
        def nll(T: float) -> float:
            q = np.clip(_sigmoid(lp / T), EPS, 1 - EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))

        T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        return Recalibrator(method, {"T": T}, lambda q: _sigmoid(_logit(q) / T))
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, dtype=float), y)
        return Recalibrator(method, {"n_knots": int(len(iso.X_thresholds_))},
                            lambda q: np.clip(iso.predict(np.asarray(q, dtype=float)), EPS, 1 - EPS))
    raise ValueError(method)


def few_shot_learning_curve(p_target: np.ndarray, y_target: np.ndarray,
                            n_grid: tuple[int, ...] = (25, 50, 100, 200, 500, 1000, 2000),
                            methods: tuple[Method, ...] = ("intercept", "platt", "temperature", "isotonic"),
                            n_repeats: int = 100, groups: np.ndarray | None = None,
                            random_state: int = 0) -> list[dict[str, float]]:
    """How many labelled target stays does each recalibration method need?

    For each n and repeat: sample n target rows (stratified so at least 2 events are present),
    fit the recalibrator on them, evaluate intercept / slope / ECE / Brier on the remaining rows.
    With ``groups`` (e.g. sex), also fit group-wise recalibrators using n rows *per group* and
    report the max-min ECE gap across groups. Returns a list of rows (one per n x method x
    repeat) ready for ``pd.DataFrame``.
    """
    rng = np.random.default_rng(random_state)
    p = np.asarray(p_target, dtype=float)
    y = np.asarray(y_target, dtype=int)
    n_total = len(y)
    rows = []
    base = calibration_report(y, p)
    for n in n_grid:
        if n >= n_total * 0.8:
            continue
        for rep in range(n_repeats):
            for _ in range(20):
                idx = rng.choice(n_total, n, replace=False)
                if y[idx].sum() >= 2 and (1 - y[idx]).sum() >= 2:
                    break
            mask = np.zeros(n_total, bool)
            mask[idx] = True
            for m in methods:
                if m == "isotonic" and n < 50:
                    continue
                rc = fit_recalibrator(p[mask], y[mask], m)
                q = rc(p[~mask])
                rep_row = calibration_report(y[~mask], q)
                row = {"n": n, "method": m, "repeat": rep, "groupwise": 0,
                       "intercept": rep_row["intercept"], "slope": rep_row["slope"],
                       "ece": rep_row["ece"], "brier": rep_row["brier"],
                       "intercept_removed_frac": 1 - abs(rep_row["intercept"]) / max(abs(base["intercept"]), 1e-6)}
                if groups is not None:
                    g = np.asarray(groups)
                    eces = []
                    for lvl in np.unique(g):
                        gm = (g == lvl) & ~mask
                        if gm.sum() > 20 and len(np.unique(y[gm])) == 2:
                            eces.append(expected_calibration_error(y[gm], q[gm]))
                    row["ece_group_gap"] = float(max(eces) - min(eces)) if len(eces) > 1 else float("nan")
                rows.append(row)
    return rows
