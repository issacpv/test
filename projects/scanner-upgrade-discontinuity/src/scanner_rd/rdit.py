"""Regression discontinuity in time (RDiT) with subject fixed effects.

Model (subject *i*, session *j*, running variable ``x = t_rel - cutoff``, ``D = 1[x >= 0]``)::

    y_ij = a_i + b1 * x + tau * D + b3 * D * x + gamma' Z_ij + e_ij

estimated by kernel-weighted least squares within a bandwidth ``h`` (triangular kernel), after a
weighted within-subject transformation (fixed effects), with cluster-robust standard errors by
subject. ``tau`` is the level shift at the upgrade, ``b3`` the slope change.

Credibility checks: :func:`placebo_cutoffs`, donut exclusion (``donut``), bandwidth sensitivity,
and covariate continuity (run the estimator with a covariate as outcome).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RDResult:
    """Estimates from :func:`local_linear_rd`."""

    jump: float
    jump_se: float
    slope_pre: float
    slope_change: float
    slope_change_se: float
    bandwidth: float
    n_obs: int
    n_subjects: int
    n_pre: int
    n_post: int
    coef: Dict[str, float] = field(default_factory=dict)

    @property
    def jump_z(self) -> float:
        return self.jump / self.jump_se if self.jump_se > 0 else np.nan

    @property
    def jump_ci95(self) -> tuple:
        return (self.jump - 1.96 * self.jump_se, self.jump + 1.96 * self.jump_se)


def _kernel_weights(x: np.ndarray, h: float, kind: str) -> np.ndarray:
    u = np.abs(x) / h
    if kind == "triangular":
        return np.clip(1.0 - u, 0.0, None)
    if kind == "uniform":
        return (u <= 1.0).astype(float)
    if kind == "epanechnikov":
        return np.clip(0.75 * (1.0 - u ** 2), 0.0, None)
    raise ValueError(f"unknown kernel {kind!r}")


def _within_transform(M: np.ndarray, groups: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Subtract weighted group means column-wise (weighted fixed-effects transformation)."""
    M = np.asarray(M, dtype=float)
    out = M.copy()
    for g in np.unique(groups):
        idx = groups == g
        wg = w[idx]
        sw = wg.sum()
        if sw <= 0:
            continue
        mean = (wg[:, None] * M[idx]).sum(axis=0) / sw
        out[idx] = M[idx] - mean
    return out


def _wls_cluster(X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, df_adjust: int = 0):
    """Weighted least squares with cluster-robust (CR1) covariance."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    meat = np.zeros_like(XtX)
    uniq = np.unique(groups)
    for g in uniq:
        idx = groups == g
        s = Xw[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G = len(uniq)
    n, k = X.shape
    k_eff = k + df_adjust
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k_eff, 1))
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, V


def local_linear_rd(
    df: pd.DataFrame,
    outcome: str,
    time_col: str = "t_rel",
    subject_col: str = "subject",
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    covariates: Sequence[str] = (),
    kernel: str = "triangular",
    donut: float = 0.0,
    fixed_effects: bool = True,
) -> RDResult:
    """Local-linear RD estimate of the level shift at ``cutoff``.

    Parameters
    ----------
    df
        Long table, one row per session.
    outcome
        Column with the imaging feature (e.g. hippocampal volume, % of first session).
    bandwidth
        Half-width of the estimation window in the units of ``time_col`` (years). ``None`` uses
        :func:`select_bandwidth_cv`.
    donut
        Sessions with ``|x| < donut`` are dropped (protocol shake-down period).
    fixed_effects
        Weighted within-subject transformation; identifies ``tau`` from subjects observed on
        both sides. If ``False`` a pooled regression with an intercept is fitted.
    """
    d = df[[outcome, time_col, subject_col, *covariates]].dropna()
    x = d[time_col].to_numpy(float) - cutoff
    if bandwidth is None:
        bandwidth = select_bandwidth_cv(x, d[outcome].to_numpy(float))
    keep = (np.abs(x) <= bandwidth) & (np.abs(x) >= donut)
    d = d.loc[keep]
    x = x[keep]
    if len(d) < 5:
        raise ValueError("fewer than 5 observations within the bandwidth")
    y = d[outcome].to_numpy(float)
    D = (x >= 0).astype(float)
    groups = d[subject_col].to_numpy()
    w = _kernel_weights(x, bandwidth, kernel)

    cols = ["x", "D", "D_x", *covariates]
    X = np.column_stack([x, D, D * x] + [d[c].to_numpy(float) for c in covariates])
    n_fe = 0
    if fixed_effects:
        X = _within_transform(X, groups, w)
        y = _within_transform(y[:, None], groups, w)[:, 0]
        n_fe = len(np.unique(groups))
    else:
        X = np.column_stack([np.ones(len(x)), X])
        cols = ["const", *cols]

    beta, V = _wls_cluster(X, y, w, groups, df_adjust=n_fe)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    coef = {c: float(b) for c, b in zip(cols, beta)}
    i_x, i_D, i_Dx = cols.index("x"), cols.index("D"), cols.index("D_x")
    return RDResult(
        jump=float(beta[i_D]),
        jump_se=float(se[i_D]),
        slope_pre=float(beta[i_x]),
        slope_change=float(beta[i_Dx]),
        slope_change_se=float(se[i_Dx]),
        bandwidth=float(bandwidth),
        n_obs=int(len(y)),
        n_subjects=int(len(np.unique(groups))),
        n_pre=int((D == 0).sum()),
        n_post=int((D == 1).sum()),
        coef=coef,
    )


def select_bandwidth_cv(
    x: np.ndarray,
    y: np.ndarray,
    grid: Optional[Iterable[float]] = None,
    min_per_side: int = 10,
) -> float:
    """Leave-one-out cross-validated bandwidth for one-sided local-linear fits.

    For each candidate ``h`` and each observation, fit a linear model to the *same-side*
    observations within ``h`` (excluding the observation) and record the squared prediction
    error. The ``h`` with the smallest mean error is returned (Ludwig-Miller style CV, pooled
    across subjects). Candidates that leave fewer than ``min_per_side`` observations on a side
    are skipped.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if grid is None:
        span = max(np.abs(x).max(), 1e-6)
        grid = np.linspace(span / 6, span, 8)
    best_h, best_err = None, np.inf
    for h in grid:
        errs: List[float] = []
        for side in (x < 0, x >= 0):
            xs, ys = x[side], y[side]
            if len(xs) < min_per_side:
                continue
            for i in range(len(xs)):
                m = (np.abs(xs - xs[i]) <= h)
                m[i] = False
                if m.sum() < 3:
                    continue
                A = np.column_stack([np.ones(m.sum()), xs[m]])
                coef, *_ = np.linalg.lstsq(A, ys[m], rcond=None)
                errs.append((ys[i] - (coef[0] + coef[1] * xs[i])) ** 2)
        if errs:
            e = float(np.mean(errs))
            if e < best_err:
                best_err, best_h = e, float(h)
    if best_h is None:
        return float(max(np.abs(x).max(), 1e-6))
    return best_h


def placebo_cutoffs(
    df: pd.DataFrame,
    outcome: str,
    cutoffs: Sequence[float],
    **kwargs,
) -> pd.DataFrame:
    """Re-estimate the jump at fake cutoffs (should be ~0). ``kwargs`` go to :func:`local_linear_rd`.

    Sessions are restricted to one side of the true cutoff (0) so that the real discontinuity
    does not contaminate the placebo estimate: negative placebo cutoffs use only pre-upgrade
    sessions, positive ones only post-upgrade sessions.
    """
    time_col = kwargs.get("time_col", "t_rel")
    rows = []
    for c in cutoffs:
        sub = df[df[time_col] < 0] if c < 0 else df[df[time_col] >= 0]
        try:
            r = local_linear_rd(sub, outcome, cutoff=c, **kwargs)
            rows.append({"cutoff": c, "jump": r.jump, "se": r.jump_se, "n": r.n_obs, "n_subjects": r.n_subjects})
        except (ValueError, np.linalg.LinAlgError):
            rows.append({"cutoff": c, "jump": np.nan, "se": np.nan, "n": 0, "n_subjects": 0})
    return pd.DataFrame(rows)


def aging_equivalent_years(jump: float, slope_per_year: float) -> float:
    """Express a level shift as years of the reference (e.g. cognitively normal) slope."""
    if slope_per_year == 0 or not np.isfinite(slope_per_year):
        return np.nan
    return float(jump / abs(slope_per_year))


def counterfactual_trajectory(df: pd.DataFrame, outcome: str, result: RDResult, time_col: str = "t_rel") -> pd.Series:
    """Post-upgrade values corrected by the estimated jump and slope change."""
    x = df[time_col].to_numpy(float)
    post = (x >= 0).astype(float)
    return df[outcome] - post * (result.jump + result.slope_change * x)
