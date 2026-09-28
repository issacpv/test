"""Normative (centile) curves for sleep metrics vs age, GAMLSS-like via quantile regression.

A cubic B-spline basis of age is fitted with linear quantile regression at several
quantiles (sklearn ``QuantileRegressor``), optionally per sex. Quantile crossing is
removed by monotone rearrangement at prediction time. Centiles/z-scores of new
observations are obtained by interpolating between the fitted quantile curves.
For a full distributional GAMLSS (BCCG/BCT) fit use R's ``gamlss`` on the same
design; this module is the dependency-light Python counterpart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


def bspline_knots(x: np.ndarray, df: int = 5, degree: int = 3, x_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Knot vector with ``df`` basis functions: quantile-placed interior knots plus clamped ends."""
    lo, hi = (float(np.min(x)), float(np.max(x))) if x_range is None else x_range
    n_int = df - degree - 1
    if n_int < 0:
        raise ValueError("df must be >= degree + 1")
    interior = np.quantile(x, np.linspace(0, 1, n_int + 2)[1:-1]) if n_int > 0 else np.empty(0)
    return np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])


def bspline_design(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Dense B-spline design matrix; ``x`` is clipped to the knot range (no extrapolation)."""
    x = np.clip(np.asarray(x, float), knots[degree], knots[-degree - 1])
    return BSpline.design_matrix(x, knots, degree).toarray()


@dataclass
class NormativeModel:
    """Quantile-regression normative model of ``y`` as a function of age (and optionally sex)."""

    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    df: int = 5
    degree: int = 3
    alpha: float = 0.0
    knots_: Optional[np.ndarray] = None
    coefs_: Dict[Tuple[str, float], np.ndarray] = field(default_factory=dict)
    groups_: Tuple[str, ...] = ("all",)

    def _key(self, sex: Optional[str]) -> str:
        return "all" if sex is None or sex not in self.groups_ else sex

    def fit(self, age: np.ndarray, y: np.ndarray, sex: Optional[Sequence[str]] = None,
            x_range: Optional[Tuple[float, float]] = None) -> "NormativeModel":
        from sklearn.linear_model import QuantileRegressor

        age = np.asarray(age, float); y = np.asarray(y, float)
        ok = np.isfinite(age) & np.isfinite(y)
        age, y = age[ok], y[ok]
        sex_arr = None if sex is None else np.asarray(sex, str)[ok]
        self.knots_ = bspline_knots(age, self.df, self.degree, x_range)
        B = bspline_design(age, self.knots_, self.degree)
        groups = ["all"] + (sorted(set(sex_arr.tolist())) if sex_arr is not None else [])
        self.groups_ = tuple(groups)
        for g in groups:
            sel = np.ones(len(age), bool) if g == "all" else (sex_arr == g)
            for q in self.quantiles:
                qr = QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs", fit_intercept=True)
                qr.fit(B[sel], y[sel])
                self.coefs_[(g, q)] = np.concatenate([[qr.intercept_], qr.coef_])
        return self

    def predict(self, age: np.ndarray, sex: Optional[str] = None) -> np.ndarray:
        """Quantile curves at ``age``: array (n_age, n_quantiles), monotone in the quantile axis."""
        if self.knots_ is None:
            raise RuntimeError("fit first")
        B = bspline_design(np.atleast_1d(age), self.knots_, self.degree)
        X = np.column_stack([np.ones(len(B)), B])
        g = self._key(sex)
        Q = np.column_stack([X @ self.coefs_[(g, q)] for q in self.quantiles])
        return np.sort(Q, axis=1)  # rearrangement removes quantile crossing

    def centile(self, age: np.ndarray, y: np.ndarray, sex: Optional[str] = None) -> np.ndarray:
        """Percentile (0-100) of each observation relative to the fitted curves (linear interpolation
        between quantile curves; tail values are extrapolated with the normal-tail assumption)."""
        Q = self.predict(age, sex)
        y = np.atleast_1d(np.asarray(y, float))
        qs = np.asarray(self.quantiles)
        out = np.empty(len(y))
        zq = norm.ppf(qs)
        for i in range(len(y)):
            row = Q[i]
            if y[i] <= row[0] or y[i] >= row[-1]:
                # normal-tail extrapolation using the two outermost quantiles
                j = (0, 1) if y[i] <= row[0] else (-2, -1)
                slope = (row[j[1]] - row[j[0]]) / max(zq[j[1]] - zq[j[0]], 1e-9)
                z = zq[j[0]] + (y[i] - row[j[0]]) / max(slope, 1e-9)
                out[i] = 100 * norm.cdf(z)
            else:
                out[i] = 100 * np.interp(y[i], row, qs)
        return out

    def zscore(self, age: np.ndarray, y: np.ndarray, sex: Optional[str] = None) -> np.ndarray:
        c = np.clip(self.centile(age, y, sex) / 100.0, 1e-6, 1 - 1e-6)
        return norm.ppf(c)


def fit_normative(age: np.ndarray, y: np.ndarray, sex: Optional[Sequence[str]] = None,
                  quantiles: Sequence[float] = DEFAULT_QUANTILES, df: int = 5) -> NormativeModel:
    return NormativeModel(tuple(quantiles), df).fit(age, y, sex)


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def centile_calibration(model: NormativeModel, age: np.ndarray, y: np.ndarray, sex: Optional[str] = None) -> Dict[float, float]:
    """Fraction of held-out observations below each fitted quantile curve (ideal: equals the quantile)."""
    Q = model.predict(age, sex)
    y = np.asarray(y, float)
    return {q: float(np.mean(y < Q[:, i])) for i, q in enumerate(model.quantiles)}


def standardized_age_slope(age: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0,
                           groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """OLS slope of z-scored ``y`` on age (per decade) with a (cluster) bootstrap CI."""
    rng = np.random.default_rng(seed)
    age = np.asarray(age, float); y = np.asarray(y, float)
    ok = np.isfinite(age) & np.isfinite(y)
    age, y = age[ok], y[ok]
    g = None if groups is None else np.asarray(groups)[ok]
    z = (y - y.mean()) / y.std()

    def slope(a, zz):
        return np.polyfit(a / 10.0, zz, 1)[0]

    obs = slope(age, z)
    boots = np.empty(n_boot)
    if g is None:
        for i in range(n_boot):
            idx = rng.integers(0, len(age), len(age))
            boots[i] = slope(age[idx], z[idx])
    else:
        ug = np.unique(g)
        members = {u: np.where(g == u)[0] for u in ug}
        for i in range(n_boot):
            idx = np.concatenate([members[u] for u in rng.choice(ug, len(ug))])
            boots[i] = slope(age[idx], z[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope_sd_per_decade": float(obs), "ci_low": float(lo), "ci_high": float(hi)}
