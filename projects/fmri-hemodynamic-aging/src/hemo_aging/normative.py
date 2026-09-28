"""Normative modelling of hemodynamic features against age.

* :class:`SplineNormativeModel` – heteroscedastic Gaussian model with cubic B-spline mean
  and log-variance (a small-footprint stand-in for GAMLSS / PCNtoolkit); gives z-scores and centiles.
* :func:`hemodynamic_age_delta` – ridge "hemodynamic age" with participant-grouped CV and
  training-fold bias correction (Beheshti et al., 2019, NeuroImage: Clinical style).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Design matrix of a B-spline basis with interior ``knots`` (clamped ends)."""
    x = np.asarray(x, dtype=float)
    lo, hi = knots[0], knots[-1]
    t = np.concatenate([[lo] * degree, knots, [hi] * degree])
    xc = np.clip(x, lo, hi)
    return BSpline.design_matrix(xc, t, degree).toarray()


@dataclass
class SplineNormativeModel:
    """Mean and log-variance of ``y`` as smooth functions of ``age`` (optionally by ``sex``)."""

    df: int = 4
    degree: int = 3
    ridge: float = 1e-3
    knots_: np.ndarray = field(default=None, repr=False)
    beta_mean_: np.ndarray = field(default=None, repr=False)
    beta_logvar_: np.ndarray = field(default=None, repr=False)
    n_groups_: int = 1

    def _design(self, age: np.ndarray, group: np.ndarray | None) -> np.ndarray:
        B = bspline_basis(age, self.knots_, self.degree)
        if self.n_groups_ == 1:
            return B
        g = np.zeros((age.size, self.n_groups_))
        g[np.arange(age.size), np.asarray(group, int)] = 1.0
        return np.hstack([B, g[:, 1:]])  # additive group offsets

    def fit(self, age: np.ndarray, y: np.ndarray, group: np.ndarray | None = None) -> "SplineNormativeModel":
        age, y = np.asarray(age, float), np.asarray(y, float)
        q = np.linspace(0, 1, self.df - self.degree + 2)[1:-1]
        interior = np.quantile(age, q) if q.size else np.array([])
        self.knots_ = np.concatenate([[age.min()], interior, [age.max()]])
        self.n_groups_ = 1 if group is None else int(np.max(group)) + 1
        X = self._design(age, group)
        I = self.ridge * np.eye(X.shape[1])
        self.beta_mean_ = np.linalg.solve(X.T @ X + I, X.T @ y)
        resid2 = np.log((y - X @ self.beta_mean_) ** 2 + 1e-12)
        self.beta_logvar_ = np.linalg.solve(X.T @ X + I, X.T @ resid2)
        # log(e^2) is biased for log(sigma^2) by E[log chi2_1] = psi(1/2) + log 2 = -1.2704.
        # B-spline bases sum to one, so adding the constant to every spline coefficient
        # shifts the fitted log-variance uniformly (there is no separate intercept column).
        n_basis = self.knots_.size + self.degree - 1
        self.beta_logvar_[:n_basis] += 1.2704
        return self

    def predict(self, age: np.ndarray, group: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, sd) at ``age``."""
        X = self._design(np.asarray(age, float), group)
        mu = X @ self.beta_mean_
        sd = np.exp(0.5 * (X @ self.beta_logvar_))
        return mu, sd

    def zscore(self, age: np.ndarray, y: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
        mu, sd = self.predict(age, group)
        return (np.asarray(y, float) - mu) / sd

    def centiles(self, age: np.ndarray, probs=(0.05, 0.25, 0.5, 0.75, 0.95),
                 group: np.ndarray | None = None) -> dict[float, np.ndarray]:
        mu, sd = self.predict(age, group)
        return {p: mu + sps.norm.ppf(p) * sd for p in probs}

    def centile_of(self, age: np.ndarray, y: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
        return sps.norm.cdf(self.zscore(age, y, group))


def hemodynamic_age_delta(features: np.ndarray, age: np.ndarray, groups: np.ndarray, n_splits: int = 10,
                          alpha: float = 10.0, bias_correct: bool = True,
                          rng: np.random.Generator | None = None) -> dict[str, np.ndarray | float]:
    """Cross-validated 'hemodynamic age' and bias-corrected delta.

    Bias correction fits ``delta ~ age`` on training-fold out-of-bag predictions (inner
    5-fold) and removes that slope from the test fold, so the test delta is uncorrelated
    with age without touching test data.
    """
    X, age, groups = np.asarray(features, float), np.asarray(age, float), np.asarray(groups)
    pred = np.full(age.shape, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, age, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        model = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, age[tr])
        p_te = model.predict((X[te] - mu) / sd)
        if bias_correct:
            inner = np.full(tr.size, np.nan)
            for itr, ite in GroupKFold(n_splits=5).split(X[tr], age[tr], groups[tr]):
                m2 = Ridge(alpha=alpha).fit((X[tr][itr] - mu) / sd, age[tr][itr])
                inner[ite] = m2.predict((X[tr][ite] - mu) / sd)
            slope, intercept = np.polyfit(age[tr], inner - age[tr], 1)
            p_te = p_te - (slope * age[te] + intercept)
        pred[te] = p_te
    delta = pred - age
    mae = float(np.mean(np.abs(delta)))
    r = float(np.corrcoef(pred, age)[0, 1])
    return {"predicted_age": pred, "delta": delta, "mae": mae, "r": r,
            "delta_age_corr": float(np.corrcoef(delta, age)[0, 1])}
