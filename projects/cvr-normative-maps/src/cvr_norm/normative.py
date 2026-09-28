"""Heteroscedastic spline normative model for CVR measures vs age, with covariates.

``y = f(age) + Σ b_j c_j + ε``, ``ε ~ N(0, σ(age)²)``, with ``f`` and ``log σ²`` both cubic
B-splines in age. Covariates (sex, protocol, motion) enter additively in the mean. This is a
transparent, dependency-free stand-in for GAMLSS / PCNtoolkit that gives centiles, z-scores and
percentile ranks; production runs should cross-check with PCNtoolkit's BLR + SHASH.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps
from scipy.interpolate import BSpline

_LOG_CHI2_1_MEAN = -1.2704  # E[log chi2_1] = psi(1/2) + log 2


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.clip(np.asarray(x, float), knots[0], knots[-1])
    t = np.concatenate([[knots[0]] * degree, knots, [knots[-1]] * degree])
    return BSpline.design_matrix(x, t, degree).toarray()


@dataclass
class CVRNormativeModel:
    df: int = 5
    degree: int = 3
    ridge: float = 1e-3
    knots_: np.ndarray = field(default=None, repr=False)
    beta_mean_: np.ndarray = field(default=None, repr=False)
    beta_logvar_: np.ndarray = field(default=None, repr=False)
    n_cov_: int = 0

    def _X(self, age: np.ndarray, cov: np.ndarray | None) -> np.ndarray:
        B = bspline_basis(age, self.knots_, self.degree)
        if self.n_cov_ == 0:
            return B
        C = np.atleast_2d(np.asarray(cov, float))
        C = C.T if C.shape[0] != B.shape[0] else C
        return np.hstack([B, C])

    @property
    def n_basis(self) -> int:
        return self.knots_.size + self.degree - 1

    def fit(self, age: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None) -> "CVRNormativeModel":
        age, y = np.asarray(age, float), np.asarray(y, float)
        q = np.linspace(0, 1, self.df - self.degree + 2)[1:-1]
        self.knots_ = np.concatenate([[age.min()], np.quantile(age, q) if q.size else [], [age.max()]])
        self.n_cov_ = 0 if cov is None else (np.atleast_2d(cov).shape[0] if np.atleast_2d(cov).shape[1] == age.size
                                             and np.atleast_2d(cov).shape[0] != age.size else np.atleast_2d(cov).shape[1])
        X = self._X(age, cov)
        I = self.ridge * np.eye(X.shape[1])
        self.beta_mean_ = np.linalg.solve(X.T @ X + I, X.T @ y)
        e2 = np.log((y - X @ self.beta_mean_) ** 2 + 1e-12)
        # variance depends on age only (spline part); covariate columns are excluded
        Bv = X[:, : self.n_basis]
        Iv = self.ridge * np.eye(Bv.shape[1])
        self.beta_logvar_ = np.linalg.solve(Bv.T @ Bv + Iv, Bv.T @ e2) - _LOG_CHI2_1_MEAN
        return self

    def predict(self, age: np.ndarray, cov: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = self._X(np.asarray(age, float), cov)
        mu = X @ self.beta_mean_
        sd = np.exp(0.5 * (X[:, : self.n_basis] @ self.beta_logvar_))
        return mu, sd

    def zscore(self, age: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        mu, sd = self.predict(age, cov)
        return (np.asarray(y, float) - mu) / sd

    def centile(self, age: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        return 100.0 * sps.norm.cdf(self.zscore(age, y, cov))

    def curves(self, age_grid: np.ndarray, probs=(0.05, 0.25, 0.5, 0.75, 0.95),
               cov: np.ndarray | None = None) -> dict[float, np.ndarray]:
        mu, sd = self.predict(age_grid, cov)
        return {p: mu + sps.norm.ppf(p) * sd for p in probs}


def peak_age(model: CVRNormativeModel, age_grid: np.ndarray, cov: np.ndarray | None = None) -> float:
    """Age at which the fitted mean is maximal (e.g. the CVR-amplitude peak in early adulthood)."""
    mu, _ = model.predict(age_grid, cov)
    return float(age_grid[int(np.argmax(mu))])


def extreme_deviation_rate(z: np.ndarray, thresh: float = 1.96) -> tuple[float, float]:
    """Fractions of participants below -thresh and above +thresh (expected ≈ 2.5 % each under the model)."""
    z = np.asarray(z, float)
    return float(np.mean(z < -thresh)), float(np.mean(z > thresh))
