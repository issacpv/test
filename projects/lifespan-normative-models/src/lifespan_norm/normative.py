"""Normative models with a natural cubic spline age basis, sex and site effects.

Two estimators share one interface (``fit(pheno, y)``, ``predict_quantiles``, ``centile``, ``zscore``):

* :class:`QuantileNormativeModel` — statsmodels ``QuantReg`` at a grid of quantiles (GAMLSS-like
  without a distributional assumption); predicted quantiles are rearranged to be monotone
  (Chernozhukov, Fernández-Val & Galichon, 2010) and the centile of an observation is obtained by
  interpolating between the fitted quantiles.
* :class:`GaussianNormativeModel` — heteroscedastic Gaussian: spline mean, log-variance regression
  on the same basis; z = (y − μ) / σ. Cheap, and it supports shift-scale *site adaptation* from a
  handful of local controls (:meth:`GaussianNormativeModel.adapt_site`).

Both models take ``site`` as a fixed effect during fitting. Sites unseen at fit time are scored with
the site effect set to the (weighted) average site unless adapted.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

_EPS = 1e-6


def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (truncated-power form; Hastie, Tibshirani & Friedman, ESL §5.2.1).

    Returns an n × (K − 1) matrix for K knots (linear term + K − 2 nonlinear terms); the intercept
    is not included.
    """
    x = np.asarray(x, float)
    knots = np.asarray(knots, float)
    K = len(knots)
    if K < 3:
        return x[:, None] if K <= 2 else x[:, None]

    def d(k: int) -> np.ndarray:
        num = np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[K - 1], 0, None) ** 3
        return num / (knots[K - 1] - knots[k])

    dK2 = d(K - 2)
    cols = [x] + [d(k) - dK2 for k in range(K - 2)]
    return np.column_stack(cols)


class _Design:
    """Shared design-matrix builder: intercept + spline(age) + sex + site dummies."""

    def __init__(self, n_knots: int = 5, sex_effect: bool = True, site_effect: bool = True):
        self.n_knots = n_knots
        self.sex_effect = sex_effect
        self.site_effect = site_effect

    def fit(self, pheno: pd.DataFrame) -> "_Design":
        age = pheno["age"].to_numpy(float)
        qs = np.linspace(0.02, 0.98, self.n_knots)
        self.knots_ = np.unique(np.quantile(age, qs))
        self.sites_ = sorted(pheno["site"].astype(str).unique()) if self.site_effect else []
        self.site_weights_ = (pheno["site"].astype(str).value_counts(normalize=True).reindex(self.sites_).fillna(0)
                              .to_numpy() if self.site_effect else np.array([]))
        self.age_range_ = (float(age.min()), float(age.max()))
        return self

    def build(self, pheno: pd.DataFrame, site_override: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(pheno)
        cols = [np.ones(n), natural_spline_basis(pheno["age"].to_numpy(float), self.knots_)]
        if self.sex_effect:
            cols.append((pheno["sex"].astype(str).str.upper().str[0] == "M").to_numpy(float)[:, None])
        if self.site_effect and self.sites_:
            site = pheno["site"].astype(str).to_numpy()
            S = np.zeros((n, len(self.sites_)))
            for j, s in enumerate(self.sites_):
                S[:, j] = site == s
            unseen = ~np.isin(site, self.sites_)
            if unseen.any():
                # unseen site: average site effect (weighted by reference site sizes)
                S[unseen] = self.site_weights_
            if site_override is not None:
                S = site_override
            cols.append(S[:, 1:])  # drop first site as reference level
        return np.column_stack(cols)

    @property
    def site_col_start(self) -> int:
        return 1 + (len(self.knots_) - 1) + (1 if self.sex_effect else 0)


class QuantileNormativeModel:
    """Quantile-regression normative model (see module docstring)."""

    def __init__(self, quantiles: Sequence[float] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99),
                 n_knots: int = 5, sex_effect: bool = True, site_effect: bool = True, max_iter: int = 2000):
        self.quantiles = np.asarray(sorted(quantiles), float)
        self.design = _Design(n_knots, sex_effect, site_effect)
        self.max_iter = max_iter

    def fit(self, pheno: pd.DataFrame, y: np.ndarray) -> "QuantileNormativeModel":
        from statsmodels.regression.quantile_regression import QuantReg

        y = np.asarray(y, float)
        ok = np.isfinite(y) & pheno["age"].notna().to_numpy() & pheno["sex"].notna().to_numpy()
        ph = pheno[ok].reset_index(drop=True)
        self.design.fit(ph)
        X = self.design.build(ph)
        self.params_ = np.column_stack([QuantReg(y[ok], X).fit(q=q, max_iter=self.max_iter).params
                                        for q in self.quantiles])
        self.n_fit_ = int(ok.sum())
        return self

    def predict_quantiles(self, pheno: pd.DataFrame) -> np.ndarray:
        """n × Q matrix of predicted quantiles, rearranged to be non-decreasing in q."""
        Q = self.design.build(pheno) @ self.params_
        return np.sort(Q, axis=1)

    def centile(self, pheno: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Centile in (0, 1) of each observation by interpolation across the fitted quantiles."""
        y = np.asarray(y, float)
        Q = self.predict_quantiles(pheno)
        out = np.full(len(y), np.nan)
        for i in range(len(y)):
            if not np.isfinite(y[i]) or not np.all(np.isfinite(Q[i])):
                continue
            q = Q[i]
            if y[i] <= q[0]:
                # tail extrapolation with the local Gaussian slope between the two lowest quantiles
                out[i] = self._tail(y[i], q[0], q[1], self.quantiles[0], self.quantiles[1])
            elif y[i] >= q[-1]:
                out[i] = self._tail(y[i], q[-1], q[-2], self.quantiles[-1], self.quantiles[-2])
            else:
                out[i] = np.interp(y[i], q, self.quantiles)
        return np.clip(out, _EPS, 1 - _EPS)

    @staticmethod
    def _tail(y: float, q_edge: float, q_inner: float, p_edge: float, p_inner: float) -> float:
        sigma = abs(q_edge - q_inner) / max(abs(norm.ppf(p_edge) - norm.ppf(p_inner)), _EPS)
        z = norm.ppf(p_edge) + (y - q_edge) / max(sigma, _EPS)
        return float(norm.cdf(z))

    def zscore(self, pheno: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return norm.ppf(self.centile(pheno, y))

    def median_curve(self, ages: np.ndarray, sex: str = "F", site: Optional[str] = None) -> np.ndarray:
        ph = pd.DataFrame({"age": ages, "sex": sex, "site": site or (self.design.sites_[0] if self.design.sites_ else "x")})
        return self.predict_quantiles(ph)[:, np.argmin(np.abs(self.quantiles - 0.5))]


class GaussianNormativeModel:
    """Heteroscedastic Gaussian normative model with shift-scale site adaptation."""

    def __init__(self, n_knots: int = 5, sex_effect: bool = True, site_effect: bool = True):
        self.design = _Design(n_knots, sex_effect, site_effect)
        self.site_adapt_: dict[str, tuple[float, float]] = {}

    def fit(self, pheno: pd.DataFrame, y: np.ndarray) -> "GaussianNormativeModel":
        y = np.asarray(y, float)
        ok = np.isfinite(y) & pheno["age"].notna().to_numpy() & pheno["sex"].notna().to_numpy()
        ph = pheno[ok].reset_index(drop=True)
        self.design.fit(ph)
        X = self.design.build(ph)
        self.beta_mu_, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        resid = y[ok] - X @ self.beta_mu_
        # log-variance regression on the same basis (variance function), with a floor
        target = np.log(np.maximum(resid ** 2, 1e-12))
        self.beta_logvar_, *_ = np.linalg.lstsq(X, target, rcond=None)
        # correct the bias of E[log(resid^2)] = log(sigma^2) - 1.27 (for Gaussian residuals)
        self.beta_logvar_[0] += 1.2704
        self.n_fit_ = int(ok.sum())
        return self

    def predict(self, pheno: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.design.build(pheno)
        mu = X @ self.beta_mu_
        sigma = np.exp(0.5 * (X @ self.beta_logvar_))
        return mu, sigma

    def zscore(self, pheno: pd.DataFrame, y: np.ndarray, use_adaptation: bool = True) -> np.ndarray:
        mu, sigma = self.predict(pheno)
        z = (np.asarray(y, float) - mu) / sigma
        if use_adaptation and self.site_adapt_:
            site = pheno["site"].astype(str).to_numpy()
            for s, (shift, scale) in self.site_adapt_.items():
                m = site == s
                z[m] = (z[m] - shift) / scale
        return z

    def centile(self, pheno: pd.DataFrame, y: np.ndarray, use_adaptation: bool = True) -> np.ndarray:
        return np.clip(norm.cdf(self.zscore(pheno, y, use_adaptation)), _EPS, 1 - _EPS)

    def predict_quantiles(self, pheno: pd.DataFrame, quantiles: Sequence[float] = (0.025, 0.5, 0.975)) -> np.ndarray:
        mu, sigma = self.predict(pheno)
        return mu[:, None] + sigma[:, None] * norm.ppf(np.asarray(quantiles))[None, :]

    def adapt_site(self, pheno_calib: pd.DataFrame, y_calib: np.ndarray, site: str,
                   min_n: int = 5) -> tuple[float, float]:
        """Estimate a shift/scale of z for a new site from local controls (unadapted z of the
        calibration subjects); with fewer than ``min_n`` subjects only the shift is estimated."""
        z = self.zscore(pheno_calib, y_calib, use_adaptation=False)
        z = z[np.isfinite(z)]
        shift = float(np.mean(z)) if len(z) else 0.0
        scale = float(np.std(z, ddof=1)) if len(z) >= min_n else 1.0
        self.site_adapt_[site] = (shift, max(scale, 0.2))
        return self.site_adapt_[site]


def fit_pcntoolkit_blr(pheno: pd.DataFrame, y: np.ndarray, out_dir: str = "outputs/pcn"):
    """Hook for PCNtoolkit (``pip install pcntoolkit``): warped Bayesian linear regression with a
    B-spline age basis and site as a batch effect, matching Rutherford et al. (2022). Writes the
    covariate/response files PCNtoolkit expects and calls ``estimate``; see its documentation for
    the returned artefacts (yhat, ys2, Z).
    """
    try:
        import pcntoolkit as pcn  # noqa: F401
        from pcntoolkit.normative import estimate
        from pcntoolkit.util.utils import create_bspline_basis
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pcntoolkit not installed") from exc
    import os

    os.makedirs(out_dir, exist_ok=True)
    age = pheno["age"].to_numpy(float)
    B = create_bspline_basis(age.min(), age.max())
    Phi = np.array([B(a) for a in age])
    sex = (pheno["sex"].astype(str).str.upper().str[0] == "M").to_numpy(float)[:, None]
    sites = pd.get_dummies(pheno["site"].astype(str)).to_numpy(float)
    X = np.column_stack([age, sex, sites, Phi])
    np.savetxt(os.path.join(out_dir, "cov.txt"), X)
    np.savetxt(os.path.join(out_dir, "resp.txt"), np.asarray(y, float))
    return estimate(os.path.join(out_dir, "cov.txt"), os.path.join(out_dir, "resp.txt"), alg="blr",
                    optimizer="powell", warp="WarpSinArcsinh", savemodel=True, standardize=False,
                    output_path=out_dir)
