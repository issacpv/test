"""Location-scale spline normative model (GAMLSS-lite).

``y = mu(age, sex, site) + sigma(age) * eps``: the mean is a B-spline in age plus sex and site
fixed effects; the log-variance is a B-spline in age. Fitting alternates OLS for the mean and a
log-squared-residual regression for the variance (two rounds of weighted least squares). This
is deliberately simple and dependency-free; use R ``gamlss`` (BCCG/SHASH families) or PCNtoolkit
HBR for the production fits and this model as the transparent reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """B-spline design matrix on ``x`` with interior ``knots`` (clamped ends)."""
    x = np.asarray(x, float)
    knots = np.asarray(knots, float)
    t = np.concatenate([[knots[0]] * degree, knots, [knots[-1]] * degree])
    xc = np.clip(x, knots[0], knots[-1] - 1e-9)
    return BSpline.design_matrix(xc, t, degree).toarray()


@dataclass
class NormativeModel:
    n_knots: int = 6
    degree: int = 3
    age_range: Optional[Tuple[float, float]] = None
    n_rounds: int = 2
    # fitted attributes
    knots_: np.ndarray = field(default=None, repr=False)
    beta_mu_: np.ndarray = field(default=None, repr=False)
    beta_sigma_: np.ndarray = field(default=None, repr=False)
    sites_: np.ndarray = field(default=None, repr=False)
    has_sex_: bool = False

    # ------------------------------------------------------------------ design
    def _design_mu(self, age, sex, site) -> np.ndarray:
        B = bspline_basis(age, self.knots_, self.degree)
        cols = [B]
        if self.has_sex_:
            cols.append(np.asarray(sex, float)[:, None])
        if self.sites_ is not None and len(self.sites_) > 1:
            site = np.asarray(site)
            cols.append(np.column_stack([(site == s).astype(float) for s in self.sites_[1:]]))
        return np.column_stack(cols)

    def _design_sigma(self, age) -> np.ndarray:
        return bspline_basis(age, self.knots_, self.degree)

    # --------------------------------------------------------------------- fit
    def fit(self, age, y, sex=None, site=None) -> "NormativeModel":
        age = np.asarray(age, float)
        y = np.asarray(y, float)
        lo, hi = self.age_range or (age.min(), age.max())
        self.knots_ = np.quantile(age, np.linspace(0, 1, self.n_knots))
        self.knots_[0], self.knots_[-1] = lo, hi + 1e-6
        self.has_sex_ = sex is not None
        self.sites_ = None if site is None else np.unique(np.asarray(site))
        X = self._design_mu(age, sex, site)
        Xs = self._design_sigma(age)
        w = np.ones(len(y))
        for _ in range(self.n_rounds):
            sw = np.sqrt(w)
            self.beta_mu_, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            r = y - X @ self.beta_mu_
            # E[log(chi2_1)] = -1.27036; regress log r^2 on the age basis and correct the bias
            target = np.log(r ** 2 + 1e-12) + 1.27036
            self.beta_sigma_, *_ = np.linalg.lstsq(Xs, target, rcond=None)
            w = 1.0 / np.exp(Xs @ self.beta_sigma_)
        return self

    # ----------------------------------------------------------------- predict
    def predict(self, age, sex=None, site=None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)`` at the requested covariates."""
        age = np.asarray(age, float)
        if self.has_sex_ and sex is None:
            sex = np.zeros(len(age))
        if self.sites_ is not None and site is None:
            site = np.full(len(age), self.sites_[0])
        mu = self._design_mu(age, sex, site) @ self.beta_mu_
        sigma = np.sqrt(np.exp(self._design_sigma(age) @ self.beta_sigma_))
        return mu, sigma

    def zscore(self, age, y, sex=None, site=None) -> np.ndarray:
        mu, sigma = self.predict(age, sex, site)
        return (np.asarray(y, float) - mu) / sigma

    def centiles(self, age, probs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95), sex=None, site=None) -> pd.DataFrame:
        mu, sigma = self.predict(age, sex, site)
        out = pd.DataFrame({"age": np.asarray(age, float)})
        for p in probs:
            out[f"c{int(round(p * 100)):02d}"] = mu + stats.norm.ppf(p) * sigma
        return out


def age_of_peak(model: NormativeModel, grid: Optional[np.ndarray] = None, sex=None, minimum: bool = False) -> float:
    """Age at which the fitted mean curve is maximal (or minimal)."""
    if grid is None:
        grid = np.linspace(model.knots_[0], model.knots_[-1] - 1e-3, 500)
    sexv = None if sex is None else np.full(len(grid), sex)
    mu, _ = model.predict(grid, sexv)
    return float(grid[np.argmin(mu) if minimum else np.argmax(mu)])


def bootstrap_age_of_peak(
    age, y, sex=None, site=None, n_boot: int = 200, minimum: bool = False, seed: int = 0, **model_kwargs
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Nonparametric bootstrap of the age of peak; returns ``(point, (lo, hi), draws)``."""
    rng = np.random.default_rng(seed)
    age, y = np.asarray(age, float), np.asarray(y, float)
    sex = None if sex is None else np.asarray(sex)
    site = None if site is None else np.asarray(site)
    base = NormativeModel(**model_kwargs).fit(age, y, sex, site)
    point = age_of_peak(base, minimum=minimum)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = NormativeModel(age_range=(age.min(), age.max()), **model_kwargs).fit(
            age[idx], y[idx], None if sex is None else sex[idx], None if site is None else site[idx]
        )
        draws[b] = age_of_peak(m, minimum=minimum)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, (float(lo), float(hi)), draws


def simulate_lifespan_dataset(
    n: int = 2000,
    peak_age: float = 30.0,
    amplitude: float = 0.08,
    baseline: float = 0.45,
    sex_effect: float = 0.01,
    site_effects: Sequence[float] = (0.0, 0.02, -0.015),
    noise_young: float = 0.02,
    noise_old: float = 0.035,
    age_range: Tuple[float, float] = (8.0, 90.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic FA-like metric with an inverted-U over age and age-dependent noise."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(*age_range, n)
    sex = rng.integers(0, 2, n)
    site = rng.integers(0, len(site_effects), n)
    mu = baseline + amplitude * (1 - ((age - peak_age) / 40.0) ** 2)
    sd = noise_young + (noise_old - noise_young) * (age - age_range[0]) / (age_range[1] - age_range[0])
    y = mu + sex_effect * sex + np.asarray(site_effects)[site] + rng.normal(0, sd)
    return pd.DataFrame({"age": age, "sex": sex, "site": site, "y": y, "mu_true": mu, "sd_true": sd})
