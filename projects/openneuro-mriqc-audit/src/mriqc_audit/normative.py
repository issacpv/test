"""Normative reference charts for image-quality metrics.

Two complementary models, both conditioning on age (spline) and categorical
scanner descriptors (manufacturer, field strength, MRIQC major version):

``QuantileNormativeModel``
    Quantile regression (Koenker & Bassett, 1978) via ``statsmodels.QuantReg``
    on a cubic B-spline basis of age (``sklearn.preprocessing.SplineTransformer``)
    plus one-hot scanner terms.  Gives centile curves directly and makes no
    distributional assumption; centiles for a new scan are obtained by
    interpolating its value among the predicted quantiles.
``LocationScaleModel``
    GAMLSS-style (Rigby & Stasinopoulos, 2005) approximation: a spline model
    for the mean and a second spline model for log residual scale, optionally
    on a log-transformed IQM (which handles the right skew of ``cjv``, ``snr``
    and friends).  Gives z-scores comparable to the normative-modelling
    convention in neuroimaging (Marquand et al., 2016; Rutherford et al., 2022).

Both accept the tidy frames produced by ``mriqc_client`` joined with
participant age from ``openneuro_client``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import SplineTransformer

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _design(df: pd.DataFrame, age_col: str, cat_cols: Sequence[str], spline: SplineTransformer,
            cat_levels: Optional[Dict[str, List]] = None, fit: bool = False) -> tuple:
    """Spline(age) + one-hot(categoricals) + intercept design matrix."""
    age = df[[age_col]].to_numpy(dtype=float)
    B = spline.fit_transform(age) if fit else spline.transform(age)
    parts = [np.ones((len(df), 1)), B]
    names = ["intercept"] + [f"age_bs{i}" for i in range(B.shape[1])]
    levels = cat_levels or {}
    for c in cat_cols:
        col = df[c].astype("string").fillna("missing")
        if fit:
            levels[c] = sorted(col.unique().tolist())
        for lev in levels[c][1:]:  # first level = reference
            parts.append((col == lev).to_numpy(dtype=float)[:, None])
            names.append(f"{c}={lev}")
    return np.hstack(parts), names, levels


@dataclass
class QuantileNormativeModel:
    """Quantile-regression centile curves for one IQM."""

    iqm: str
    age_col: str = "age"
    cat_cols: Sequence[str] = ()
    quantiles: Sequence[float] = DEFAULT_QUANTILES
    n_knots: int = 5
    log_transform: bool = False
    results_: Dict[float, np.ndarray] = field(default_factory=dict, repr=False)
    names_: List[str] = field(default_factory=list, repr=False)
    levels_: Dict[str, List] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False, extrapolation="linear")

    def _y(self, df: pd.DataFrame) -> np.ndarray:
        y = df[self.iqm].to_numpy(dtype=float)
        return np.log(y) if self.log_transform else y

    def fit(self, df: pd.DataFrame) -> "QuantileNormativeModel":
        df = df.dropna(subset=[self.iqm, self.age_col])
        X, self.names_, self.levels_ = _design(df, self.age_col, self.cat_cols, self.spline_, fit=True)
        y = self._y(df)
        for q in self.quantiles:
            res = sm.QuantReg(y, X).fit(q=q, max_iter=5000)
            self.results_[q] = np.asarray(res.params)
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicted quantile values (original scale) for each row of ``df``."""
        X, _, _ = _design(df, self.age_col, self.cat_cols, self.spline_, self.levels_, fit=False)
        out = {}
        for q, beta in self.results_.items():
            pred = X @ beta
            out[q] = np.exp(pred) if self.log_transform else pred
        return pd.DataFrame(out, index=df.index)

    def centile(self, df: pd.DataFrame) -> np.ndarray:
        """Approximate centile of each observed IQM among the fitted quantiles.

        Linear interpolation between adjacent quantile curves; values below the
        lowest / above the highest curve are clipped to those quantiles.
        """
        Q = self.predict_quantiles(df)
        qs = np.array(sorted(self.results_))
        vals = df[self.iqm].to_numpy(dtype=float)
        cent = np.full(vals.shape, np.nan)
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            curve = Q.iloc[i][qs].to_numpy(dtype=float)
            order = np.argsort(curve)
            cent[i] = float(np.interp(v, curve[order], qs[order]))
        return cent

    def curves(self, age_grid: Sequence[float], **cats) -> pd.DataFrame:
        """Centile curves along ``age_grid`` for one scanner configuration."""
        grid = pd.DataFrame({self.age_col: np.asarray(age_grid, dtype=float)})
        for c in self.cat_cols:
            grid[c] = cats.get(c, self.levels_[c][0])
        out = self.predict_quantiles(grid)
        out.insert(0, self.age_col, grid[self.age_col].values)
        return out


@dataclass
class LocationScaleModel:
    """Spline mean + spline log-scale model giving normative z-scores."""

    iqm: str
    age_col: str = "age"
    cat_cols: Sequence[str] = ()
    n_knots: int = 5
    log_transform: bool = True
    beta_mu_: Optional[np.ndarray] = field(default=None, repr=False)
    beta_sigma_: Optional[np.ndarray] = field(default=None, repr=False)
    levels_: Dict[str, List] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False, extrapolation="linear")

    def _y(self, df: pd.DataFrame) -> np.ndarray:
        y = df[self.iqm].to_numpy(dtype=float)
        return np.log(y) if self.log_transform else y

    def fit(self, df: pd.DataFrame) -> "LocationScaleModel":
        df = df.dropna(subset=[self.iqm, self.age_col])
        X, _, self.levels_ = _design(df, self.age_col, self.cat_cols, self.spline_, fit=True)
        y = self._y(df)
        self.beta_mu_ = np.asarray(sm.OLS(y, X).fit().params)
        resid = y - X @ self.beta_mu_
        # log|resid| regression estimates log sigma up to a constant (Harvey, 1976 style)
        z = np.log(np.abs(resid) + 1e-9)
        beta_s = np.asarray(sm.OLS(z, X).fit().params)
        # calibrate so that mean squared standardized residual is 1
        sigma = np.exp(X @ beta_s)
        scale = np.sqrt(np.mean((resid / sigma) ** 2))
        beta_s[0] += np.log(scale)
        self.beta_sigma_ = beta_s
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _, _ = _design(df, self.age_col, self.cat_cols, self.spline_, self.levels_, fit=False)
        mu = X @ self.beta_mu_
        sigma = np.exp(X @ self.beta_sigma_)
        return pd.DataFrame({"mu": mu, "sigma": sigma}, index=df.index)

    def zscore(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict(df)
        return (self._y(df) - p["mu"].to_numpy()) / p["sigma"].to_numpy()

    def centile(self, df: pd.DataFrame) -> np.ndarray:
        return stats.norm.cdf(self.zscore(df))


def coverage_check(centiles: np.ndarray, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Empirical vs nominal coverage of fitted centiles (calibration table)."""
    c = np.asarray(centiles, dtype=float)
    c = c[~np.isnan(c)]
    rows = [{"nominal": q, "empirical": float(np.mean(c <= q)), "n": int(c.size)} for q in quantiles]
    return pd.DataFrame(rows)


def simulate_iqm_cohort(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic age/scanner-dependent CJV-like IQM for tests and demos.

    log(cjv) rises in children (< 12 y) and the elderly (> 65 y), is higher at
    1.5T than 3T and has larger spread in children (motion).
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(4, 90, n)
    field_strength = rng.choice([1.5, 3.0], size=n, p=[0.3, 0.7])
    manufacturer = rng.choice(["Siemens", "GE", "Philips"], size=n, p=[0.6, 0.25, 0.15])
    mu = -1.0 + 0.04 * np.maximum(12 - age, 0) + 0.006 * np.maximum(age - 65, 0) + 0.15 * (field_strength == 1.5)
    mu += 0.05 * (manufacturer == "GE")
    sigma = 0.15 + 0.1 * (age < 12)
    cjv = np.exp(mu + rng.normal(0, 1, n) * sigma)
    group = np.where(rng.random(n) < 0.4, "patient", "control")
    # patients move more: shift IQM upward
    cjv[group == "patient"] *= np.exp(0.12)
    return pd.DataFrame({"age": age, "MagneticFieldStrength": field_strength, "Manufacturer": manufacturer,
                         "cjv": cjv, "group": group})
