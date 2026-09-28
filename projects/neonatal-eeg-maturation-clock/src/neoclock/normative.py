"""Normative centiles of maturational features against PMA.

Each feature is modelled as Gaussian with a polynomial mean and a polynomial
log-SD in PMA (a light-weight stand-in for GAMLSS / warped normative models).
Z-scores, centiles and a multivariate deviation score (mean |z|) are provided so
that a new recording can be placed on a chart and flagged when out of range.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


class FeatureCentiles:
    def __init__(self, degree_mean: int = 2, degree_sd: int = 1, min_sd: float = 1e-6):
        self.degree_mean, self.degree_sd, self.min_sd = degree_mean, degree_sd, min_sd
        self.names: Sequence[str] = ()
        self.coef_mean_: Optional[np.ndarray] = None
        self.coef_sd_: Optional[np.ndarray] = None
        self.age_center_ = 40.0

    def _design(self, age: np.ndarray, degree: int) -> np.ndarray:
        a = (np.asarray(age, float) - self.age_center_) / 5.0
        return np.column_stack([a ** k for k in range(degree + 1)])

    def fit(self, F: np.ndarray, age: np.ndarray, names: Optional[Sequence[str]] = None) -> "FeatureCentiles":
        F, age = np.asarray(F, float), np.asarray(age, float)
        self.names = list(names) if names is not None else [f"f{i}" for i in range(F.shape[1])]
        self.age_center_ = float(np.nanmedian(age))
        Am, As = self._design(age, self.degree_mean), self._design(age, self.degree_sd)
        self.coef_mean_ = np.zeros((Am.shape[1], F.shape[1]))
        self.coef_sd_ = np.zeros((As.shape[1], F.shape[1]))
        for j in range(F.shape[1]):
            ok = np.isfinite(F[:, j]) & np.isfinite(age)
            if ok.sum() <= Am.shape[1] + 1:
                continue
            bm, *_ = np.linalg.lstsq(Am[ok], F[ok, j], rcond=None)
            self.coef_mean_[:, j] = bm
            resid = F[ok, j] - Am[ok] @ bm
            target = np.log(np.abs(resid) + 1e-12) + 0.5 * np.log(np.pi / 2)
            bs, *_ = np.linalg.lstsq(As[ok], target, rcond=None)
            self.coef_sd_[:, j] = bs
        return self

    def expected(self, age: np.ndarray):
        mu = self._design(age, self.degree_mean) @ self.coef_mean_
        sd = np.maximum(np.exp(self._design(age, self.degree_sd) @ self.coef_sd_), self.min_sd)
        return mu, sd

    def zscore(self, F: np.ndarray, age: np.ndarray) -> np.ndarray:
        mu, sd = self.expected(age)
        return (np.asarray(F, float) - mu) / sd

    def centile(self, F: np.ndarray, age: np.ndarray) -> np.ndarray:
        return stats.norm.cdf(self.zscore(F, age)) * 100.0

    def centile_table(self, ages: Sequence[float], qs: Sequence[float] = (3, 10, 50, 90, 97)) -> pd.DataFrame:
        ages = np.asarray(ages, float)
        mu, sd = self.expected(ages)
        rows = []
        for j, name in enumerate(self.names):
            for i, a in enumerate(ages):
                row = {"feature": name, "pma_weeks": float(a)}
                for q in qs:
                    row[f"c{q}"] = float(mu[i, j] + stats.norm.ppf(q / 100.0) * sd[i, j])
                rows.append(row)
        return pd.DataFrame(rows)


def multivariate_deviation(Z: np.ndarray) -> np.ndarray:
    """Mean |z| across features per recording (NaN-aware); > ~1.5 suggests an out-of-range recording."""
    return np.nanmean(np.abs(np.asarray(Z, float)), axis=1)


def out_of_distribution_flag(Z: np.ndarray, thr: float = 1.5) -> np.ndarray:
    return multivariate_deviation(Z) > thr


def calibration_check(Z: np.ndarray, age: np.ndarray, n_bins: int = 4) -> pd.DataFrame:
    """Mean and SD of z-scores per age quantile bin (should be ~0 and ~1 on held-out data)."""
    Z, age = np.asarray(Z, float), np.asarray(age, float)
    edges = np.quantile(age, np.linspace(0, 1, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (age >= lo) & (age <= hi)
        rows.append(dict(age_lo=float(lo), age_hi=float(hi), n=int(m.sum()), z_mean=float(np.nanmean(Z[m])),
                         z_sd=float(np.nanstd(Z[m]))))
    return pd.DataFrame(rows)
