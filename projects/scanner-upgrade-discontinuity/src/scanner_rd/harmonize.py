"""Harmonization arms used in the benchmark.

* :class:`ComBat` — cross-sectional parametric empirical-Bayes ComBat (Johnson et al., 2007;
  Fortin et al., 2018) with a fit/transform API so that parameters can be estimated on a
  reference set (e.g. baseline sessions of cognitively normal subjects) and applied elsewhere.
* :class:`LongitudinalComBatLite` — a simplified longitudinal ComBat in the spirit of Beer et
  al. (2020): scanner additive/multiplicative effects are estimated after removing subject
  effects by a within-subject transformation (no empirical-Bayes shrinkage). Use the R package
  ``longCombat`` as the reference implementation for the paper.
* :func:`rd_anchored_correction` — subtract the RDiT-estimated jump (and slope change) from
  post-upgrade sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _design(covars: Optional[np.ndarray], n: int) -> np.ndarray:
    if covars is None:
        return np.ones((n, 1))
    covars = np.asarray(covars, float)
    if covars.ndim == 1:
        covars = covars[:, None]
    return np.column_stack([np.ones(n), covars])


@dataclass
class ComBat:
    """Cross-sectional parametric ComBat.

    ``fit(X, batch, covars)`` estimates the biological model and per-batch location/scale;
    ``transform`` applies them. ``X`` is ``(n_samples, n_features)``.
    """

    eb: bool = True
    n_iter: int = 30

    def fit(self, X: np.ndarray, batch: np.ndarray, covars: Optional[np.ndarray] = None) -> "ComBat":
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        n, p = X.shape
        self.batches_ = np.unique(batch)
        B = np.column_stack([(batch == b).astype(float) for b in self.batches_])
        Z = _design(covars, n)[:, 1:]  # covariates without intercept (batch dummies span it)
        design = np.column_stack([B, Z]) if Z.shape[1] else B
        beta, *_ = np.linalg.lstsq(design, X, rcond=None)
        nb = len(self.batches_)
        gamma_hat = beta[:nb]  # batch means
        self.beta_cov_ = beta[nb:]
        # grand mean weighted by batch size
        counts = B.sum(axis=0)
        self.grand_mean_ = (counts[:, None] * gamma_hat).sum(axis=0) / n
        fitted = design @ beta
        self.pooled_var_ = ((X - fitted) ** 2).mean(axis=0)
        s = np.sqrt(self.pooled_var_)
        stand = (X - self.grand_mean_ - (Z @ self.beta_cov_ if Z.shape[1] else 0)) / s
        gam = np.zeros((nb, p))
        delta = np.ones((nb, p))
        for i, b in enumerate(self.batches_):
            idx = batch == b
            gam[i] = stand[idx].mean(axis=0)
            delta[i] = stand[idx].var(axis=0, ddof=1) if idx.sum() > 1 else 1.0
        if self.eb:
            gam, delta = self._eb_shrink(stand, batch, gam, delta)
        self.gamma_star_, self.delta_star_ = gam, delta
        return self

    def _eb_shrink(self, stand, batch, gam, delta):
        nb, p = gam.shape
        gam_s, del_s = gam.copy(), delta.copy()
        for i, b in enumerate(self.batches_):
            idx = batch == b
            n_b = idx.sum()
            g_bar, t2 = gam[i].mean(), gam[i].var(ddof=1) if p > 1 else 1.0
            d_mean, d_var = delta[i].mean(), delta[i].var(ddof=1) if p > 1 else 1.0
            a_prior = (2 * d_var + d_mean ** 2) / max(d_var, 1e-12)
            b_prior = (d_mean * d_var + d_mean ** 3) / max(d_var, 1e-12)
            g_new, d_new = gam[i].copy(), delta[i].copy()
            for _ in range(self.n_iter):
                g_post = (n_b * t2 * gam[i] + d_new * g_bar) / (n_b * t2 + d_new)
                ss = ((stand[idx] - g_post) ** 2).sum(axis=0)
                d_post = (0.5 * ss + b_prior) / (n_b / 2 + a_prior - 1)
                if np.allclose(g_post, g_new) and np.allclose(d_post, d_new):
                    break
                g_new, d_new = g_post, d_post
            gam_s[i], del_s[i] = g_new, np.clip(d_new, 1e-8, None)
        return gam_s, del_s

    def transform(self, X: np.ndarray, batch: np.ndarray, covars: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        n = X.shape[0]
        Z = _design(covars, n)[:, 1:]
        cov_part = Z @ self.beta_cov_ if Z.shape[1] else 0.0
        s = np.sqrt(self.pooled_var_)
        stand = (X - self.grand_mean_ - cov_part) / s
        out = np.empty_like(X)
        for i, b in enumerate(self.batches_):
            idx = batch == b
            if not idx.any():
                continue
            out[idx] = (stand[idx] - self.gamma_star_[i]) / np.sqrt(self.delta_star_[i])
        unseen = ~np.isin(batch, self.batches_)
        if unseen.any():
            raise ValueError(f"unseen batches: {np.unique(batch[unseen])}")
        return out * s + self.grand_mean_ + cov_part

    def fit_transform(self, X, batch, covars=None) -> np.ndarray:
        return self.fit(X, batch, covars).transform(X, batch, covars)


@dataclass
class LongitudinalComBatLite:
    """Longitudinal ComBat (lite): subject effects removed before estimating scanner effects.

    For each feature: ``y = subject_i + Z beta + gamma_scanner + delta_scanner * e``. The subject
    effect is removed by within-subject demeaning, ``gamma`` is estimated from the demeaned data
    (identified by subjects scanned on several scanners), ``delta`` from residual SDs by scanner.
    """

    def fit(self, X, batch, subject, covars=None) -> "LongitudinalComBatLite":
        X = np.asarray(X, float)
        batch, subject = np.asarray(batch), np.asarray(subject)
        n, p = X.shape
        self.batches_ = np.unique(batch)
        B = np.column_stack([(batch == b).astype(float) for b in self.batches_])
        Z = _design(covars, n)[:, 1:]
        M = np.column_stack([B, Z]) if Z.shape[1] else B
        # within-subject transformation
        Xd, Md = X.copy(), M.copy()
        for s in np.unique(subject):
            idx = subject == s
            Xd[idx] -= X[idx].mean(axis=0)
            Md[idx] -= M[idx].mean(axis=0)
        coef, *_ = np.linalg.lstsq(Md, Xd, rcond=None)
        nb = len(self.batches_)
        gam = coef[:nb]
        self.beta_cov_ = coef[nb:]
        # centre scanner effects so that the average scanner is the reference
        counts = B.sum(axis=0)
        self.gamma_ = gam - (counts[:, None] * gam).sum(axis=0) / n
        resid = Xd - Md @ coef
        self.delta_ = np.ones((nb, p))
        pooled = resid.std(axis=0, ddof=1)
        for i, b in enumerate(self.batches_):
            idx = batch == b
            if idx.sum() > 2:
                self.delta_[i] = resid[idx].std(axis=0, ddof=1) / np.clip(pooled, 1e-12, None)
        self.delta_ = np.clip(self.delta_, 1e-3, None)
        self.grand_mean_ = X.mean(axis=0)
        return self

    def _gamma_of(self, batch: np.ndarray) -> np.ndarray:
        G = np.zeros((len(batch), self.gamma_.shape[1]))
        D = np.ones((len(batch), self.gamma_.shape[1]))
        for i, b in enumerate(self.batches_):
            idx = batch == b
            G[idx] = self.gamma_[i]
            D[idx] = self.delta_[i]
        unseen = ~np.isin(batch, self.batches_)
        if unseen.any():
            raise ValueError(f"unseen batches: {np.unique(batch[unseen])}")
        return G, D

    def transform(self, X, batch, subject=None, covars=None) -> np.ndarray:
        """Remove scanner location/scale effects.

        The scale is applied to residuals around the fitted biological part (covariates plus the
        subject's own mean after location correction), as in ComBat, never to raw values: for a
        feature with a large mean (e.g. 3,600 mm^3) dividing raw values by a scale of 1.02 would
        itself create an offset of ~70 mm^3.
        """
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        n = X.shape[0]
        G, D = self._gamma_of(batch)
        Z = _design(covars, n)[:, 1:]
        cov_part = Z @ self.beta_cov_ if Z.shape[1] else np.zeros_like(X)
        loc = X - G - cov_part
        if subject is not None:
            subject = np.asarray(subject)
            fitted = np.empty_like(X)
            for s in np.unique(subject):
                idx = subject == s
                fitted[idx] = loc[idx].mean(axis=0)
        else:
            fitted = np.broadcast_to(self.grand_mean_, X.shape)
        resid = loc - fitted
        return resid / D + fitted + cov_part

    def fit_transform(self, X, batch, subject, covars=None) -> np.ndarray:
        return self.fit(X, batch, subject, covars).transform(X, batch, subject, covars)


def rd_anchored_correction(
    values: np.ndarray,
    t_rel: np.ndarray,
    jump: float,
    slope_change: float = 0.0,
) -> np.ndarray:
    """Subtract an RDiT-estimated level shift (and slope change) from post-upgrade sessions."""
    values = np.asarray(values, float)
    t_rel = np.asarray(t_rel, float)
    post = (t_rel >= 0).astype(float)
    return values - post * (jump + slope_change * t_rel)


def batch_effect_size(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Per-feature standardised mean difference between the first two batches (audit metric)."""
    X = np.asarray(X, float)
    b = np.unique(batch)
    if len(b) < 2:
        return np.zeros(X.shape[1])
    a, c = X[batch == b[0]], X[batch == b[1]]
    sd = np.sqrt(0.5 * (a.var(axis=0, ddof=1) + c.var(axis=0, ddof=1)))
    return (a.mean(axis=0) - c.mean(axis=0)) / np.clip(sd, 1e-12, None)
