"""ComBat harmonisation with separate fit / transform steps.

Implements the parametric empirical-Bayes ComBat of Johnson, Li & Rabinovic
(2007, *Biostatistics*) as adapted to neuroimaging features by Fortin et al.
(2018, *NeuroImage*; ``neuroCombat``).  Two things matter for a
*transportability* audit and are not offered by the reference implementation:

1. ``fit`` on a reference cohort (e.g. the training scanners) and
   ``transform`` new samples from a *known* batch without refitting, so that
   test-set information never leaks into the harmonisation parameters.
2. Optional biological covariates (age, sex) are preserved, and the same
   design is reused at transform time.

If ``neuroCombat`` is installed, :func:`neurocombat_harmonize` wraps it for a
one-shot comparison; the results should agree to numerical precision on the
same data when ``eb=True`` and ``parametric=True``.

Notation follows the original paper: data matrix ``X`` is *features x samples*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _design(batch_codes: np.ndarray, n_batches: int, covars: Optional[np.ndarray]) -> np.ndarray:
    onehot = np.zeros((batch_codes.size, n_batches))
    onehot[np.arange(batch_codes.size), batch_codes] = 1.0
    if covars is None or covars.size == 0:
        return onehot
    return np.hstack([onehot, covars])


def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1)


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=1000):
    """Iterative EB solution for one batch (features x samples in ``sdat``)."""
    n = (~np.isnan(sdat)).sum(axis=1).astype(float)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum((sdat - g_new[:, None]) ** 2, axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
            np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12)),
        )
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_old, d_old


@dataclass
class ComBat:
    """Parametric empirical-Bayes ComBat with fit/transform semantics.

    Parameters
    ----------
    eb
        Use empirical-Bayes shrinkage of batch parameters (True) or plain
        per-batch location/scale estimates (False).
    mean_only
        Adjust only the location (variance left untouched).
    conv, max_iter
        Convergence tolerance / cap for the EB iteration.

    Examples
    --------
    >>> cb = ComBat().fit(X_train, batch_train, covars_train)
    >>> X_train_h = cb.transform(X_train, batch_train, covars_train)
    >>> X_test_h = cb.transform(X_test, batch_test, covars_test)   # no refit
    """

    eb: bool = True
    mean_only: bool = False
    conv: float = 1e-4
    max_iter: int = 1000

    def __post_init__(self) -> None:
        self.batches_: Optional[list] = None
        self.grand_mean_: Optional[np.ndarray] = None
        self.beta_covar_: Optional[np.ndarray] = None
        self.var_pooled_: Optional[np.ndarray] = None
        self.gamma_star_: Optional[np.ndarray] = None
        self.delta_star_: Optional[np.ndarray] = None
        self.n_covars_: int = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_matrix(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (samples x features)")
        return X

    def _codes(self, batch: Sequence) -> np.ndarray:
        assert self.batches_ is not None
        lookup = {b: i for i, b in enumerate(self.batches_)}
        try:
            return np.array([lookup[b] for b in batch])
        except KeyError as e:  # pragma: no cover - defensive
            raise ValueError(f"Unseen batch label {e}; refit or map it to a known scanner") from e

    # ------------------------------------------------------------------- fit
    def fit(self, X, batch: Sequence, covars: Optional[np.ndarray] = None) -> "ComBat":
        """Estimate harmonisation parameters.

        Parameters
        ----------
        X : (n_samples, n_features) array-like
        batch : sequence of length n_samples with scanner / site labels
        covars : optional (n_samples, n_covars) numeric design (age, sex ...)
            whose effects are preserved.
        """
        Xs = self._as_matrix(X)
        batch = np.asarray(batch)
        self.batches_ = list(pd.unique(batch))
        codes = self._codes(batch)
        n_b = len(self.batches_)
        if covars is not None:
            covars = np.asarray(covars, dtype=float)
            if covars.ndim == 1:
                covars = covars[:, None]
        self.n_covars_ = 0 if covars is None else covars.shape[1]

        D = _design(codes, n_b, covars)                     # n x (B+q)
        Xf = Xs.T                                            # p x n
        beta_hat = np.linalg.lstsq(D, Xf.T, rcond=None)[0]  # (B+q) x p
        n_per_batch = np.bincount(codes, minlength=n_b).astype(float)
        self.grand_mean_ = (n_per_batch / codes.size) @ beta_hat[:n_b]  # p
        self.beta_covar_ = beta_hat[n_b:]                    # q x p
        fitted = (D @ beta_hat).T                            # p x n
        self.var_pooled_ = ((Xf - fitted) ** 2).mean(axis=1)  # p
        self.var_pooled_ = np.where(self.var_pooled_ <= 0, 1e-12, self.var_pooled_)

        stand_mean = self.grand_mean_[:, None] + (covars @ self.beta_covar_).T if covars is not None else self.grand_mean_[:, None]
        s_data = (Xf - stand_mean) / np.sqrt(self.var_pooled_)[:, None]

        gamma_hat = np.vstack([s_data[:, codes == b].mean(axis=1) for b in range(n_b)])  # B x p
        delta_hat = np.vstack([
            s_data[:, codes == b].var(axis=1, ddof=1) if (codes == b).sum() > 1 else np.ones(Xf.shape[0])
            for b in range(n_b)
        ])

        if self.eb and n_b > 1:
            gamma_bar = gamma_hat.mean(axis=1)
            t2 = gamma_hat.var(axis=1, ddof=1)
            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)
            for b in range(n_b):
                a, bb = _aprior(delta_hat[b]), _bprior(delta_hat[b])
                g, d = _it_sol(s_data[:, codes == b], gamma_hat[b], delta_hat[b],
                               gamma_bar[b], t2[b], a, bb, self.conv, self.max_iter)
                gamma_star[b], delta_star[b] = g, d
        else:
            gamma_star, delta_star = gamma_hat, delta_hat
        if self.mean_only:
            delta_star = np.ones_like(delta_star)
        self.gamma_star_, self.delta_star_ = gamma_star, delta_star
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X, batch: Sequence, covars: Optional[np.ndarray] = None) -> np.ndarray:
        """Harmonise ``X`` using parameters from :meth:`fit` (no refitting)."""
        if self.gamma_star_ is None:
            raise RuntimeError("ComBat must be fitted first")
        Xs = self._as_matrix(X)
        codes = self._codes(np.asarray(batch))
        if self.n_covars_:
            if covars is None:
                raise ValueError("covars were used at fit time and must be given to transform")
            covars = np.asarray(covars, dtype=float)
            if covars.ndim == 1:
                covars = covars[:, None]
            stand_mean = self.grand_mean_[:, None] + (covars @ self.beta_covar_).T
        else:
            stand_mean = self.grand_mean_[:, None]
        sd = np.sqrt(self.var_pooled_)[:, None]
        s_data = (Xs.T - stand_mean) / sd
        gamma = self.gamma_star_[codes].T           # p x n
        delta = np.sqrt(self.delta_star_[codes]).T  # p x n
        adjusted = (s_data - gamma) / delta * sd + stand_mean
        return adjusted.T

    def fit_transform(self, X, batch: Sequence, covars: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, batch, covars).transform(X, batch, covars)

    def batch_effect_table(self, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Tidy table of estimated location (gamma*) / scale (delta*) per batch."""
        if self.gamma_star_ is None:
            raise RuntimeError("ComBat must be fitted first")
        p = self.gamma_star_.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(p)]
        rows = []
        for bi, b in enumerate(self.batches_):
            for fi, f in enumerate(names):
                rows.append({"batch": b, "feature": f,
                             "gamma_star": self.gamma_star_[bi, fi],
                             "delta_star": self.delta_star_[bi, fi]})
        return pd.DataFrame(rows)


def neurocombat_harmonize(X: np.ndarray, batch: Sequence, covars: Optional[pd.DataFrame] = None) -> np.ndarray:
    """One-shot harmonisation with the reference ``neuroCombat`` package.

    Parameters mirror :class:`ComBat` but everything is fitted on ``X`` (no
    transport).  Raises ``ImportError`` with an install hint when the package
    is not available.
    """
    try:
        from neuroCombat import neuroCombat  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install neuroCombat") from e
    covars_df = pd.DataFrame({"batch": np.asarray(batch)})
    continuous = []
    if covars is not None:
        for c in covars.columns:
            covars_df[c] = np.asarray(covars[c])
            continuous.append(c)
    out = neuroCombat(dat=np.asarray(X).T, covars=covars_df, batch_col="batch",
                      continuous_cols=continuous or None)
    return np.asarray(out["data"]).T


def batch_variance_explained(X: np.ndarray, batch: Sequence) -> pd.Series:
    """Fraction of variance explained by batch (eta^2) per feature.

    A quick diagnostic to report before/after harmonisation.
    """
    Xs = np.asarray(X, dtype=float)
    batch = np.asarray(batch)
    grand = Xs.mean(axis=0)
    ss_tot = ((Xs - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(Xs.shape[1])
    for b in pd.unique(batch):
        m = batch == b
        ss_between += m.sum() * (Xs[m].mean(axis=0) - grand) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        eta2 = np.where(ss_tot > 0, ss_between / ss_tot, np.nan)
    return pd.Series(eta2, name="eta2_batch")
