"""Representational-drift metrics on trial x unit response matrices.

All functions take ``X`` (n_trials, n_units), ``cond`` (n_trials,) integer
condition labels and ``block`` (n_trials,) integer time-block labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .responses import condition_means


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def population_vector_correlation(X: np.ndarray, cond: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Block x block matrix of PV correlations, averaged over conditions.

    For each condition the population vector (mean response over trials in a block)
    is correlated across units between blocks; the matrix is the mean over conditions.
    """
    M, _ = condition_means(X, cond, block)
    nb = M.shape[0]
    out = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            vals = [_nan_corr(M[i, c], M[j, c]) for c in range(M.shape[1])]
            out[i, j] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
    return out


def drift_rate_from_matrix(C: np.ndarray) -> Dict[str, float]:
    """Summaries of a block x block similarity matrix: mean similarity vs lag and its slope."""
    nb = C.shape[0]
    lags = np.arange(1, nb)
    by_lag = np.array([np.nanmean([C[i, i + k] for i in range(nb - k)]) for k in lags])
    slope = np.polyfit(lags, by_lag, 1)[0] if nb > 2 else (by_lag[0] - np.nanmean(np.diag(C)))
    return {"lag1": float(by_lag[0]) if len(by_lag) else np.nan, "slope_per_block": float(slope),
            "mean_offdiag": float(np.nanmean(C[~np.eye(nb, dtype=bool)])) if nb > 1 else np.nan}


def rdm_stability(X: np.ndarray, cond: np.ndarray, block: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Block x block Spearman similarity between representational dissimilarity matrices."""
    M, _ = condition_means(X, cond, block)
    nb = M.shape[0]
    rdms = []
    for b in range(nb):
        m = M[b]
        ok = np.isfinite(m).all(axis=1)
        rdms.append(pdist(m[ok], metric=metric) if ok.sum() >= 3 else np.full(1, np.nan))
    out = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            if rdms[i].shape == rdms[j].shape and np.isfinite(rdms[i]).all() and np.isfinite(rdms[j]).all():
                out[i, j] = spearmanr(rdms[i], rdms[j]).correlation
    return out


def tuning_correlation_per_unit(X: np.ndarray, cond: np.ndarray, block: np.ndarray,
                                b0: Optional[int] = None, b1: Optional[int] = None) -> np.ndarray:
    """Per-unit Pearson correlation of tuning curves between two blocks (default first vs last)."""
    M, _ = condition_means(X, cond, block)
    b0 = 0 if b0 is None else b0
    b1 = M.shape[0] - 1 if b1 is None else b1
    return np.array([_nan_corr(M[b0, :, u], M[b1, :, u]) for u in range(X.shape[1])])


def _default_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def decoder_cross_time(X: np.ndarray, cond: np.ndarray, block: np.ndarray, clf_factory: Callable = _default_clf,
                       n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Block x block decoding accuracy matrix.

    Diagonal entries are within-block stratified k-fold cross-validated accuracies;
    off-diagonal entries train on all trials of block *i* and test on block *j*.
    Accuracy is balanced accuracy (mean per-class recall).
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    blocks = np.unique(block)
    nb = len(blocks)
    acc = np.full((nb, nb), np.nan)
    for i, bi in enumerate(blocks):
        tr = block == bi
        Xi, yi = X[tr], cond[tr]
        counts = np.bincount(yi)
        k = int(min(n_splits, counts[counts > 0].min()))
        if k >= 2:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            preds = np.empty_like(yi)
            for a, b in skf.split(Xi, yi):
                preds[b] = clf_factory().fit(Xi[a], yi[a]).predict(Xi[b])
            acc[i, i] = balanced_accuracy_score(yi, preds)
        model = clf_factory().fit(Xi, yi)
        for j, bj in enumerate(blocks):
            if i == j:
                continue
            te = block == bj
            acc[i, j] = balanced_accuracy_score(cond[te], model.predict(X[te]))
    return acc


def decoder_drift_index(acc: np.ndarray) -> float:
    """``1 - mean(off-diagonal accuracy) / mean(diagonal accuracy)``; 0 = no loss across time."""
    nb = acc.shape[0]
    diag = np.nanmean(np.diag(acc))
    off = np.nanmean(acc[~np.eye(nb, dtype=bool)])
    return float(1 - off / diag) if diag > 0 else np.nan


@dataclass
class DriftSummary:
    pv_lag1: float
    pv_slope: float
    rdm_lag1: float
    tuning_corr_median: float
    decoder_within: float
    decoder_cross: float
    decoder_drift_index: float
    n_units: int
    n_trials: int

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def drift_summary(X: np.ndarray, cond: np.ndarray, block: np.ndarray, decode: bool = True) -> DriftSummary:
    """Compute the standard set of drift statistics for one population."""
    pv = population_vector_correlation(X, cond, block)
    pvs = drift_rate_from_matrix(pv)
    rd = rdm_stability(X, cond, block)
    rds = drift_rate_from_matrix(rd)
    tc = tuning_correlation_per_unit(X, cond, block)
    if decode:
        acc = decoder_cross_time(X, cond, block)
        within = float(np.nanmean(np.diag(acc)))
        cross = float(np.nanmean(acc[~np.eye(acc.shape[0], dtype=bool)]))
        ddi = decoder_drift_index(acc)
    else:
        within = cross = ddi = np.nan
    return DriftSummary(pvs["lag1"], pvs["slope_per_block"], rds["lag1"], float(np.nanmedian(tc)),
                        within, cross, ddi, X.shape[1], X.shape[0])


def readout_shielding(tuning_corr: np.ndarray, decoder_index: float) -> float:
    """Single-unit drift (1 - median tuning correlation) minus population decoder drift index.

    Positive values mean the population readout is more stable than single units.
    """
    return float((1 - np.nanmedian(tuning_corr)) - decoder_index)
