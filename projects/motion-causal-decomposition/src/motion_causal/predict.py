"""Cross-validated prediction that returns per-run predictions for held-out subjects.

The model is trained on the subject-mean FC of the training subjects and applied to every run
of each held-out subject. Run-level predictions are what the within-subject artifact-sensitivity
estimator (:mod:`motion_causal.decomposition`) needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold


@dataclass
class RunPredictions:
    """Out-of-fold predictions: ``yhat_runs`` is ``(n_subjects, n_runs)``, ``yhat`` its row mean."""

    yhat_runs: np.ndarray
    yhat: np.ndarray
    weights: np.ndarray  # (n_folds, n_edges) fitted edge weights per fold
    alphas: np.ndarray

    @property
    def n_runs(self) -> int:
        return self.yhat_runs.shape[1]


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Ridge with centred X and y. Returns ``(w, intercept, x_mean, y_mean)``."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    n, p = Xc.shape
    if p > n:  # dual form
        K = Xc @ Xc.T
        a = np.linalg.solve(K + alpha * np.eye(n), yc)
        w = Xc.T @ a
    else:
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    return w, float(ym), xm, float(ym)


def _inner_cv_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float], n_splits: int) -> float:
    best, best_err = alphas[0], np.inf
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for a in alphas:
        err = 0.0
        for tr, te in gkf.split(X, y, groups):
            w, b, xm, _ = fit_ridge(X[tr], y[tr], a)
            pred = (X[te] - xm) @ w + b
            err += ((pred - y[te]) ** 2).sum()
        if err < best_err:
            best, best_err = a, err
    return float(best)


def cpm_select_edges(X: np.ndarray, y: np.ndarray, p_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Connectome-based predictive modelling edge selection (Shen et al., 2017).

    Returns boolean masks ``(positive_edges, negative_edges)`` for edges whose correlation with
    ``y`` is significant at ``p_threshold``.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    Xc = (X - X.mean(0)) / np.where(X.std(0) == 0, np.inf, X.std(0))
    yc = (y - y.mean()) / y.std()
    r = (Xc * yc[:, None]).sum(0) / n
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return (p < p_threshold) & (r > 0), (p < p_threshold) & (r < 0)


def crossval_run_predictions(
    X_runs: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    inner_splits: int = 3,
    method: str = "ridge",
    cpm_p: float = 0.01,
    seed: int = 0,
) -> RunPredictions:
    """Grouped K-fold prediction with per-run outputs.

    Parameters
    ----------
    X_runs
        ``(n_subjects, n_runs, n_edges)`` FC features per run (NaN runs allowed; they are
        ignored in the subject mean and produce NaN run predictions).
    y
        ``(n_subjects,)`` phenotype.
    groups
        Family / site identifiers used to keep related subjects in the same fold.
    method
        ``"ridge"`` (alpha by inner grouped CV) or ``"cpm"`` (positive-minus-negative edge sums
        with a univariate linear model).
    """
    X_runs = np.asarray(X_runs, float)
    y = np.asarray(y, float)
    n, R, p = X_runs.shape
    groups = np.arange(n) if groups is None else np.asarray(groups)
    Xbar = np.nanmean(X_runs, axis=1)
    rng = np.random.default_rng(seed)
    # shuffle group order for fold assignment reproducibly
    uniq = np.unique(groups)
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    g_perm = np.array([perm[g] for g in groups])
    gkf = GroupKFold(n_splits=min(n_splits, len(uniq)))
    yhat_runs = np.full((n, R), np.nan)
    weights, chosen = [], []
    for tr, te in gkf.split(Xbar, y, g_perm):
        if method == "ridge":
            a = _inner_cv_alpha(Xbar[tr], y[tr], g_perm[tr], alphas, inner_splits)
            w, b, xm, _ = fit_ridge(Xbar[tr], y[tr], a)
            for r in range(R):
                yhat_runs[te, r] = (X_runs[te, r] - xm) @ w + b
            weights.append(w)
            chosen.append(a)
        elif method == "cpm":
            pos, neg = cpm_select_edges(Xbar[tr], y[tr], cpm_p)
            feat = Xbar[tr][:, pos].sum(1) - Xbar[tr][:, neg].sum(1)
            slope, intercept = np.polyfit(feat, y[tr], 1) if feat.std() > 0 else (0.0, y[tr].mean())
            for r in range(R):
                f = X_runs[te, r][:, pos].sum(1) - X_runs[te, r][:, neg].sum(1)
                yhat_runs[te, r] = slope * f + intercept
            w = np.zeros(p)
            w[pos], w[neg] = slope, -slope
            weights.append(w)
            chosen.append(np.nan)
        else:
            raise ValueError("method must be 'ridge' or 'cpm'")
    return RunPredictions(yhat_runs=yhat_runs, yhat=np.nanmean(yhat_runs, axis=1), weights=np.array(weights), alphas=np.array(chosen))


def haufe_pattern(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Activation pattern ``cov(X) w`` (Haufe et al., 2014) for an interpretable edge map."""
    Xc = X - X.mean(0)
    return (Xc.T @ (Xc @ w)) / (X.shape[0] - 1)
