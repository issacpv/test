"""Grouped nested cross-validated prediction with out-of-fold predictions and edge patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold


@dataclass
class CVResult:
    yhat: np.ndarray            # (n,) out-of-fold predictions
    fold: np.ndarray            # (n,) fold index of each subject
    patterns: np.ndarray        # (n_folds, p) Haufe activation patterns (or CPM masks)
    fold_r: np.ndarray          # (n_folds,) out-of-fold r per fold
    alphas: np.ndarray

    @property
    def r(self) -> float:
        return float(np.corrcoef(self.yhat, self._y)[0, 1]) if hasattr(self, "_y") else np.nan


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    n, p = Xc.shape
    if p > n:
        w = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(n), yc)
    else:
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    return w, xm, ym


def haufe_pattern(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Activation pattern ``cov(X) w`` (Haufe et al., 2014)."""
    Xc = X - X.mean(0)
    return (Xc.T @ (Xc @ w)) / max(X.shape[0] - 1, 1)


def _folds(n: int, groups: Optional[np.ndarray], n_splits: int, seed: int):
    groups = np.arange(n) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    remap = {g: i for i, g in enumerate(rng.permutation(uniq))}
    g = np.array([remap[x] for x in groups])
    return g, GroupKFold(n_splits=min(n_splits, len(uniq)))


def ridge_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    inner_splits: int = 3,
    seed: int = 0,
) -> CVResult:
    """Ridge regression with inner grouped CV for alpha; returns out-of-fold predictions."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    g, outer = _folds(n, groups, n_splits, seed)
    yhat, fold = np.empty(n), np.empty(n, int)
    patterns, fold_r, chosen = [], [], []
    for k, (tr, te) in enumerate(outer.split(X, y, g)):
        best, best_err = alphas[0], np.inf
        _, inner = _folds(len(tr), g[tr], inner_splits, seed + 1)
        for a in alphas:
            err = 0.0
            for itr, ite in inner.split(X[tr], y[tr], g[tr]):
                w, xm, ym = _ridge_fit(X[tr][itr], y[tr][itr], a)
                err += (((X[tr][ite] - xm) @ w + ym - y[tr][ite]) ** 2).sum()
            if err < best_err:
                best, best_err = a, err
        w, xm, ym = _ridge_fit(X[tr], y[tr], best)
        yhat[te] = (X[te] - xm) @ w + ym
        fold[te] = k
        patterns.append(haufe_pattern(X[tr], w))
        fold_r.append(np.corrcoef(yhat[te], y[te])[0, 1] if len(te) > 2 else np.nan)
        chosen.append(best)
    res = CVResult(yhat, fold, np.array(patterns), np.array(fold_r), np.array(chosen))
    res._y = y
    return res


def cpm_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    p_threshold: float = 0.01,
    seed: int = 0,
) -> CVResult:
    """Connectome-based predictive modelling (Shen et al., 2017): pos-neg edge sums + linear model."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    g, outer = _folds(n, groups, n_splits, seed)
    yhat, fold = np.empty(n), np.empty(n, int)
    patterns, fold_r = [], []
    for k, (tr, te) in enumerate(outer.split(X, y, g)):
        Xt, yt = X[tr], y[tr]
        Xs = (Xt - Xt.mean(0)) / np.where(Xt.std(0) == 0, np.inf, Xt.std(0))
        ys = (yt - yt.mean()) / yt.std()
        r = (Xs * ys[:, None]).sum(0) / len(yt)
        t = r * np.sqrt((len(yt) - 2) / np.clip(1 - r ** 2, 1e-12, None))
        pv = 2 * stats.t.sf(np.abs(t), df=len(yt) - 2)
        pos, neg = (pv < p_threshold) & (r > 0), (pv < p_threshold) & (r < 0)
        f_tr = Xt[:, pos].sum(1) - Xt[:, neg].sum(1)
        if f_tr.std() > 0:
            slope, intercept = np.polyfit(f_tr, yt, 1)
        else:
            slope, intercept = 0.0, yt.mean()
        f_te = X[te][:, pos].sum(1) - X[te][:, neg].sum(1)
        yhat[te] = slope * f_te + intercept
        fold[te] = k
        mask = np.zeros(p)
        mask[pos], mask[neg] = 1.0, -1.0
        patterns.append(mask)
        fold_r.append(np.corrcoef(yhat[te], y[te])[0, 1] if len(te) > 2 and yhat[te].std() > 0 else np.nan)
    res = CVResult(yhat, fold, np.array(patterns), np.array(fold_r), np.full(n_splits, np.nan))
    res._y = y
    return res


def prediction_accuracy(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = float(np.corrcoef(y, yhat)[0, 1]) if yhat.std() > 0 else 0.0
    ss = ((y - y.mean()) ** 2).sum()
    return {"r": r, "r2": float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else np.nan, "mae": float(np.abs(y - yhat).mean())}
