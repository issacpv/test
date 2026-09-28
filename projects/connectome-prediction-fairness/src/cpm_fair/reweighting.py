"""Group-aware training baselines: inverse-frequency reweighting, balanced
subsampling and a group-DRO ridge (Sagawa et al., 2020, ICLR).

These are the "does group-aware training fix it without loss?" arm of the
audit. Balanced subsampling / balanced weighting were the best performers in
the 2025-2026 ABCD benchmarks; group-DRO has, to our knowledge, not been
evaluated for connectome-based regression.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .predictors import cross_val_predict_family
from .subgroup_metrics import subgroup_performance


def inverse_frequency_weights(groups: Sequence, power: float = 1.0) -> np.ndarray:
    """Sample weights ``(n_total / (n_groups * n_g)) ** power``, normalised to mean 1."""
    groups = np.asarray(groups)
    uniq, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = (len(groups) / (len(uniq) * counts[inverse])) ** power
    return w / w.mean()


def balanced_subsample(
    groups: Sequence, n_per_group: Optional[int] = None, seed: int = 0
) -> np.ndarray:
    """Indices of a subsample with equal size per group (default: size of the smallest group)."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    k = int(counts.min()) if n_per_group is None else int(n_per_group)
    idx = []
    for g in uniq:
        members = np.where(groups == g)[0]
        if len(members) < k:
            raise ValueError(f"group {g!r} has fewer than {k} members")
        idx.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(idx))


class GroupDRORidge(BaseEstimator, RegressorMixin):
    """Group distributionally robust ridge regression.

    Minimises the worst-case (over groups) mean squared error by
    exponentiated-gradient ascent on group weights ``q`` interleaved with a
    closed-form weighted ridge solve (the online algorithm of Sagawa et al.,
    2020, specialised to a convex least-squares inner problem).

    Parameters
    ----------
    alpha
        L2 penalty.
    n_iter
        Number of outer iterations.
    eta
        Step size for the group-weight update.
    q_floor
        Minimum weight for any group (keeps the objective from collapsing on
        one group; 0 = pure DRO).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_iter: int = 50,
        eta: float = 0.5,
        q_floor: float = 0.0,
        standardize: bool = True,
    ):
        self.alpha = alpha
        self.n_iter = n_iter
        self.eta = eta
        self.q_floor = q_floor
        self.standardize = standardize

    def _weighted_ridge(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n, p = X.shape
        Xw = X * w[:, None]
        A = X.T @ Xw + self.alpha * np.eye(p)
        b = Xw.T @ y
        return np.linalg.solve(A, b)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Sequence, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_sd_ = X.std(axis=0, ddof=0) if self.standardize else np.ones(X.shape[1])
        self.x_sd_[self.x_sd_ == 0] = 1.0
        self.y_mean_ = float(np.average(y, weights=sample_weight))
        Xs = (X - self.x_mean_) / self.x_sd_
        yc = y - self.y_mean_
        uniq, inverse = np.unique(groups, return_inverse=True)
        G = len(uniq)
        q = np.full(G, 1.0 / G)
        history: List[Dict[str, float]] = []
        coef = None
        for it in range(self.n_iter):
            # per-sample weights: q_g / n_g so that each group's mean loss is weighted by q_g
            n_g = np.bincount(inverse, weights=sample_weight)
            w = sample_weight * q[inverse] / n_g[inverse] * len(y)
            coef = self._weighted_ridge(Xs, yc, w)
            resid = yc - Xs @ coef
            group_loss = np.array(
                [np.average(resid[inverse == g] ** 2, weights=sample_weight[inverse == g]) for g in range(G)]
            )
            history.append({"iter": it, "worst_group_mse": float(group_loss.max()), **{f"q_{u}": float(q[i]) for i, u in enumerate(uniq)}})
            q = q * np.exp(self.eta * group_loss / (group_loss.mean() + 1e-12))
            q = q / q.sum()
            if self.q_floor > 0:
                q = np.maximum(q, self.q_floor)
                q = q / q.sum()
        self.coef_ = coef
        self.group_weights_ = pd.Series(q, index=uniq)
        self.history_ = pd.DataFrame(history)
        self.groups_ = uniq
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_sd_
        return Xs @ self.coef_ + self.y_mean_


class ERMRidge(GroupDRORidge):
    """Same closed-form ridge without any group weighting (n_iter = 1, uniform q)."""

    def __init__(self, alpha: float = 1.0, standardize: bool = True):
        super().__init__(alpha=alpha, n_iter=1, eta=0.0, q_floor=0.0, standardize=standardize)


def evaluate_training_schemes(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    families: Sequence,
    alpha: float = 1.0,
    n_splits: int = 5,
    schemes: Sequence[str] = ("erm", "inverse_frequency", "balanced_subsample", "group_dro"),
    confounds: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Run each training scheme under identical family-aware folds and return
    per-subgroup metric tables (keys = scheme names).

    ``balanced_subsample`` trains only on a balanced subset of the training
    fold but is evaluated on the full test fold, so all schemes share the same
    out-of-fold subjects.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    results: Dict[str, pd.DataFrame] = {}
    for scheme in schemes:
        if scheme == "erm":
            est = ERMRidge(alpha=alpha)
            preds = cross_val_predict_family(
                est, X, y, families, n_splits=n_splits, stratify=groups, confounds=confounds,
                fit_params_fn=lambda tr: {"groups": groups[tr]}, random_state=random_state,
            )
        elif scheme == "inverse_frequency":
            est = ERMRidge(alpha=alpha)
            preds = cross_val_predict_family(
                est, X, y, families, n_splits=n_splits, stratify=groups, confounds=confounds,
                sample_weight_fn=lambda tr: inverse_frequency_weights(groups[tr]),
                fit_params_fn=lambda tr: {"groups": groups[tr]}, random_state=random_state,
            )
        elif scheme == "balanced_subsample":
            est = ERMRidge(alpha=alpha)

            def _w(tr, _groups=groups):
                keep = balanced_subsample(_groups[tr], seed=random_state)
                w = np.zeros(len(tr))
                w[keep] = 1.0
                return w

            preds = cross_val_predict_family(
                est, X, y, families, n_splits=n_splits, stratify=groups, confounds=confounds,
                sample_weight_fn=_w, fit_params_fn=lambda tr: {"groups": groups[tr]}, random_state=random_state,
            )
        elif scheme == "group_dro":
            est = GroupDRORidge(alpha=alpha)
            preds = cross_val_predict_family(
                est, X, y, families, n_splits=n_splits, stratify=groups, confounds=confounds,
                fit_params_fn=lambda tr: {"groups": groups[tr]}, random_state=random_state,
            )
        else:
            raise ValueError(f"unknown scheme {scheme}")
        results[scheme] = subgroup_performance(preds["y"], preds["y_hat"], groups)
    return results
