"""Connectome-based predictive modelling (CPM) and ridge-on-FC with
family-aware cross-validation.

Key design constraints
----------------------
* Feature selection, confound residualisation and hyper-parameter tuning all
  happen *inside* the training fold (Rosenblatt et al., 2024, Nat Commun show
  that leakage inflates CPM accuracy).
* HCP families are never split across folds (:class:`FamilyKFold`).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Family-aware CV
# --------------------------------------------------------------------------- #
class FamilyKFold:
    """K-fold CV that keeps all members of a family in the same fold.

    Families are assigned to folds greedily (largest first) to balance fold
    sizes; when ``stratify`` labels are given (e.g. a subgroup label), the
    greedy assignment is done separately within each stratum so that each
    fold contains a similar share of every subgroup. A family is assigned to
    the stratum of the majority of its members.

    Parameters
    ----------
    n_splits
        Number of folds.
    shuffle
        Shuffle families before greedy assignment (recommended).
    random_state
        Seed for the shuffle.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, *args, **kwargs) -> int:  # sklearn-compatible
        return self.n_splits

    def _fold_of_family(
        self, families: np.ndarray, stratify: Optional[np.ndarray]
    ) -> Dict[object, int]:
        rng = np.random.default_rng(self.random_state)
        fam_ids, inverse = np.unique(families, return_inverse=True)
        sizes = np.bincount(inverse)
        if stratify is None:
            fam_stratum = np.zeros(len(fam_ids), dtype=object)
        else:
            stratify = np.asarray(stratify)
            fam_stratum = np.empty(len(fam_ids), dtype=object)
            for i in range(len(fam_ids)):
                members = stratify[inverse == i]
                vals, counts = np.unique(members, return_counts=True)
                fam_stratum[i] = vals[np.argmax(counts)]
        fold_sizes = np.zeros(self.n_splits, dtype=int)
        assignment: Dict[object, int] = {}
        for stratum in np.unique(fam_stratum):
            idx = np.where(fam_stratum == stratum)[0]
            if self.shuffle:
                idx = rng.permutation(idx)
            # largest families first for better balance, ties broken by shuffle
            idx = idx[np.argsort(-sizes[idx], kind="stable")]
            for i in idx:
                fold = int(np.argmin(fold_sizes))
                assignment[fam_ids[i]] = fold
                fold_sizes[fold] += sizes[i]
        return assignment

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        families: Optional[Sequence] = None,
        stratify: Optional[Sequence] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs."""
        n = X.shape[0]
        if families is None:
            families = np.arange(n)
        families = np.asarray(families)
        if families.shape[0] != n:
            raise ValueError("families must have one entry per row of X")
        assignment = self._fold_of_family(families, None if stratify is None else np.asarray(stratify))
        fold = np.array([assignment[f] for f in families])
        for k in range(self.n_splits):
            test = np.where(fold == k)[0]
            train = np.where(fold != k)[0]
            if len(test) == 0:
                continue
            yield train, test


def check_no_family_leak(train_idx: np.ndarray, test_idx: np.ndarray, families: Sequence) -> bool:
    """True if no family appears in both train and test."""
    fam = np.asarray(families)
    return len(set(fam[train_idx]) & set(fam[test_idx])) == 0


# --------------------------------------------------------------------------- #
# Confound residualisation (fit on train, applied to test)
# --------------------------------------------------------------------------- #
def residualize(
    X_train: np.ndarray,
    C_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    C_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Regress confounds ``C`` out of ``X`` using coefficients fit on train only.

    Works for feature matrices (``X`` is ``(n, p)``) and targets (``(n,)``).
    """
    Xt = np.asarray(X_train, dtype=float)
    squeeze = Xt.ndim == 1
    if squeeze:
        Xt = Xt[:, None]
    Ct = np.column_stack([np.ones(len(Xt)), np.asarray(C_train, dtype=float)])
    beta, *_ = np.linalg.lstsq(Ct, Xt, rcond=None)
    Xt_res = Xt - Ct @ beta
    Xs_res = None
    if X_test is not None:
        Xs = np.asarray(X_test, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        Cs = np.column_stack([np.ones(len(Xs)), np.asarray(C_test, dtype=float)])
        Xs_res = Xs - Cs @ beta
        if squeeze:
            Xs_res = Xs_res[:, 0]
    if squeeze:
        Xt_res = Xt_res[:, 0]
    return Xt_res, Xs_res


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def edgewise_correlation(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Pearson r (and two-sided p) of every column of X with y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / denom
    r = np.nan_to_num(r, nan=0.0)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return r, p


class CPMRegressor(BaseEstimator, RegressorMixin):
    """Connectome-based predictive modelling (Shen et al., 2017, Nat Protoc).

    Edges whose correlation with the target passes ``p_threshold`` in the
    training set are split into a positive and a negative network; the sum of
    edge strengths in each network is the feature for a linear model.

    Parameters
    ----------
    p_threshold
        Edge-selection p-value threshold (0.01 is the conventional choice).
    mode
        ``"both"`` (two features), ``"positive"``, ``"negative"`` or
        ``"difference"`` (pos sum minus neg sum, one feature).
    min_edges
        If fewer edges survive, fall back to the ``min_edges`` strongest
        absolute correlations so the model is always fit.
    """

    def __init__(self, p_threshold: float = 0.01, mode: str = "both", min_edges: int = 5):
        self.p_threshold = p_threshold
        self.mode = mode
        self.min_edges = min_edges

    def _features(self, X: np.ndarray) -> np.ndarray:
        pos = X[:, self.pos_mask_].sum(axis=1)
        neg = X[:, self.neg_mask_].sum(axis=1)
        if self.mode == "both":
            return np.column_stack([pos, neg])
        if self.mode == "positive":
            return pos[:, None]
        if self.mode == "negative":
            return neg[:, None]
        if self.mode == "difference":
            return (pos - neg)[:, None]
        raise ValueError(f"unknown mode {self.mode}")

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        r, p = edgewise_correlation(X, y)
        self.r_ = r
        pos = (r > 0) & (p < self.p_threshold)
        neg = (r < 0) & (p < self.p_threshold)
        if pos.sum() + neg.sum() < self.min_edges:
            top = np.argsort(-np.abs(r))[: self.min_edges]
            pos = np.zeros_like(pos)
            neg = np.zeros_like(neg)
            pos[top[r[top] > 0]] = True
            neg[top[r[top] < 0]] = True
        self.pos_mask_ = pos
        self.neg_mask_ = neg
        F = self._features(X)
        self.model_ = LinearRegression().fit(F, y, sample_weight=sample_weight)
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._features(np.asarray(X, dtype=float)))


class RidgeFC(BaseEstimator, RegressorMixin):
    """Standardised ridge regression on vectorised FC with inner-CV alpha.

    Parameters
    ----------
    alphas
        Grid searched by :class:`sklearn.linear_model.RidgeCV` (efficient
        leave-one-out by default).
    """

    def __init__(self, alphas: Optional[Sequence[float]] = None):
        self.alphas = alphas

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        alphas = self.alphas if self.alphas is not None else np.logspace(-2, 6, 17)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.model_ = RidgeCV(alphas=alphas).fit(Xs, y, sample_weight=sample_weight)
        self.alpha_ = float(self.model_.alpha_)
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.scaler_.transform(X))


# --------------------------------------------------------------------------- #
# Cross-validated predictions
# --------------------------------------------------------------------------- #
def cross_val_predict_family(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    families: Sequence,
    n_splits: int = 5,
    stratify: Optional[Sequence] = None,
    confounds: Optional[np.ndarray] = None,
    residualize_target: bool = False,
    sample_weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fit_params_fn: Optional[Callable[[np.ndarray], Dict]] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Out-of-fold predictions with family-aware folds and fold-wise confound removal.

    Parameters
    ----------
    estimator
        Any object with ``fit(X, y, sample_weight=...)`` / ``predict``.
    confounds
        ``(n, k)`` nuisance matrix (e.g. age, sex, mean FD) regressed out of
        each feature using training-fold coefficients only.
    residualize_target
        Also residualise ``y`` on the confounds (fit on train).
    sample_weight_fn
        ``f(train_idx) -> weights`` (for reweighting schemes).
    fit_params_fn
        ``f(train_idx) -> dict`` of extra keyword arguments for ``fit`` (e.g.
        ``{"groups": groups[train_idx]}`` for group-DRO).

    Returns
    -------
    pandas.DataFrame
        Columns ``y``, ``y_hat``, ``fold``; index = row position in ``X``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    families = np.asarray(families)
    cv = FamilyKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_hat = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for k, (tr, te) in enumerate(cv.split(X, y, families=families, stratify=stratify)):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        if confounds is not None:
            C = np.asarray(confounds, dtype=float)
            Xtr, Xte = residualize(Xtr, C[tr], Xte, C[te])
            if residualize_target:
                ytr, _ = residualize(ytr, C[tr])
        est = clone(estimator)
        kwargs: Dict = {}
        if sample_weight_fn is not None:
            kwargs["sample_weight"] = np.asarray(sample_weight_fn(tr), dtype=float)
        if fit_params_fn is not None:
            kwargs.update(fit_params_fn(tr))
        est.fit(Xtr, ytr, **kwargs)
        y_hat[te] = est.predict(Xte)
        fold_id[te] = k
    return pd.DataFrame({"y": y, "y_hat": y_hat, "fold": fold_id})


def repeated_cv_predictions(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    families: Sequence,
    n_repeats: int = 10,
    **kwargs,
) -> np.ndarray:
    """Stack out-of-fold predictions over repeated fold assignments ``(n_repeats, n)``."""
    preds = []
    for rep in range(n_repeats):
        df = cross_val_predict_family(estimator, X, y, families, random_state=rep, **kwargs)
        preds.append(df["y_hat"].to_numpy())
    return np.vstack(preds)
