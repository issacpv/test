"""Modality baselines, late/early fusion and covariate residualization.

* :class:`ModalityModel` - standardized, class-balanced logistic regression on
  one feature block (EEG or ECG), with NaN-tolerant imputation.
* :class:`LateFusion` - fits one ModalityModel per block, obtains out-of-fold
  probabilities with subject-grouped CV on the training set, and stacks them
  with a logistic meta-model.  This is the "does ECG add to EEG" comparator.
* :func:`early_fusion_matrix` - concatenation baseline.
* :func:`residualize` - removes linear effects of covariates (time of day,
  vigilance proxy, movement) from a feature block, fitting on training rows only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_lr(C: float = 1.0, seed: int = 0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


class ModalityModel:
    """Logistic baseline on one feature block."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.pipe = _make_lr(C, seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModalityModel":
        self.pipe.fit(np.asarray(X, float), np.asarray(y, int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(np.asarray(X, float))[:, 1]


class LateFusion:
    """Stacked late fusion over named feature blocks with grouped out-of-fold stacking."""

    def __init__(self, C: float = 1.0, n_splits: int = 5, seed: int = 0):
        self.C, self.n_splits, self.seed = C, n_splits, seed
        self.models: Dict[str, ModalityModel] = {}
        self.meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        self.block_names: Sequence[str] = ()

    def _oof(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        oof = np.zeros(len(y))
        n_groups = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(self.n_splits, n_groups))
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                oof[te] = y[tr].mean()
                continue
            oof[te] = ModalityModel(self.C, self.seed).fit(X[tr], y[tr]).predict_proba(X[te])
        return oof

    def fit(self, blocks: Dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray) -> "LateFusion":
        y = np.asarray(y, int)
        groups = np.asarray(groups)
        self.block_names = list(blocks)
        Z = np.column_stack([self._oof(np.asarray(blocks[k], float), y, groups) for k in self.block_names])
        self.meta.fit(Z, y)
        for k in self.block_names:
            self.models[k] = ModalityModel(self.C, self.seed).fit(blocks[k], y)
        return self

    def predict_proba(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        Z = np.column_stack([self.models[k].predict_proba(blocks[k]) for k in self.block_names])
        return self.meta.predict_proba(Z)[:, 1]

    def modality_weights(self) -> Dict[str, float]:
        """Meta-model coefficients: how much each modality's probability contributes."""
        return {k: float(c) for k, c in zip(self.block_names, self.meta.coef_[0])}


def early_fusion_matrix(blocks: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate feature blocks (column order = dict order)."""
    return np.column_stack([np.asarray(v, float) for v in blocks.values()])


def residualize(X: np.ndarray, C: np.ndarray, train_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove the linear effect of covariates ``C`` from every column of ``X``.

    The regression is fitted on ``train_mask`` rows (all rows if None) and applied
    to all rows, so that the confound model never sees test data.  NaNs in ``X`` are
    ignored in the fit and preserved in the output.
    """
    X = np.asarray(X, float)
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    A = np.column_stack([np.ones(len(C)), C])
    mask = np.ones(len(X), dtype=bool) if train_mask is None else np.asarray(train_mask, bool)
    out = X.copy()
    for j in range(X.shape[1]):
        ok = mask & np.isfinite(X[:, j])
        if ok.sum() <= A.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(A[ok], X[ok, j], rcond=None)
        fin = np.isfinite(X[:, j])
        out[fin, j] = X[fin, j] - A[fin] @ beta + X[ok, j].mean()
    return out


def patient_specific_update(base: ModalityModel, X_new: np.ndarray, y_new: np.ndarray, X_old: np.ndarray,
                            y_old: np.ndarray, weight_new: float = 5.0) -> ModalityModel:
    """Cheap patient-specific fine-tuning: refit with the patient's own labelled windows up-weighted."""
    X = np.vstack([np.asarray(X_old, float), np.asarray(X_new, float)])
    y = np.concatenate([np.asarray(y_old, int), np.asarray(y_new, int)])
    w = np.concatenate([np.ones(len(y_old)), np.full(len(y_new), weight_new)])
    m = ModalityModel()
    m.pipe.fit(X, y, lr__sample_weight=w)
    return m
