"""Strategy-aware brain-age estimation with age-bias correction and grouped CV.

Strategies
----------
pooled          one model, sex not used.
pooled_sex      one model, sex (0/1) appended as a feature.
stratified      one model per sex.
pooled_sexbias  one model without sex, but age-bias correction fitted per sex.

Bias correction
---------------
'beheshti'  delta ~ a + b·age on training predictions; corrected = pred − (a + b·age)
            (Beheshti et al., 2019).
'cole'      pred ~ a + b·age on training data; corrected = (pred − a)/b
            (de Lange & Cole, 2020).
'none'      raw delta.
Both corrections are fitted on *training-fold* out-of-bag predictions only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .freesurfer import TIVCorrector, TIVMethod

Strategy = Literal["pooled", "pooled_sex", "stratified", "pooled_sexbias"]
BiasMethod = Literal["none", "beheshti", "cole"]


def base_regressor(kind: str = "ridge") -> Pipeline:
    if kind == "ridge":
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13))
    elif kind == "kernel_ridge":
        reg = KernelRidge(alpha=1.0, kernel="rbf", gamma=None)
    elif kind == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("reg", reg)])


@dataclass
class _BiasParams:
    a: float = 0.0
    b: float = 0.0


@dataclass
class BrainAgeEstimator:
    """Brain-age model implementing the four strategies with in-fold bias correction."""

    strategy: Strategy = "pooled_sex"
    bias: BiasMethod = "beheshti"
    tiv: TIVMethod = "residual"
    regressor: str = "ridge"
    inner_folds: int = 5
    seed: int = 0
    models_: dict = field(default_factory=dict, init=False)
    bias_: dict = field(default_factory=dict, init=False)
    tiv_: TIVCorrector | None = field(default=None, init=False)

    # -- helpers -----------------------------------------------------------
    def _design(self, X: pd.DataFrame, sex: np.ndarray) -> pd.DataFrame:
        Xd = self.tiv_.transform(X)
        if self.strategy == "pooled_sex":
            Xd = Xd.assign(sex_male=sex.astype(float))
        return Xd

    def _keys(self, sex: np.ndarray) -> list:
        return [0, 1] if self.strategy == "stratified" else ["all"]

    def _mask(self, key, sex: np.ndarray) -> np.ndarray:
        return (sex == key) if key in (0, 1) else np.ones_like(sex, dtype=bool)

    def _bias_keys(self) -> list:
        return [0, 1] if self.strategy in ("stratified", "pooled_sexbias") else ["all"]

    def _bias_key(self, s: int):
        return s if self.strategy in ("stratified", "pooled_sexbias") else "all"

    # -- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, age: np.ndarray, sex: np.ndarray, groups: np.ndarray | None = None) -> "BrainAgeEstimator":
        age = np.asarray(age, dtype=float)
        sex = np.asarray(sex, dtype=int)
        groups = np.arange(len(age)) if groups is None else np.asarray(groups)
        self.tiv_ = TIVCorrector(self.tiv).fit(X)
        Xd = self._design(X, sex)
        # inner OOB predictions for bias-correction fitting
        oob = np.full(len(age), np.nan)
        for key in self._keys(sex):
            m = self._mask(key, sex)
            idx = np.flatnonzero(m)
            splitter = GroupKFold(n_splits=min(self.inner_folds, len(np.unique(groups[idx]))))
            for tr, te in splitter.split(idx, groups=groups[idx]):
                mdl = base_regressor(self.regressor).fit(Xd.iloc[idx[tr]], age[idx[tr]])
                oob[idx[te]] = mdl.predict(Xd.iloc[idx[te]])
            self.models_[key] = base_regressor(self.regressor).fit(Xd.iloc[idx], age[idx])
        for bk in self._bias_keys():
            m = self._mask(bk, sex) if bk in (0, 1) else np.ones_like(sex, dtype=bool)
            self.bias_[bk] = self._fit_bias(oob[m], age[m])
        return self

    def _fit_bias(self, pred: np.ndarray, age: np.ndarray) -> _BiasParams:
        ok = ~np.isnan(pred)
        pred, age = pred[ok], age[ok]
        if self.bias == "none" or len(age) < 3:
            return _BiasParams(0.0, 0.0 if self.bias != "cole" else 1.0)
        if self.bias == "beheshti":
            b, a = np.polyfit(age, pred - age, 1)
            return _BiasParams(a, b)
        b, a = np.polyfit(age, pred, 1)  # cole
        return _BiasParams(a, b if abs(b) > 1e-6 else 1.0)

    # -- prediction --------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame, sex: np.ndarray) -> np.ndarray:
        sex = np.asarray(sex, dtype=int)
        Xd = self._design(X, sex)
        out = np.empty(len(sex))
        for key, mdl in self.models_.items():
            m = self._mask(key, sex)
            if m.any():
                out[m] = mdl.predict(Xd[m])
        return out

    def predict(self, X: pd.DataFrame, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
        """Bias-corrected predicted age."""
        age = np.asarray(age, dtype=float)
        sex = np.asarray(sex, dtype=int)
        raw = self.predict_raw(X, sex)
        out = raw.copy()
        for s in (0, 1):
            m = sex == s
            if not m.any():
                continue
            p = self.bias_[self._bias_key(s)]
            if self.bias == "beheshti":
                out[m] = raw[m] - (p.a + p.b * age[m])
            elif self.bias == "cole":
                out[m] = (raw[m] - p.a) / p.b
        return out

    def delta(self, X: pd.DataFrame, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
        return self.predict(X, age, sex) - np.asarray(age, dtype=float)


def cross_validated_delta(
    X: pd.DataFrame,
    age: np.ndarray,
    sex: np.ndarray,
    groups: np.ndarray,
    estimator: BrainAgeEstimator,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold predicted age, corrected delta and raw delta per subject (averaged over repeats)."""
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=int)
    groups = np.asarray(groups)
    pred = np.zeros((n_repeats, len(age)))
    raw = np.zeros_like(pred)
    uniq = np.unique(groups)
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        perm = rng.permutation(len(uniq))
        gmap = dict(zip(uniq, perm))
        g_shuffled = np.array([gmap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=g_shuffled):
            est = BrainAgeEstimator(**{k: getattr(estimator, k) for k in
                                       ("strategy", "bias", "tiv", "regressor", "inner_folds", "seed")})
            est.fit(X.iloc[tr], age[tr], sex[tr], groups[tr])
            pred[r, te] = est.predict(X.iloc[te], age[te], sex[te])
            raw[r, te] = est.predict_raw(X.iloc[te], sex[te])
    return pd.DataFrame({
        "age": age, "sex": sex, "group": groups,
        "pred_age": pred.mean(axis=0), "delta": pred.mean(axis=0) - age, "raw_delta": raw.mean(axis=0) - age,
    }, index=X.index)


def pooled_subsampled_control(
    X: pd.DataFrame, age: np.ndarray, sex: np.ndarray, groups: np.ndarray,
    estimator: BrainAgeEstimator, frac: float = 0.5, seed: int = 0, **cv_kwargs,
) -> pd.DataFrame:
    """Pooled strategy trained on a random ``frac`` of subjects (sample-size control for 'stratified')."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    keep = set(rng.choice(uniq, size=int(frac * len(uniq)), replace=False))
    m = np.array([g in keep for g in groups])
    est = BrainAgeEstimator(**{**{k: getattr(estimator, k) for k in ("bias", "tiv", "regressor", "inner_folds", "seed")},
                               "strategy": "pooled"})
    return cross_validated_delta(X[m], np.asarray(age)[m], np.asarray(sex)[m], np.asarray(groups)[m], est, **cv_kwargs)


def strategy_grid(strategies=("pooled", "pooled_sex", "stratified", "pooled_sexbias"),
                  tivs=("none", "covariate", "proportion", "residual", "power"),
                  bias: BiasMethod = "beheshti", regressor: str = "ridge") -> list[BrainAgeEstimator]:
    """All strategy × TIV-correction estimators for the factorial design."""
    return [BrainAgeEstimator(strategy=s, tiv=t, bias=bias, regressor=regressor) for s in strategies for t in tivs]
