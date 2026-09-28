"""Maturation clock: PMA regression, subject-grouped CV, bias-corrected deviation, reliability.

The deviation ("EEG-age delta") is predicted - true PMA.  Because regression
toward the mean makes raw deltas negatively correlated with age, the delta is
corrected by regressing it on age in the *training* data and subtracting the fit
(Smith et al., 2019, *NeuroImage*); :func:`bias_correct_delta` implements this and
:func:`delta_bias` reports the slope before/after.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def default_regressor() -> object:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


class MaturationClock:
    """Feature-based PMA regressor with subject-grouped cross-validation."""

    def __init__(self, make_model: Callable[[], object] = default_regressor):
        self.make_model = make_model
        self.model: Optional[object] = None
        self.bias_coef_: Optional[Tuple[float, float]] = None

    def fit(self, F: np.ndarray, pma: np.ndarray) -> "MaturationClock":
        F, pma = np.asarray(F, float), np.asarray(pma, float)
        self.model = self.make_model()
        self.model.fit(F, pma)
        delta = self.model.predict(F) - pma
        self.bias_coef_ = fit_bias(delta, pma)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        return self.model.predict(np.asarray(F, float))

    def delta(self, F: np.ndarray, pma: np.ndarray, corrected: bool = True) -> np.ndarray:
        d = self.predict(F) - np.asarray(pma, float)
        if corrected and self.bias_coef_ is not None:
            a, b = self.bias_coef_
            d = d - (a + b * np.asarray(pma, float))
        return d

    def cross_val_predict(self, F: np.ndarray, pma: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> Dict[str, np.ndarray]:
        """Out-of-fold predictions and deltas; the bias correction is fitted on each training fold."""
        F, pma, groups = np.asarray(F, float), np.asarray(pma, float), np.asarray(groups)
        pred = np.full(pma.shape, np.nan)
        delta_c = np.full(pma.shape, np.nan)
        k = min(n_splits, len(np.unique(groups)))
        for tr, te in GroupKFold(n_splits=k).split(F, pma, groups):
            m = self.make_model()
            m.fit(F[tr], pma[tr])
            a, b = fit_bias(m.predict(F[tr]) - pma[tr], pma[tr])
            pred[te] = m.predict(F[te])
            delta_c[te] = (pred[te] - pma[te]) - (a + b * pma[te])
        return {"pred": pred, "delta_raw": pred - pma, "delta_corrected": delta_c}


def fit_bias(delta: np.ndarray, age: np.ndarray) -> Tuple[float, float]:
    """Intercept and slope of delta ~ age (least squares)."""
    delta, age = np.asarray(delta, float), np.asarray(age, float)
    ok = np.isfinite(delta) & np.isfinite(age)
    if ok.sum() < 3 or age[ok].std() == 0:
        return 0.0, 0.0
    b, a = np.polyfit(age[ok], delta[ok], 1)
    return float(a), float(b)


def bias_correct_delta(delta: np.ndarray, age: np.ndarray, train_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract the age trend of delta fitted on ``train_mask`` rows (all rows if None)."""
    delta, age = np.asarray(delta, float), np.asarray(age, float)
    mask = np.ones(delta.shape, bool) if train_mask is None else np.asarray(train_mask, bool)
    a, b = fit_bias(delta[mask], age[mask])
    return delta - (a + b * age)


def delta_bias(delta: np.ndarray, age: np.ndarray) -> Dict[str, float]:
    """Slope and Pearson r of delta vs age."""
    a, b = fit_bias(delta, age)
    ok = np.isfinite(delta) & np.isfinite(age)
    r = float(np.corrcoef(delta[ok], age[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    return {"slope": b, "intercept": a, "r": r}


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.asarray(pred, float) - np.asarray(true, float)
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def mae_by_age_bin(pred: np.ndarray, true: np.ndarray, edges: Sequence[float] = (35, 37, 39, 41, 43, 45.01)) -> pd.DataFrame:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true >= lo) & (true < hi)
        rows.append(dict(bin=f"{lo}-{hi}", n=int(m.sum()), mae=mae(pred[m], true[m]) if m.any() else np.nan,
                         bias=float(np.nanmean(pred[m] - true[m])) if m.any() else np.nan))
    return pd.DataFrame(rows)


def icc_oneway(values: np.ndarray, subject_ids: np.ndarray) -> float:
    """ICC(1) (one-way random effects) of repeated values within subjects."""
    v, s = np.asarray(values, float), np.asarray(subject_ids)
    ok = np.isfinite(v)
    v, s = v[ok], s[ok]
    uniq = np.unique(s)
    k = np.array([np.sum(s == u) for u in uniq])
    if uniq.size < 2 or (k < 2).all():
        return float("nan")
    grand = v.mean()
    means = np.array([v[s == u].mean() for u in uniq])
    ss_between = np.sum(k * (means - grand) ** 2)
    ss_within = np.sum([np.sum((v[s == u] - m) ** 2) for u, m in zip(uniq, means)])
    df_b, df_w = uniq.size - 1, v.size - uniq.size
    if df_w <= 0:
        return float("nan")
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    k0 = (v.size - np.sum(k ** 2) / v.size) / df_b
    return float((ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w + 1e-12))


def permutation_mae_null(F: np.ndarray, pma: np.ndarray, groups: np.ndarray, n_perm: int = 50, seed: int = 0,
                         make_model: Callable[[], object] = default_regressor) -> Dict[str, float]:
    """MAE of the clock under subject-level permutation of PMA (chance level)."""
    rng = np.random.default_rng(seed)
    F, pma, groups = np.asarray(F, float), np.asarray(pma, float), np.asarray(groups)
    obs = mae(MaturationClock(make_model).cross_val_predict(F, pma, groups)["pred"], pma)
    uniq = np.unique(groups)
    sub_age = {u: pma[groups == u][0] for u in uniq}
    null = []
    for _ in range(n_perm):
        perm = dict(zip(uniq, rng.permutation([sub_age[u] for u in uniq])))
        p_age = np.asarray([perm[g] for g in groups])
        null.append(mae(MaturationClock(make_model).cross_val_predict(F, p_age, groups)["pred"], p_age))
    null = np.asarray(null)
    return {"mae": obs, "null_mean": float(null.mean()), "p": float((np.sum(null <= obs) + 1) / (n_perm + 1))}
