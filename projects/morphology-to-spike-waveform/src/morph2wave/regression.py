"""Morphology -> waveform regression and factorial variance partition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CVResult:
    r2: float
    r2_folds: np.ndarray
    rmse: float
    importance: pd.Series          # permutation importance (mean decrease in R^2)
    predictions: np.ndarray


def _model(kind: str):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    if kind == "gbr":
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    raise ValueError(kind)


def cross_validated_regression(X: pd.DataFrame, y: pd.Series, groups: Sequence, kind: str = "ridge",
                               n_splits: int = 5, seed: int = 0) -> CVResult:
    """Grouped K-fold (all placements of one cell in the same fold) regression of one waveform feature.

    Permutation importance is computed on each held-out fold and averaged.
    """
    Xa = X.to_numpy(float)
    ya = y.to_numpy(float)
    ok = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
    Xa, ya, g = Xa[ok], ya[ok], np.asarray(groups)[ok]
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(g))))
    preds = np.zeros_like(ya)
    r2s, imps = [], []
    for tr, te in gkf.split(Xa, ya, g):
        m = _model(kind).fit(Xa[tr], ya[tr])
        preds[te] = m.predict(Xa[te])
        ss_res = np.sum((ya[te] - preds[te]) ** 2)
        ss_tot = np.sum((ya[te] - ya[te].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        pi = permutation_importance(m, Xa[te], ya[te], n_repeats=10, random_state=seed, scoring="r2")
        imps.append(pi.importances_mean)
    ss_res = np.sum((ya - preds) ** 2)
    ss_tot = np.sum((ya - ya.mean()) ** 2)
    return CVResult(r2=float(1 - ss_res / ss_tot), r2_folds=np.array(r2s), rmse=float(np.sqrt(ss_res / len(ya))),
                    importance=pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False),
                    predictions=preds)


def shuffled_negative_control(X: pd.DataFrame, y: pd.Series, groups: Sequence, kind: str = "ridge", seed: int = 0) -> float:
    """R^2 after shuffling morphology rows across cells (should be about 0 or negative)."""
    rng = np.random.default_rng(seed)
    Xs = X.iloc[rng.permutation(len(X))].reset_index(drop=True)
    return cross_validated_regression(Xs, y.reset_index(drop=True), groups, kind=kind).r2


def factorial_variance_partition(df: pd.DataFrame, value: str, factor_a: str = "morphology",
                                 factor_b: str = "channel_set", nested: Optional[str] = "placement") -> Dict[str, float]:
    """Two-way (A x B) sum-of-squares partition with an optional nested placement factor.

    Computes eta^2 for A, B, their interaction, the nested factor (within A x B cells) and the
    residual, from cell means of a balanced or near-balanced design:
        SS_A     = sum_i n_i (m_i - m)^2
        SS_B     = sum_j n_j (m_j - m)^2
        SS_AB    = sum_ij n_ij (m_ij - m_i - m_j + m)^2
        SS_nest  = sum_ijk n_ijk (m_ijk - m_ij)^2
        SS_res   = remaining within-cell variance
    """
    d = df[[factor_a, factor_b, value] + ([nested] if nested else [])].dropna()
    y = d[value].to_numpy(float)
    m = y.mean()
    ss_tot = np.sum((y - m) ** 2)
    m_a = d.groupby(factor_a)[value].transform("mean").to_numpy()
    m_b = d.groupby(factor_b)[value].transform("mean").to_numpy()
    m_ab = d.groupby([factor_a, factor_b])[value].transform("mean").to_numpy()
    ss_a = np.sum((m_a - m) ** 2)
    ss_b = np.sum((m_b - m) ** 2)
    ss_ab = np.sum((m_ab - m_a - m_b + m) ** 2)
    if nested:
        m_n = d.groupby([factor_a, factor_b, nested])[value].transform("mean").to_numpy()
        ss_nest = np.sum((m_n - m_ab) ** 2)
        ss_res = np.sum((y - m_n) ** 2)
    else:
        ss_nest = 0.0
        ss_res = np.sum((y - m_ab) ** 2)
    out = {"eta2_" + factor_a: ss_a / ss_tot, "eta2_" + factor_b: ss_b / ss_tot, "eta2_interaction": ss_ab / ss_tot,
           "eta2_residual": ss_res / ss_tot, "ss_total": float(ss_tot), "n": int(len(y))}
    if nested:
        out["eta2_" + nested] = ss_nest / ss_tot
    return {k: float(v) for k, v in out.items()}


def bootstrap_partition(df: pd.DataFrame, value: str, unit: str = "morphology", n_boot: int = 500, seed: int = 0,
                        **kwargs) -> pd.DataFrame:
    """Bootstrap the variance partition over cells (resampling levels of ``unit``)."""
    rng = np.random.default_rng(seed)
    levels = df[unit].unique()
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(levels, len(levels), replace=True)
        boot = pd.concat([df[df[unit] == lv].assign(**{unit: f"{lv}_{k}"}) for k, lv in enumerate(pick)])
        rows.append(factorial_variance_partition(boot, value, factor_a=unit, **kwargs))
    b = pd.DataFrame(rows)
    return b.describe(percentiles=[0.025, 0.5, 0.975]).T[["mean", "2.5%", "50%", "97.5%"]]
