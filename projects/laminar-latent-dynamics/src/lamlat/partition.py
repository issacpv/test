"""Encoding models and stimulus-vs-behaviour variance partitioning per unit (or per latent dimension).

Following the nested-model logic of Musall et al. (2019): unique variance of a regressor group is the drop in
cross-validated R^2 when that group is removed from the full model; shared variance is what remains.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .latent import blocked_folds


def onehot_design(conditions: np.ndarray, bin_condition: np.ndarray, levels: Optional[Sequence] = None) -> np.ndarray:
    """One-hot stimulus design per time bin from a per-bin condition label (-1 or NaN = no stimulus)."""
    lv = list(levels) if levels is not None else [c for c in pd.unique(conditions) if not (isinstance(c, float) and np.isnan(c)) and c != -1]
    D = np.zeros((len(bin_condition), len(lv)))
    for j, c in enumerate(lv):
        D[:, j] = (bin_condition == c).astype(float)
    return D


def lagged_design(x: np.ndarray, lags: Iterable[int]) -> np.ndarray:
    """Stack time-shifted copies of a regressor (positive lag = past values), zero-padded."""
    x = np.asarray(x, dtype=float)
    cols = []
    for lag in lags:
        y = np.zeros_like(x)
        if lag >= 0:
            y[lag:] = x[: len(x) - lag] if lag > 0 else x
        else:
            y[:lag] = x[-lag:]
        cols.append(y)
    return np.column_stack(cols)


def ridge_cv_r2(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0, n_folds: int = 5) -> np.ndarray:
    """Blocked-CV R^2 per target column for a ridge model with intercept (NaN-safe on constant targets)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    ss_res = np.zeros(Y.shape[1])
    ss_tot = np.zeros(Y.shape[1])
    for tr, te in blocked_folds(len(X), n_folds):
        model = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        pred = model.predict(X[te])
        ss_res += np.sum((Y[te] - pred) ** 2, axis=0)
        ss_tot += np.sum((Y[te] - Y[tr].mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def select_alpha(X: np.ndarray, Y: np.ndarray, alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0), n_folds: int = 5) -> float:
    """Alpha maximising the mean CV R^2 across targets."""
    best, best_r2 = alphas[0], -np.inf
    for a in alphas:
        r2 = np.nanmean(ridge_cv_r2(X, Y, a, n_folds))
        if r2 > best_r2:
            best, best_r2 = a, r2
    return float(best)


def variance_partition(Y: np.ndarray, groups: Dict[str, np.ndarray], alpha: float = 1.0, n_folds: int = 5) -> pd.DataFrame:
    """Unique and shared CV R^2 per target for named regressor groups (e.g., {'stim': Xs, 'behav': Xb}).

    Returns a DataFrame with columns full, unique_<group> for each group, shared (= full - sum of uniques,
    clipped at 0) and single_<group> (each group alone)."""
    names = list(groups)
    Xfull = np.column_stack([groups[n] for n in names])
    full = ridge_cv_r2(Xfull, Y, alpha, n_folds)
    out = {"full": full}
    uniq_sum = np.zeros_like(full)
    for n in names:
        others = [groups[m] for m in names if m != n]
        if others:
            reduced = ridge_cv_r2(np.column_stack(others), Y, alpha, n_folds)
        else:
            reduced = np.zeros_like(full)
        u = np.clip(full - reduced, 0, None)
        out[f"unique_{n}"] = u
        uniq_sum += np.nan_to_num(u)
        out[f"single_{n}"] = ridge_cv_r2(groups[n], Y, alpha, n_folds)
    out["shared"] = np.clip(full - uniq_sum, 0, None)
    return pd.DataFrame(out)


def partition_by_layer(part: pd.DataFrame, labels: np.ndarray, groups: Sequence[str]) -> pd.DataFrame:
    """Median unique/shared R^2 per layer label (units as rows of ``part``)."""
    df = part.copy()
    df["layer"] = np.asarray(labels)
    cols = ["full", "shared"] + [f"unique_{g}" for g in groups]
    return df.groupby("layer")[cols].median()


def synthetic_encoding_data(n_time: int, rng: np.random.Generator, n_stim_units: int = 10, n_behav_units: int = 10,
                            n_conditions: int = 8, noise: float = 0.5) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Units driven either by a one-hot stimulus or by a smooth behavioural signal (for tests)."""
    cond = rng.integers(0, n_conditions, size=n_time)
    Xs = onehot_design(np.arange(n_conditions), cond, levels=list(range(n_conditions)))
    run = np.convolve(rng.normal(size=n_time), np.ones(20) / 20, mode="same")
    run = (run - run.mean()) / run.std()
    Xb = lagged_design(run, [0, 1, 2])
    Ys = Xs @ rng.normal(size=(n_conditions, n_stim_units)) + noise * rng.normal(size=(n_time, n_stim_units))
    Yb = np.outer(run, rng.normal(size=n_behav_units)) * 3 + noise * rng.normal(size=(n_time, n_behav_units))
    Y = np.hstack([Ys, Yb])
    labels = np.array(["stim"] * n_stim_units + ["behav"] * n_behav_units)
    return Y, {"stim": Xs, "behav": Xb}, labels
