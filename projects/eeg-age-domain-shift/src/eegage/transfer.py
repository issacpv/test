"""Age-bin transfer matrix, performance-vs-age-distance decay fit and data-value curves.

All fits are subject-grouped: a subject never contributes windows to both the
training and the test side of any cell.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .ages import BIN_LABELS, bin_center, age_distance

ModelFactory = Callable[[], object]


def default_model_factory() -> object:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _fit_predict(model_factory: ModelFactory, Xtr, ytr, Xte) -> np.ndarray:
    m = model_factory()
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def transfer_matrix(F: np.ndarray, y: np.ndarray, groups: np.ndarray, bins: np.ndarray,
                    model_factory: ModelFactory = default_model_factory, n_splits: int = 5,
                    min_subjects: int = 3, labels: Sequence[str] = BIN_LABELS) -> pd.DataFrame:
    """AUROC of a model trained on bin *i* (rows) and tested on bin *j* (columns).

    Diagonal cells use subject-grouped ``n_splits``-fold CV.  Bins with fewer than
    ``min_subjects`` subjects or a single class are NaN.
    """
    F, y, groups, bins = np.asarray(F, float), np.asarray(y, int), np.asarray(groups), np.asarray(bins)
    present = [l for l in labels if (bins == l).any()]
    M = pd.DataFrame(np.nan, index=present, columns=present, dtype=float)
    for li in present:
        tr = bins == li
        if len(np.unique(groups[tr])) < min_subjects or len(np.unique(y[tr])) < 2:
            continue
        for lj in present:
            te = bins == lj
            if len(np.unique(y[te])) < 2:
                continue
            if li == lj:
                k = min(n_splits, len(np.unique(groups[tr])))
                if k < 2:
                    continue
                p = np.zeros(tr.sum())
                idx = np.where(tr)[0]
                for a, b in GroupKFold(n_splits=k).split(F[idx], y[idx], groups[idx]):
                    if len(np.unique(y[idx][a])) < 2:
                        p[b] = y[idx][a].mean()
                        continue
                    p[b] = _fit_predict(model_factory, F[idx][a], y[idx][a], F[idx][b])
                M.loc[li, lj] = _auc(y[idx], p)
            else:
                M.loc[li, lj] = _auc(y[te], _fit_predict(model_factory, F[tr], y[tr], F[te]))
    return M


def matrix_to_long(M: pd.DataFrame, same_site: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Long table with columns train, test, perf, dist (log-age), signed_dist, same_site."""
    rows = []
    for li in M.index:
        for lj in M.columns:
            v = M.loc[li, lj]
            if not np.isfinite(v):
                continue
            ci, cj = bin_center(li), bin_center(lj)
            rows.append(dict(train=li, test=lj, perf=float(v), dist=age_distance(ci, cj),
                             signed_dist=float(np.log1p(cj) - np.log1p(ci)),
                             same_site=bool(same_site.loc[li, lj]) if same_site is not None else True,
                             diagonal=li == lj))
    return pd.DataFrame(rows)


def fit_decay(long: pd.DataFrame, n_boot: int = 1000, seed: int = 0, use_signed: bool = False) -> Dict[str, float]:
    """OLS of performance on age distance (+ same-site indicator if it varies), with a bootstrap CI on the slope.

    With ``use_signed=True`` an extra term for signed distance (test older than train > 0)
    captures asymmetry (H4).
    """
    df = long.copy()
    cols = ["dist"]
    if use_signed:
        cols.append("signed_dist")
    if df["same_site"].nunique() > 1:
        cols.append("same_site")
    A = np.column_stack([np.ones(len(df))] + [df[c].astype(float).values for c in cols])
    yv = df["perf"].values
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ beta
    ss_res, ss_tot = ((yv - pred) ** 2).sum(), ((yv - yv.mean()) ** 2).sum() + 1e-12
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        if len(np.unique(df["dist"].values[idx])) < 2:
            continue
        b, *_ = np.linalg.lstsq(A[idx], yv[idx], rcond=None)
        boots.append(b)
    boots = np.asarray(boots) if boots else np.zeros((1, A.shape[1]))
    out = {"intercept": float(beta[0]), "slope_dist": float(beta[1]), "r2": float(1 - ss_res / ss_tot),
           "slope_dist_ci_lo": float(np.percentile(boots[:, 1], 2.5)),
           "slope_dist_ci_hi": float(np.percentile(boots[:, 1], 97.5)), "n_cells": int(len(df))}
    for k, c in enumerate(cols[1:], start=2):
        out[f"coef_{c}"] = float(beta[k])
    return out


def sample_value_curve(F: np.ndarray, y: np.ndarray, groups: np.ndarray, is_target_bin: np.ndarray,
                       ks: Sequence[int] = (2, 5, 10), far_multiplier: int = 5,
                       model_factory: ModelFactory = default_model_factory, n_rep: int = 5, seed: int = 0) -> pd.DataFrame:
    """Marginal value of age-matched vs far-age training subjects for a fixed target bin.

    Target subjects are split in half (train-eligible / evaluation).  Starting from all
    far-age subjects, we add ``k`` target-bin subjects or ``far_multiplier * k`` extra
    far-age subjects (sampled with replacement of *records*, i.e. duplicated windows are
    not created; instead we subsample the far set first so that additions are possible).
    """
    rng = np.random.default_rng(seed)
    F, y, groups = np.asarray(F, float), np.asarray(y, int), np.asarray(groups)
    tgt_subj = np.unique(groups[is_target_bin])
    far_subj = np.unique(groups[~is_target_bin])
    rows = []
    for rep in range(n_rep):
        rng.shuffle(tgt_subj)
        half = len(tgt_subj) // 2
        pool, evalset = tgt_subj[:half], tgt_subj[half:]
        te = np.isin(groups, evalset)
        if len(np.unique(y[te])) < 2:
            continue
        # base far set: hold out a reserve of far subjects that can be "added"
        rng.shuffle(far_subj)
        reserve_n = min(len(far_subj) // 2, max(ks) * far_multiplier)
        reserve, base_far = far_subj[:reserve_n], far_subj[reserve_n:]
        base = np.isin(groups, base_far)
        for k in ks:
            add_t = np.isin(groups, pool[:k])
            add_f = np.isin(groups, reserve[: k * far_multiplier])
            for kind, tr in (("age_matched", base | add_t), ("far_age", base | add_f), ("base", base)):
                if len(np.unique(y[tr])) < 2:
                    continue
                auc = _auc(y[te], _fit_predict(model_factory, F[tr], y[tr], F[te]))
                rows.append(dict(rep=rep, k=k, added=kind, n_train_subjects=len(np.unique(groups[tr])), auroc=auc))
    return pd.DataFrame(rows)
