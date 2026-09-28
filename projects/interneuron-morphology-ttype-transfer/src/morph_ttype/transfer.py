"""Cross-dataset / cross-species transfer of morphology -> subclass classifiers.

Includes per-dataset standardisation, CORAL alignment (Sun, Feng & Saenko, 2016),
within-dataset stratified CV, leave-dataset-out evaluation, a position-only
baseline, and permutation nulls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(kind: str = "logreg", seed: int = 0) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, min_samples_leaf=2)
    else:
        raise ValueError(kind)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])


def standardise_per_group(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Z-score every feature within each dataset (removes dataset mean/scale shifts)."""
    out = X.copy()
    for g, idx in groups.groupby(groups).groups.items():
        sub = out.loc[idx]
        sd = sub.std(ddof=0).replace(0, 1.0)
        out.loc[idx] = (sub - sub.mean()) / sd
    return out


def coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """CORAL: re-colour source features so their covariance matches the target's.

    Xs_aligned = Xs @ Cs^{-1/2} @ Ct^{1/2}, with identity regularisation ``eps``.
    Features should already be centred/standardised per dataset.
    """
    Xs = np.asarray(Xs, float)
    Xt = np.asarray(Xt, float)
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)

    def _sqrt(C: np.ndarray, inv: bool) -> np.ndarray:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        s = w ** (-0.5) if inv else w ** 0.5
        return (V * s) @ V.T

    return Xs @ _sqrt(Cs, True) @ _sqrt(Ct, False)


@dataclass
class Result:
    balanced_accuracy: float
    macro_f1: float
    n_test: int
    design: str
    per_class_recall: Dict[str, float]


def _score(y_true: Sequence[str], y_pred: Sequence[str], design: str) -> Result:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true))
    rec = {c: float(np.mean(y_pred[y_true == c] == c)) for c in classes}
    return Result(float(balanced_accuracy_score(y_true, y_pred)),
                  float(f1_score(y_true, y_pred, average="macro")), int(len(y_true)), design, rec)


def within_dataset_cv(X: pd.DataFrame, y: pd.Series, kind: str = "logreg", n_splits: int = 5, seed: int = 0) -> Result:
    """Stratified K-fold within one dataset."""
    y = y.astype(str)
    n_splits = min(n_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError("need at least 2 cells per class for CV")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    Xv = X.to_numpy(float)
    for tr, te in skf.split(Xv, y):
        m = make_classifier(kind, seed).fit(Xv[tr], y.iloc[tr])
        pred[te] = m.predict(Xv[te])
    return _score(y, pred, "within")


def transfer(X_src: pd.DataFrame, y_src: pd.Series, X_tgt: pd.DataFrame, y_tgt: pd.Series, kind: str = "logreg",
             align: str = "none", seed: int = 0) -> Result:
    """Train on source, test on target. ``align`` in {"none", "standardise", "coral"}.

    "standardise" z-scores each dataset separately (unsupervised, no target labels);
    "coral" additionally matches second-order statistics. Target labels are used only
    for scoring.
    """
    cols = [c for c in X_src.columns if c in X_tgt.columns]
    Xs = X_src[cols].copy()
    Xt = X_tgt[cols].copy()
    if align in ("standardise", "coral"):
        Xs = (Xs - Xs.mean()) / Xs.std(ddof=0).replace(0, 1.0)
        Xt = (Xt - Xt.mean()) / Xt.std(ddof=0).replace(0, 1.0)
    Xs_v = SimpleImputer(strategy="median").fit(pd.concat([Xs, Xt])).transform(Xs)
    Xt_v = SimpleImputer(strategy="median").fit(pd.concat([Xs, Xt])).transform(Xt)
    if align == "coral":
        Xs_v = coral(Xs_v, Xt_v)
    m = make_classifier(kind, seed).fit(Xs_v, y_src.astype(str))
    return _score(y_tgt.astype(str), m.predict(Xt_v), f"transfer:{align}")


def leave_dataset_out(X: pd.DataFrame, y: pd.Series, dataset: pd.Series, kind: str = "logreg",
                      align: str = "standardise", seed: int = 0) -> Dict[str, Result]:
    """Hold out each dataset in turn, train on the others."""
    out = {}
    for d in sorted(dataset.unique()):
        te = dataset == d
        out[str(d)] = transfer(X[~te], y[~te], X[te], y[te], kind=kind, align=align, seed=seed)
    return out


def position_only_baseline(pos: pd.DataFrame, y: pd.Series, kind: str = "logreg", seed: int = 0) -> Result:
    """Reference floor: predict subclass from soma depth / layer only."""
    return within_dataset_cv(pos, y, kind=kind, seed=seed)


def permutation_null(fn, y: pd.Series, n_perm: int = 100, seed: int = 0, **kw) -> Tuple[float, np.ndarray, float]:
    """Generic label-permutation null for any scoring function ``fn(y_perm, **kw) -> Result``.

    Returns (observed balanced accuracy, null distribution, p-value).
    """
    rng = np.random.default_rng(seed)
    obs = fn(y, **kw).balanced_accuracy
    null = np.array([fn(pd.Series(rng.permutation(y.to_numpy()), index=y.index), **kw).balanced_accuracy
                     for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, null, p


def bootstrap_gap(res_within: Result, X_src: pd.DataFrame, y_src: pd.Series, X_tgt: pd.DataFrame, y_tgt: pd.Series,
                  n_boot: int = 200, seed: int = 0, **kw) -> Dict[str, float]:
    """Bootstrap (over target cells) the within-target minus transfer balanced-accuracy gap."""
    rng = np.random.default_rng(seed)
    gaps = []
    y_tgt = y_tgt.astype(str)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_tgt), len(y_tgt))
        if len(set(y_tgt.iloc[idx])) < 2:
            continue
        r = transfer(X_src, y_src, X_tgt.iloc[idx], y_tgt.iloc[idx], **kw)
        gaps.append(res_within.balanced_accuracy - r.balanced_accuracy)
    gaps = np.asarray(gaps)
    return {"gap_mean": float(gaps.mean()), "gap_ci_low": float(np.percentile(gaps, 2.5)),
            "gap_ci_high": float(np.percentile(gaps, 97.5)), "n_boot": int(len(gaps))}
