"""Provenance detectability: grouped cross-validation, permutation nulls, feature contrasts, risk scores."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


def make_model(kind: str = "hgb", seed: int = 0):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed)
    raise ValueError("kind must be 'hgb' or 'rf'")


def _prepare(X: pd.DataFrame) -> np.ndarray:
    return X.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def grouped_cv_predictions(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], n_splits: int = 5,
                           kind: str = "hgb", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold labels and class probabilities with GroupKFold (or stratified CV when groups is None)."""
    Xa = _prepare(X)
    y = np.asarray(y)
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    pred = np.empty(len(y), dtype=object)
    if groups is not None:
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        splits = splitter.split(Xa, y, groups)
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xa, y)
    for tr, te in splits:
        model = make_model(kind, seed)
        if kind == "rf":
            Xtr = np.nan_to_num(Xa[tr], nan=0.0)
            Xte = np.nan_to_num(Xa[te], nan=0.0)
        else:
            Xtr, Xte = Xa[tr], Xa[te]
        model.fit(Xtr, y[tr])
        p = model.predict_proba(Xte)
        for j, c in enumerate(model.classes_):
            proba[te, np.flatnonzero(classes == c)[0]] = p[:, j]
        pred[te] = model.predict(Xte)
    return pred, proba


def detectability(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None, n_splits: int = 5,
                  kind: str = "hgb", n_repeats: int = 1, seed: int = 0) -> Dict[str, object]:
    """Balanced accuracy of provenance prediction with archives held out, and the detectability index.

    detectability = (BA - 1/K) / (1 - 1/K), where K is the number of classes (chance for balanced accuracy).
    Repeats reshuffle group assignment to folds (via a seed-dependent permutation of group ids).
    """
    y = np.asarray(y)
    K = len(np.unique(y))
    bas, f1s = [], []
    last_pred, last_proba = None, None
    for r in range(n_repeats):
        if groups is not None:
            rng = np.random.default_rng(seed + r)
            uniq = np.unique(groups)
            remap = dict(zip(uniq, rng.permutation(len(uniq))))
            g = np.array([remap[v] for v in groups])
        else:
            g = None
        pred, proba = grouped_cv_predictions(X, y, g, n_splits, kind, seed + r)
        bas.append(balanced_accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro"))
        last_pred, last_proba = pred, proba
    ba = float(np.mean(bas))
    chance = 1.0 / K
    return {"balanced_accuracy": ba, "balanced_accuracy_sd": float(np.std(bas)) if n_repeats > 1 else 0.0,
            "macro_f1": float(np.mean(f1s)), "chance": chance, "detectability": float((ba - chance) / (1 - chance)),
            "n_classes": K, "pred": last_pred, "proba": last_proba,
            "confusion": pd.crosstab(pd.Series(y, name="true"), pd.Series(last_pred, name="pred"))}


def archive_level_permutation_null(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_perm: int = 50,
                                   n_splits: int = 5, kind: str = "hgb", seed: int = 0) -> Dict[str, object]:
    """Null distribution of balanced accuracy when provenance labels are shuffled *across archives*.

    Each archive keeps a single label (as in reality: an archive uses one software for a set of
    cells), but which label it gets is permuted. This is the correct null for a grouped design;
    shuffling cell-level labels would destroy the group structure and give an optimistic null.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    # majority label per archive
    lab_of = {g: pd.Series(y[groups == g]).mode().iloc[0] for g in uniq}
    labels = np.array([lab_of[g] for g in uniq], dtype=object)
    obs = detectability(X, y, groups, n_splits, kind, seed=seed)["balanced_accuracy"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = dict(zip(uniq, rng.permutation(labels)))
        y_perm = np.array([perm_labels[g] for g in groups], dtype=object)
        if len(np.unique(y_perm)) < 2:
            null[i] = np.nan
            continue
        null[i] = detectability(X, y_perm, groups, n_splits, kind, seed=seed + i + 1)["balanced_accuracy"]
    null = null[np.isfinite(null)]
    p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if len(null) > 1 else np.nan,
            "p_perm": float(p), "detectability_vs_null": float((obs - null.mean()) / (1 - null.mean())) if null.mean() < 1 else np.nan}


def feature_subset_contrast(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], subsets: Dict[str, Sequence[str]],
                            n_splits: int = 5, kind: str = "hgb", seed: int = 0) -> pd.DataFrame:
    """Detectability per named feature subset (e.g. morphometrics-only vs sampling-only vs all)."""
    rows = []
    for name, cols in subsets.items():
        cols = [c for c in cols if c in X.columns]
        res = detectability(X[cols], y, groups, n_splits, kind, seed=seed)
        rows.append({"subset": name, "n_features": len(cols), "balanced_accuracy": res["balanced_accuracy"],
                     "detectability": res["detectability"], "macro_f1": res["macro_f1"]})
    return pd.DataFrame(rows)


def permutation_importance_grouped(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], n_repeats: int = 5,
                                   kind: str = "hgb", seed: int = 0) -> pd.DataFrame:
    """Permutation importance on a held-out group fold (drop in balanced accuracy per feature)."""
    from sklearn.inspection import permutation_importance

    Xa = _prepare(X)
    y = np.asarray(y)
    if groups is not None:
        tr, te = next(GroupKFold(n_splits=5).split(Xa, y, groups))
    else:
        tr, te = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Xa, y))
    model = make_model(kind, seed).fit(Xa[tr], y[tr])
    imp = permutation_importance(model, Xa[te], y[te], scoring="balanced_accuracy", n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_sd": imp.importances_std}
                        ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def provenance_risk(proba: np.ndarray, classes: Sequence[str]) -> pd.DataFrame:
    """Per-record provenance risk: max out-of-fold class probability and its entropy.

    High max-probability = the record carries a strong pipeline signature (high risk of leakage).
    """
    proba = np.asarray(proba, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(proba > 0, proba * np.log(proba), 0.0), axis=1)
    return pd.DataFrame({"risk_max_proba": proba.max(axis=1), "risk_entropy": ent,
                         "predicted": np.asarray(classes)[np.argmax(proba, axis=1)]})


__all__ = ["make_model", "grouped_cv_predictions", "detectability", "archive_level_permutation_null",
           "feature_subset_contrast", "permutation_importance_grouped", "provenance_risk"]
