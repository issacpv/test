"""Covariate baseline vs ROI models with nested, subject-grouped cross-validation.

The central comparison of the project is *paired*: the same outer folds are
used for every model so that ΔAUC can be bootstrapped over participants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COVARIATES = ("age", "sex_male", "apoe_e4", "education", "mmse", "cdr_sb")


def make_model(kind: str) -> tuple[Pipeline, dict]:
    """Return an sklearn pipeline and its inner-CV grid for ``kind``."""
    if kind == "logistic":
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
        grid = {"clf__C": [0.1, 1.0, 10.0]}
    elif kind == "elasticnet":
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000))])
        grid = {"clf__C": [0.05, 0.2, 1.0], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    elif kind == "hgb":
        pipe = Pipeline([("clf", HistGradientBoostingClassifier(max_iter=200, early_stopping=False))])
        grid = {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]}
    else:
        raise ValueError(kind)
    return pipe, grid


@dataclass
class CVResult:
    name: str
    oof_prob: np.ndarray  # shape (n_repeats, n)
    y: np.ndarray
    groups: np.ndarray

    def auc(self) -> float:
        return float(np.mean([roc_auc_score(self.y, p) for p in self.oof_prob]))

    def brier(self) -> float:
        return float(np.mean([brier_score_loss(self.y, p) for p in self.oof_prob]))

    def mean_prob(self) -> np.ndarray:
        return self.oof_prob.mean(axis=0)


def nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logistic",
    *,
    n_outer: int = 5,
    n_inner: int = 3,
    n_repeats: int = 2,
    seed: int = 0,
    name: str | None = None,
) -> CVResult:
    """Repeated nested StratifiedGroupKFold CV returning out-of-fold probabilities.

    Inner grid search is grouped as well; Platt scaling is left to the caller
    (``platt_scale``) so that calibration can be assessed both ways.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    Xv = X.to_numpy(dtype=float)
    pipe, grid = make_model(kind)
    oof = np.full((n_repeats, len(y)), np.nan)
    for r in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed + r)
        for tr, te in outer.split(Xv, y, groups):
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 100 + r)
            gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="roc_auc", n_jobs=1)
            gs.fit(Xv[tr], y[tr], groups=groups[tr])
            oof[r, te] = gs.best_estimator_.predict_proba(Xv[te])[:, 1]
    return CVResult(name or kind, oof, y, groups)


def platt_scale(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit logit(y) ~ a + b·logit(p); returns (a, b) = calibration intercept & slope."""
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def bootstrap_delta_auc(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster (subject-level) bootstrap of AUC(b) − AUC(a).

    Returns point estimate, percentile 95% CI and a two-sided bootstrap p-value
    for the null Δ = 0.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(yb, p_b[idx]) - roc_auc_score(yb, p_a[idx])
    deltas = deltas[~np.isnan(deltas)]
    point = roc_auc_score(y, p_b) - roc_auc_score(y, p_a)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_val = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta_auc": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(min(1.0, p_val))}


def permutation_null_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logistic",
    n_perm: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """AUC under label permutation *within* subject-grouped structure (labels shuffled across subjects)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq, first_idx = np.unique(groups, return_index=True)
    subj_label = y[first_idx]
    aucs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(subj_label)
        lab_map = dict(zip(uniq, perm))
        y_perm = np.array([lab_map[g] for g in groups])
        res = nested_cv(X, y_perm, groups, kind, n_repeats=1, seed=seed + i)
        aucs[i] = res.auc()
    return aucs


def covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Coordinator-computable covariates (age, sex, APOE ε4, education, MMSE, CDR-SB)."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"]
    out["sex_male"] = (df["sex"].astype(str).str.upper().str[0] == "M").astype(float)
    out["apoe_e4"] = df["apoe_e4"]
    out["education"] = df["education"]
    out["mmse"] = df["mmse"]
    out["cdr_sb"] = df["cdr_sb"]
    return out.fillna(out.median(numeric_only=True))


def summarize(results: list[CVResult], baseline: str, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Table of AUC/Brier/calibration for each model plus ΔAUC vs the named baseline."""
    base = next(r for r in results if r.name == baseline)
    rows = []
    for r in results:
        a, b = platt_scale(r.mean_prob(), r.y)
        row = {"model": r.name, "auc": r.auc(), "brier": r.brier(), "cal_intercept": a, "cal_slope": b}
        if r.name != baseline:
            row.update({f"vs_{baseline}_{k}": v for k, v in
                        bootstrap_delta_auc(base.mean_prob(), r.mean_prob(), r.y, r.groups, n_boot, seed).items()})
        rows.append(row)
    return pd.DataFrame(rows)
