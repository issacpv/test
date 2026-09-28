"""Baseline and ROI models with nested, subject-grouped cross-validation and calibration.

Design principles (these are the audit's methodological claims, so they are enforced in code):

* every split is grouped by subject (``StratifiedGroupKFold``), so repeated sessions never leak;
* hyper-parameters are chosen in an inner grouped CV; Platt scaling is fitted on inner out-of-fold
  logits and applied to the outer test fold;
* optional ComBat harmonization is fitted on the training fold only;
* metrics come with subject-level cluster-bootstrap confidence intervals.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .roi_features import ComBat

DEMOGRAPHIC_FEATURES: tuple[str, ...] = ("age", "sex_male", "apoe_e4_count", "mmse")

_EPS = 1e-6


def make_estimator(kind: str) -> tuple[object, dict]:
    """Return (estimator, hyper-parameter grid) for ``kind`` in
    {'demographic', 'roi_enet', 'gbm'}."""
    if kind == "demographic":
        est = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000))])
        grid = {"clf__C": [0.1, 1.0, 10.0]}
    elif kind == "roi_enet":
        est = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                                   l1_ratio=0.5))])
        grid = {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    elif kind == "gbm":
        est = HistGradientBoostingClassifier(random_state=0)
        grid = {"max_depth": [2, 3], "learning_rate": [0.05, 0.1], "max_iter": [100, 300]}
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return est, grid


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, float), _EPS, 1 - _EPS))


def platt_fit(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``P(y=1) = sigmoid(a * logit + b)`` (Platt, 1999). Returns (a, b)."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(logits, float).reshape(-1, 1), np.asarray(y, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return expit(a * _safe_logit(p_raw) + b)


def nested_cv_predict(X: np.ndarray | pd.DataFrame, y: np.ndarray, groups: Sequence,
                      kind: str = "roi_enet", outer_splits: int = 5, inner_splits: int = 3,
                      n_repeats: int = 1, random_state: int = 0, calibrate: bool = True,
                      batch: Optional[Sequence] = None, combat_covars: Optional[np.ndarray] = None,
                      n_jobs: Optional[int] = None) -> pd.DataFrame:
    """Nested subject-grouped CV returning out-of-fold predictions.

    Parameters
    ----------
    X, y, groups
        Features (n×p), binary labels, subject ids (CV groups).
    batch
        Optional scanner/site labels; when given, :class:`ComBat` is fitted on the training fold
        (with ``combat_covars`` preserved) and applied to the test fold.

    Returns
    -------
    DataFrame with columns repeat, fold, row, y, p_raw, p_cal (``p_cal == p_raw`` if not calibrated).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    batch = None if batch is None else np.asarray(batch)
    C = None if combat_covars is None else np.asarray(combat_covars, float)
    est, grid = make_estimator(kind)
    records = []
    for rep in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=random_state + rep)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            Xtr, Xte = X[tr], X[te]
            if batch is None:
                pass
            else:
                cb = ComBat().fit(Xtr, batch[tr], None if C is None else C[tr])
                Xtr = cb.transform(Xtr, batch[tr], None if C is None else C[tr])
                Xte = cb.transform(Xte, batch[te], None if C is None else C[te])
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + rep)
            inner_splits_list = list(inner.split(Xtr, y[tr], groups[tr]))
            gs = GridSearchCV(clone(est), grid, cv=inner_splits_list, scoring="roc_auc", n_jobs=n_jobs)
            gs.fit(Xtr, y[tr])
            best = gs.best_estimator_
            p_raw = best.predict_proba(Xte)[:, 1]
            if calibrate:
                p_inner = cross_val_predict(clone(best), Xtr, y[tr], cv=inner_splits_list,
                                            method="predict_proba")[:, 1]
                a, b = platt_fit(_safe_logit(p_inner), y[tr])
                p_cal = platt_apply(p_raw, a, b)
            else:
                p_cal = p_raw
            records.append(pd.DataFrame({"repeat": rep, "fold": fold, "row": te, "y": y[te],
                                         "p_raw": p_raw, "p_cal": p_cal}))
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of y on logit(p): slope 1 / intercept 0 means perfect calibration."""
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan, np.nan
    return platt_fit(_safe_logit(p), y)


def _cluster_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    draw = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[i] for i in draw])


def evaluate(y: np.ndarray, p: np.ndarray, groups: Optional[Sequence] = None, n_boot: int = 1000,
             random_state: int = 0) -> dict[str, float]:
    """AUC, Brier, calibration slope/intercept with subject-level bootstrap CIs for AUC and Brier."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    out = {"n": int(len(y)), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p))}
    out["cal_slope"], out["cal_intercept"] = calibration_slope_intercept(y, p)
    rng = np.random.default_rng(random_state)
    aucs, briers = [], []
    for _ in range(n_boot):
        idx = _cluster_bootstrap_indices(groups, rng)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        briers.append(brier_score_loss(y[idx], p[idx]))
    if aucs:
        out["auc_ci"] = tuple(np.percentile(aucs, [2.5, 97.5]).round(4))
        out["brier_ci"] = tuple(np.percentile(briers, [2.5, 97.5]).round(4))
    return out


def paired_auc_difference(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                          groups: Optional[Sequence] = None, n_boot: int = 2000,
                          random_state: int = 0) -> dict[str, float]:
    """ΔAUC = AUC(new) − AUC(ref) with cluster-bootstrap CI and a two-sided bootstrap p-value.

    This is the primary test for "does MRI beat age + APOE + MMSE?".
    """
    y = np.asarray(y, int)
    p_new, p_ref = np.asarray(p_new, float), np.asarray(p_ref, float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    delta = roc_auc_score(y, p_new) - roc_auc_score(y, p_ref)
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(n_boot):
        idx = _cluster_bootstrap_indices(groups, rng)
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta_auc": float(delta), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p_value)), "n_boot": int(len(deltas))}


def stratified_auc(y: np.ndarray, p: np.ndarray, strata: Sequence, groups: Optional[Sequence] = None,
                   n_boot: int = 500) -> pd.DataFrame:
    """AUC (with CI) per stratum, e.g. cognitive status or age tertile (the proxy test)."""
    y, p, strata = np.asarray(y, int), np.asarray(p, float), np.asarray(strata)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    rows = []
    for s in pd.unique(strata):
        m = strata == s
        if m.sum() < 10 or y[m].min() == y[m].max():
            rows.append({"stratum": s, "n": int(m.sum()), "auc": np.nan})
            continue
        r = evaluate(y[m], p[m], groups[m], n_boot=n_boot)
        rows.append({"stratum": s, "n": r["n"], "prevalence": r["prevalence"], "auc": r["auc"],
                     "auc_ci": r.get("auc_ci")})
    return pd.DataFrame(rows)


def age_matched_subset(age: np.ndarray, y: np.ndarray, caliper: float = 2.0,
                       random_state: int = 0) -> np.ndarray:
    """Greedy 1:1 caliper matching of positives to negatives on age; returns row indices.

    Used for the age-proxy test: AUC on this subset cannot come from age differences.
    """
    age, y = np.asarray(age, float), np.asarray(y, int)
    rng = np.random.default_rng(random_state)
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = list(np.flatnonzero(y == 0))
    keep = []
    for i in pos:
        if not neg:
            break
        d = np.abs(age[neg] - age[i])
        j = int(np.argmin(d))
        if d[j] <= caliper:
            keep += [i, neg.pop(j)]
    return np.asarray(sorted(keep), int)
