"""Error-structure analyses: confusion matrices by stratum, kappa, N3% bias, event-locked and differential error.

These functions operate on aligned per-epoch arrays (``y_true``, ``y_pred``) plus per-epoch or per-record
covariates; the record identifier is used for cluster-robust inference.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .hypnogram import N3, STAGE_NAMES, UNSCORED, W, n3_percent

CLASSES = (0, 1, 2, 3, 4)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> np.ndarray:
    """5x5 confusion matrix (rows = true stage, cols = predicted); unscored epochs are dropped."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    keep = (yt != UNSCORED) & (yp != UNSCORED)
    cm = np.zeros((5, 5), dtype=float)
    np.add.at(cm, (yt[keep], yp[keep]), 1.0)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    return cm


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa over scored epochs."""
    cm = confusion_matrix(y_true, y_pred, normalize=False)
    n = cm.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-stage sensitivity (recall), PPV (precision) and F1, plus overall accuracy and kappa."""
    cm = confusion_matrix(y_true, y_pred, normalize=False)
    rows = []
    for k in CLASSES:
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) and not np.isnan(sens + ppv) else np.nan
        rows.append({"stage": STAGE_NAMES[k], "n_true": int(tp + fn), "sensitivity": sens, "ppv": ppv, "f1": f1})
    df = pd.DataFrame(rows)
    df.attrs["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() else np.nan
    df.attrs["kappa"] = cohens_kappa(y_true, y_pred)
    return df


def confusion_by_stratum(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray) -> Dict[object, np.ndarray]:
    """Row-normalised confusion matrix for each unique value of ``strata`` (per-epoch array)."""
    strata = np.asarray(strata)
    return {s: confusion_matrix(y_true[strata == s], y_pred[strata == s]) for s in pd.unique(strata)}


def ahi_stratum(ahi: np.ndarray, edges: Sequence[float] = (5.0, 15.0, 30.0)) -> np.ndarray:
    """Standard OSA severity strata: 0 (<5), 1 (5-15), 2 (15-30), 3 (>=30)."""
    return np.digitize(np.asarray(ahi, dtype=float), np.asarray(edges, dtype=float))


def n3_bias_per_record(y_true: np.ndarray, y_pred: np.ndarray, record: np.ndarray,
                       posteriors: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per-record N3% (human, argmax-model, optional expected-model) and their differences."""
    rows = []
    record = np.asarray(record)
    for rid in pd.unique(record):
        m = record == rid
        row = {"record": rid, "n3_true": n3_percent(y_true[m]), "n3_pred": n3_percent(y_pred[m])}
        if posteriors is not None:
            p = posteriors[m]
            p_sleep = 1.0 - p[:, W]
            row["n3_expected"] = 100.0 * float(p[:, N3].sum() / max(p_sleep.sum(), 1e-9))
        row["diff"] = row["n3_pred"] - row["n3_true"]
        rows.append(row)
    return pd.DataFrame(rows)


def bland_altman(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Bias and 95% limits of agreement of b - a."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[~np.isnan(d)]
    return {"bias": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)) if len(d) > 1 else np.nan,
            "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)) if len(d) > 1 else np.nan, "n": int(len(d))}


def event_locked_error(y_true: np.ndarray, y_pred: np.ndarray, event_mask: np.ndarray,
                       restrict_to: Optional[Iterable[int]] = (1, 2, 3)) -> Dict[str, float]:
    """Error rate inside vs outside event-adjacent epochs and the odds ratio with a Wald 95% CI.

    ``restrict_to`` limits the comparison to epochs whose true stage is in the set (default: NREM),
    so that the comparison is not driven by Wake epochs, which are more common around events.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    em = np.asarray(event_mask, dtype=bool)
    keep = yt != UNSCORED
    if restrict_to is not None:
        keep &= np.isin(yt, list(restrict_to))
    err = (yt != yp)[keep]
    em = em[keep]
    a = float((err & em).sum()) + 0.5      # errors, in events (Haldane-Anscombe correction)
    b = float((~err & em).sum()) + 0.5
    c = float((err & ~em).sum()) + 0.5
    d = float((~err & ~em).sum()) + 0.5
    or_ = (a * b ** -1) / (c * d ** -1)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"error_rate_in_events": (a - 0.5) / max(a + b - 1.0, 1.0),
            "error_rate_outside": (c - 0.5) / max(c + d - 1.0, 1.0),
            "odds_ratio": float(or_), "ci_low": float(np.exp(np.log(or_) - 1.96 * se)),
            "ci_high": float(np.exp(np.log(or_) + 1.96 * se)), "n_event_epochs": int(a + b - 1.0)}


def differential_misclassification_test(df: pd.DataFrame, covariates: Sequence[str], true_stage: int = N3,
                                        cluster_col: str = "record") -> pd.DataFrame:
    """Cluster-robust logistic regression of misclassification on covariates among epochs of one true stage.

    ``df`` needs columns ``y_true``, ``y_pred``, ``cluster_col`` and the covariates (per-epoch or broadcast
    per-record values such as AHI, age, prevalent heart failure, outcome status). A significant coefficient for
    an outcome or a strong outcome cause, conditional on AHI, indicates *differential* misclassification.
    """
    d = df[df["y_true"] == true_stage].copy()
    d["misclassified"] = (d["y_true"] != d["y_pred"]).astype(int)
    X = sm.add_constant(d[list(covariates)].astype(float), has_constant="add")
    model = sm.Logit(d["misclassified"], X)
    res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster_col])[0]})
    out = pd.DataFrame({"coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues})
    out["odds_ratio"] = np.exp(out["coef"])
    out.attrs["n_epochs"] = int(len(d))
    out.attrs["n_clusters"] = int(d[cluster_col].nunique())
    return out


def bootstrap_records(stat_fn, y_true: np.ndarray, y_pred: np.ndarray, record: np.ndarray,
                      n_boot: int = 1000, rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Subject-level bootstrap of any scalar statistic stat_fn(y_true, y_pred): (point, ci_low, ci_high)."""
    rng = np.random.default_rng(0) if rng is None else rng
    record = np.asarray(record)
    ids = pd.unique(record)
    idx = {r: np.flatnonzero(record == r) for r in ids}
    point = stat_fn(y_true, y_pred)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx[r] for r in pick])
        vals[b] = stat_fn(y_true[sel], y_pred[sel])
    return float(point), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
