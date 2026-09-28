"""Uncertainty decomposition, calibration, Mondrian conformal prediction and subgroup audits.

All functions are model-agnostic: they take probability arrays produced by
an ensemble (``[M, N, K]``) or a single model (``[N, K]``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def ensemble_decomposition(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose ensemble predictive uncertainty (Depeweg et al., 2018 style).

    Parameters
    ----------
    probs : ``[M, N, K]`` member probabilities.

    Returns
    -------
    dict with ``mean`` [N, K], ``total`` (predictive entropy), ``aleatoric``
    (expected member entropy) and ``epistemic`` (mutual information) [N].
    """
    p = np.asarray(probs, float)
    if p.ndim != 3:
        raise ValueError("probs must be [M, N, K]")
    mean = p.mean(0)
    total = _entropy(mean)
    aleatoric = _entropy(p).mean(0)
    return {"mean": mean, "total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE for multiclass ``[N, K]`` probabilities or ``[N]`` binary probabilities."""
    p = np.asarray(probs, float)
    y = np.asarray(y, int)
    if p.ndim == 1:
        conf = np.where(p >= 0.5, p, 1 - p)
        pred = (p >= 0.5).astype(int)
    else:
        conf = p.max(1)
        pred = p.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(probs, float)
    y = np.asarray(y, int)
    if p.ndim == 1:
        return float(np.mean((p - y) ** 2))
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def mondrian_conformal_thresholds(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Class-conditional (Mondrian) split-conformal thresholds on the nonconformity ``1 - p_y``.

    Returns one threshold per class; a class ``k`` is included in the
    prediction set for a new sample when ``1 - p_k <= threshold[k]``.
    """
    p = np.asarray(cal_probs, float)
    y = np.asarray(cal_y, int)
    k_classes = p.shape[1]
    thr = np.ones(k_classes)
    for k in range(k_classes):
        scores = 1 - p[y == k, k]
        n = len(scores)
        if n == 0:
            continue
        q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        thr[k] = float(np.quantile(scores, q, method="higher"))
    return thr


def conformal_prediction_sets(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean ``[N, K]`` prediction sets from probabilities and Mondrian thresholds."""
    p = np.asarray(probs, float)
    return (1 - p) <= np.asarray(thresholds)[None, :]


def coverage_and_size(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    s = np.asarray(sets, bool)
    y = np.asarray(y, int)
    covered = s[np.arange(len(y)), y]
    return {"coverage": float(covered.mean()), "mean_size": float(s.sum(1).mean()), "frac_empty": float((s.sum(1) == 0).mean())}


def uncertainty_vs_disagreement_auroc(uncertainty: np.ndarray, disagree: np.ndarray) -> float:
    """AUROC of an uncertainty score for predicting reader disagreement (H2)."""
    d = np.asarray(disagree, int)
    if d.sum() in (0, len(d)):
        return float("nan")
    return float(roc_auc_score(d, np.asarray(uncertainty, float)))


def subgroup_metrics(probs: np.ndarray, y: np.ndarray, groups: np.ndarray, sets: np.ndarray | None = None, min_n: int = 20) -> pd.DataFrame:
    """Per-subgroup AUROC (binary from top class), ECE, Brier and conformal coverage; plus max gaps in the last row."""
    p = np.asarray(probs, float)
    y = np.asarray(y, int)
    g = np.asarray(groups)
    rows = []
    for name in np.unique(g):
        m = g == name
        if m.sum() < min_n:
            continue
        row = {"group": str(name), "n": int(m.sum()), "ece": expected_calibration_error(p[m], y[m]), "brier": brier_score(p[m], y[m])}
        if p.ndim == 2 and p.shape[1] == 2:
            row["auroc"] = float(roc_auc_score(y[m], p[m, 1])) if len(np.unique(y[m])) == 2 else np.nan
        if sets is not None:
            row["coverage"] = coverage_and_size(np.asarray(sets)[m], y[m])["coverage"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        gaps = {"group": "MAX_GAP", "n": int(df["n"].sum())}
        for col in df.columns:
            if col not in ("group", "n"):
                gaps[col] = float(df[col].max() - df[col].min())
        df = pd.concat([df, pd.DataFrame([gaps])], ignore_index=True)
    return df
