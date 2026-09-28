"""Discrimination, calibration, clinical utility and subgroup metrics with bootstrap CIs."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auroc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def auprc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(average_precision_score(y, p, sample_weight=w))


def logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=w, labels=[0, 1]))


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(brier_score_loss(y, p, sample_weight=w))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of y on logit(p): returns (intercept at slope fixed to 1, slope)."""
    z = _logit(p).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return np.nan, np.nan
    slope = float(LogisticRegression(C=1e6).fit(z, y).coef_[0, 0])
    # calibration-in-the-large: intercept with offset = logit(p) (slope fixed at 1)
    lr = LogisticRegression(C=1e6, fit_intercept=True)
    lr.fit(np.zeros_like(z), y)  # placeholder to get shapes; closed form below is exact for offset model
    # Newton iterations for intercept a in  y ~ expit(z + a)
    a = 0.0
    zf = z.ravel()
    for _ in range(50):
        q = 1.0 / (1.0 + np.exp(-(zf + a)))
        g = np.sum(y - q)
        h = -np.sum(q * (1 - q))
        step = g / h if h != 0 else 0.0
        a -= step
        if abs(step) < 1e-8:
            break
    return float(a), slope


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with quantile bins."""
    y, p = np.asarray(y).astype(float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tot = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)) -> pd.DataFrame:
    """Decision-curve net benefit of the model vs treat-all at each threshold (Vickers & Elkin, 2006)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    n = len(y)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        nb = tp / n - fp / n * (t / (1 - t))
        prev = y.mean()
        nb_all = prev - (1 - prev) * (t / (1 - t))
        rows.append({"threshold": t, "net_benefit": nb, "net_benefit_all": nb_all})
    return pd.DataFrame(rows)


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    a, b = calibration_intercept_slope(y, p)
    return {"auroc": auroc(y, p), "brier": brier(y, p), "logloss": logloss(y, p), "cal_intercept": a, "cal_slope": b, "ece": ece(y, p)}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 1000, seed: int = 0, groups: np.ndarray | None = None) -> dict[str, float]:
    """Bootstrap CI of a scalar metric; cluster bootstrap when ``groups`` (e.g. subject ids) given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            vals.append(fn(y[i], p[i]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            draw = rng.choice(uniq, len(uniq), replace=True)
            i = np.concatenate([members[g] for g in draw])
            vals.append(fn(y[i], p[i]))
    vals = np.asarray(vals, float)
    return {"estimate": float(fn(y, p)), "ci_low": float(np.nanquantile(vals, 0.025)), "ci_high": float(np.nanquantile(vals, 0.975))}


def subgroup_table(y: np.ndarray, p: np.ndarray, group: np.ndarray, alert_rate: float = 0.05, min_n: int = 50) -> pd.DataFrame:
    """Per-group AUROC, calibration intercept/slope, ECE and TPR/FPR at a fixed alert-rate threshold, plus max-min gaps."""
    y, p, group = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(group)
    thr = np.quantile(p, 1 - alert_rate)
    rows = []
    for g in np.unique(group):
        m = group == g
        if m.sum() < min_n or y[m].min() == y[m].max():
            continue
        a, b = calibration_intercept_slope(y[m], p[m])
        pred = p[m] >= thr
        tpr = np.mean(pred[y[m] == 1]) if (y[m] == 1).any() else np.nan
        fpr = np.mean(pred[y[m] == 0]) if (y[m] == 0).any() else np.nan
        rows.append({"group": g, "n": int(m.sum()), "prevalence": float(y[m].mean()), "auroc": auroc(y[m], p[m]), "cal_intercept": a, "cal_slope": b, "ece": ece(y[m], p[m]), "tpr": tpr, "fpr": fpr})
    out = pd.DataFrame(rows)
    if len(out):
        for c in ["auroc", "cal_intercept", "cal_slope", "ece", "tpr", "fpr"]:
            out.attrs[f"gap_{c}"] = float(out[c].max() - out[c].min())
    return out
