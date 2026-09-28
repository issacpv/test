"""Metrics shared by every modality, with group-level bootstrap.

Discrimination (AUROC, AUPRC), proper scores (Brier, log-loss), calibration
(intercept, slope, ECE), subgroup gaps, regression metrics with uncertainty
coverage, and any-overlap event scoring for seizure-style tasks.  All binary
metrics accept ``sample_weight`` so the gap decomposition in
:mod:`bioshift.diagnostics` can reuse them.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #
def auroc(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def auprc(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s, sample_weight=w))


def brier(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    return float(np.average((np.asarray(s) - np.asarray(y)) ** 2, weights=w))


def log_loss(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.average(y * np.log(s) + (1 - y) * np.log(1 - s), weights=w))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None, n_iter: int = 50) -> Tuple[float, float]:
    """Logistic recalibration ``y ~ a + b * logit(s)`` fitted by weighted Newton iterations.

    Returns ``(intercept_in_the_large, slope)`` where the intercept is estimated
    with slope fixed at 1 (calibration-in-the-large) and the slope from the
    two-parameter fit.  Perfect calibration gives (0, 1).
    """
    y = np.asarray(y, float)
    x = _logit(s)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    def newton(design: np.ndarray, offset: np.ndarray) -> np.ndarray:
        beta = np.zeros(design.shape[1])
        for _ in range(n_iter):
            eta = offset + design @ beta
            p = 1 / (1 + np.exp(-eta))
            g = design.T @ (w * (y - p))
            H = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-8 * np.eye(design.shape[1])
            step = np.linalg.solve(H, g)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        return beta

    a_only = newton(np.ones((len(y), 1)), x)[0]
    ab = newton(np.c_[np.ones(len(y)), x], np.zeros(len(y)))
    return float(a_only), float(ab[1])


def ece(y: np.ndarray, s: np.ndarray, n_bins: int = 10, quantile: bool = True, w: Optional[np.ndarray] = None) -> float:
    """Expected calibration error with quantile (default) or equal-width bins."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    if quantile:
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return 0.0
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    total = w.sum()
    out = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        out += wb / total * abs(np.average(y[m], weights=w[m]) - np.average(s[m], weights=w[m]))
    return float(out)


def binary_metrics(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    a, b = calibration_intercept_slope(y, s, w)
    return {
        "auroc": auroc(y, s, w),
        "auprc": auprc(y, s, w),
        "brier": brier(y, s, w),
        "log_loss": log_loss(y, s, w),
        "cal_intercept": a,
        "cal_slope": b,
        "ece": ece(y, s, w=w),
        "prevalence": float(np.average(y, weights=w)),
    }


# --------------------------------------------------------------------------- #
# multilabel / regression / events
# --------------------------------------------------------------------------- #
def multilabel_metrics(Y: np.ndarray, S: np.ndarray) -> Dict[str, float]:
    Y, S = np.asarray(Y), np.asarray(S)
    per = [auroc(Y[:, k], S[:, k]) for k in range(Y.shape[1])]
    per_ap = [auprc(Y[:, k], S[:, k]) for k in range(Y.shape[1])]
    cal = [calibration_intercept_slope(Y[:, k], S[:, k]) for k in range(Y.shape[1]) if len(np.unique(Y[:, k])) > 1]
    out = {
        "macro_auroc": float(np.nanmean(per)),
        "macro_auprc": float(np.nanmean(per_ap)),
        "cal_intercept": float(np.nanmean([c[0] for c in cal])) if cal else float("nan"),
        "cal_slope": float(np.nanmean([c[1] for c in cal])) if cal else float("nan"),
        "ece": float(np.mean([ece(Y[:, k], S[:, k]) for k in range(Y.shape[1])])),
    }
    for k, v in enumerate(per):
        out[f"auroc_class{k}"] = v
    return out


def regression_metrics(y: np.ndarray, pred: np.ndarray, pred_std: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    ss_res = float(np.average(err**2, weights=w))
    ss_tot = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    out = {
        "mae": float(np.average(np.abs(err), weights=w)),
        "rmse": float(np.sqrt(ss_res)),
        "r2": float(1 - ss_res / max(ss_tot, EPS)),
        "bias": float(np.average(err, weights=w)),  # calibration-in-the-large for regression
        "slope": float(np.polyfit(pred, y, 1)[0]) if np.std(pred) > 0 else float("nan"),
    }
    if pred_std is not None:
        z = np.abs(err) / np.maximum(np.asarray(pred_std, float), EPS)
        out["coverage90"] = float(np.average(z <= 1.645, weights=w))
        out["coverage50"] = float(np.average(z <= 0.674, weights=w))
    return out


def windows_to_events(labels: np.ndarray, t_start: np.ndarray, record_id: np.ndarray, window_s: float = 4.0) -> List[Tuple[object, float, float]]:
    """Merge contiguous positive windows into ``(record, onset, offset)`` events."""
    events: List[Tuple[object, float, float]] = []
    labels = np.asarray(labels).astype(int)
    for r in pd.unique(record_id):
        idx = np.flatnonzero(record_id == r)
        idx = idx[np.argsort(t_start[idx], kind="stable")]
        cur: Optional[List[float]] = None
        for i in idx:
            if labels[i]:
                if cur is None:
                    cur = [float(t_start[i]), float(t_start[i] + window_s)]
                elif t_start[i] <= cur[1] + 1e-9:
                    cur[1] = float(t_start[i] + window_s)
                else:
                    events.append((r, cur[0], cur[1]))
                    cur = [float(t_start[i]), float(t_start[i] + window_s)]
            elif cur is not None:
                events.append((r, cur[0], cur[1]))
                cur = None
        if cur is not None:
            events.append((r, cur[0], cur[1]))
    return events


def event_scores(ref: Sequence[Tuple[object, float, float]], hyp: Sequence[Tuple[object, float, float]], tol_pre: float = 30.0, tol_post: float = 60.0, total_hours: Optional[float] = None) -> Dict[str, float]:
    """Any-overlap event scoring with pre/post tolerances (simplified SzCORE-style).

    A reference event is detected if any hypothesis overlaps ``[onset - tol_pre,
    offset + tol_post]``; a hypothesis is a false alarm if it overlaps no
    reference (with the same tolerance).  ``total_hours`` gives FP/24 h.
    """
    ref_hit = np.zeros(len(ref), bool)
    hyp_hit = np.zeros(len(hyp), bool)
    for i, (r, on, off) in enumerate(ref):
        lo, hi = on - tol_pre, off + tol_post
        for j, (hr, hon, hoff) in enumerate(hyp):
            if hr == r and hon < hi and hoff > lo:
                ref_hit[i] = True
                hyp_hit[j] = True
    sens = float(ref_hit.mean()) if len(ref) else float("nan")
    prec = float(hyp_hit.mean()) if len(hyp) else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if (len(ref) and len(hyp) and (sens + prec) > 0) else 0.0
    out = {"event_sensitivity": sens, "event_precision": prec, "event_f1": float(f1), "n_ref": float(len(ref)), "n_hyp": float(len(hyp))}
    if total_hours:
        out["fp_per_24h"] = float((~hyp_hit).sum() / total_hours * 24.0)
    return out


def event_metrics(y: np.ndarray, s: np.ndarray, meta: pd.DataFrame, threshold: float = 0.5, window_s: float = 4.0, step_s: float = 2.0) -> Dict[str, float]:
    """Window AUROC plus event-level scores from thresholded window scores."""
    out = {"window_auroc": auroc(y, s)}
    if "record_id" not in meta.columns or "t_start" not in meta.columns:
        return out
    rec, t0 = meta["record_id"].to_numpy(), meta["t_start"].to_numpy(float)
    ref = windows_to_events(y, t0, rec, window_s)
    hyp = windows_to_events((np.asarray(s) >= threshold).astype(int), t0, rec, window_s)
    total_hours = float(len(y) * step_s / 3600.0)
    out.update(event_scores(ref, hyp, total_hours=total_hours))
    return out


# --------------------------------------------------------------------------- #
# subgroups and bootstrap
# --------------------------------------------------------------------------- #
def subgroup_gaps(y: np.ndarray, s: np.ndarray, meta: pd.DataFrame, columns: Iterable[str], metric: Callable[[np.ndarray, np.ndarray], float] = auroc, min_n: int = 30) -> Dict[str, float]:
    """max - min of ``metric`` over levels of each column (levels with < ``min_n`` rows or one class skipped)."""
    out: Dict[str, float] = {}
    for col in columns:
        if col not in meta.columns:
            continue
        vals = []
        for lvl, idx in meta.groupby(col).indices.items():
            if len(idx) < min_n or len(np.unique(np.asarray(y)[idx])) < 2:
                continue
            v = metric(np.asarray(y)[idx], np.asarray(s)[idx])
            if np.isfinite(v):
                vals.append(v)
                out[f"{col}={lvl}"] = float(v)
        out[f"gap_{col}"] = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    return out


def group_bootstrap(fn: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of ``fn(index_array)`` under resampling of whole groups."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def evaluate(task_type: str, y: np.ndarray, s: np.ndarray, groups: Optional[np.ndarray] = None, meta: Optional[pd.DataFrame] = None, subgroup_columns: Sequence[str] = ("sex", "age_band"), n_boot: int = 0, seed: int = 0, pred_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Dispatch on task type; adds ``<metric>_lo/_hi`` for the primary metric if ``n_boot > 0``."""
    task_type = getattr(task_type, "value", task_type)
    meta = meta if meta is not None else pd.DataFrame(index=range(len(y)))
    if task_type == "binary":
        out = binary_metrics(y, s)
        out.update(subgroup_gaps(y, s, meta, subgroup_columns))
        primary = "auroc"
        fn = lambda idx: auroc(np.asarray(y)[idx], np.asarray(s)[idx])  # noqa: E731
    elif task_type == "multilabel":
        out = multilabel_metrics(y, s)
        primary = "macro_auroc"
        fn = lambda idx: multilabel_metrics(np.asarray(y)[idx], np.asarray(s)[idx])["macro_auroc"]  # noqa: E731
    elif task_type == "regression":
        out = regression_metrics(y, s, pred_std)
        out.update(subgroup_gaps(y, s, meta, subgroup_columns, metric=lambda a, b: regression_metrics(a, b)["mae"]))
        primary = "mae"
        fn = lambda idx: regression_metrics(np.asarray(y)[idx], np.asarray(s)[idx])["mae"]  # noqa: E731
    elif task_type == "event_detection":
        out = event_metrics(y, s, meta)
        out.update(binary_metrics(y, s))
        primary = "event_f1" if "event_f1" in out else "window_auroc"
        fn = lambda idx: event_metrics(np.asarray(y)[idx], np.asarray(s)[idx], meta.iloc[idx].reset_index(drop=True)).get("event_f1", float("nan"))  # noqa: E731
    else:
        raise ValueError(task_type)
    if n_boot > 0 and groups is not None:
        lo, hi = group_bootstrap(fn, np.asarray(groups), n_boot, seed)
        out[f"{primary}_lo"], out[f"{primary}_hi"] = lo, hi
    out["primary"] = out[primary]
    out["n"] = float(len(y))
    return out
