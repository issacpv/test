"""Clock-feature ablation: how much benchmark performance is the clock?"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLOCK_COLUMNS = ["hour_sin", "hour_cos", "admit_hour_sin", "admit_hour_cos", "is_weekend", "is_night", "hours_since_admit"]


def clock_features(prediction_time: pd.Series, admit_time: pd.Series, night: tuple = (19.0, 7.0)) -> pd.DataFrame:
    """Clock features at prediction time: hour (sin/cos), admission hour (sin/cos), weekend, night, hours since admission."""
    pt = pd.to_datetime(prediction_time)
    at = pd.to_datetime(admit_time)
    h = pt.dt.hour + pt.dt.minute / 60.0
    ah = at.dt.hour + at.dt.minute / 60.0
    start, end = night
    is_night = (h >= start) | (h < end) if start > end else (h >= start) & (h < end)
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
        "admit_hour_sin": np.sin(2 * np.pi * ah / 24), "admit_hour_cos": np.cos(2 * np.pi * ah / 24),
        "is_weekend": (pt.dt.dayofweek >= 5).astype(float), "is_night": is_night.astype(float),
        "hours_since_admit": (pt - at).dt.total_seconds() / 3600.0,
    }, index=prediction_time.index)


def default_model() -> Callable[[], object]:
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan, "n": int(len(y))}
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    lr0 = LogisticRegression(C=1e6, max_iter=1000, fit_intercept=True)
    # calibration-in-the-large: intercept with slope fixed at 1 == mean(y) - mean(p) on the logit scale
    lr0.fit(np.zeros((len(y), 1)), y, sample_weight=None)
    citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "cal_intercept": citl, "cal_slope": float(lr.coef_[0, 0]), "n": int(len(y))}


@dataclass
class AblationResult:
    overall: pd.DataFrame          # rows: feature set; columns: metrics
    by_hour: pd.DataFrame          # rows: (feature set, hour bin)
    predictions: Dict[str, np.ndarray]
    clock_share: float             # AUROC(clock only) - 0.5
    clock_increment: float         # AUROC(full) - AUROC(no clock)


def clock_ablation(
    X_phys: pd.DataFrame, X_clock: pd.DataFrame, y: pd.Series, groups: Sequence, hour: pd.Series,
    model_factory: Optional[Callable[[], object]] = None, n_splits: int = 5, hour_bins: int = 6,
) -> AblationResult:
    """Grouped cross-validated comparison of full / no-clock / clock-only feature sets with hour-stratified metrics."""
    mf = model_factory or default_model()
    sets = {"physiology_only": X_phys, "physiology_plus_clock": pd.concat([X_phys, X_clock], axis=1), "clock_only": X_clock}
    ya = y.to_numpy(int)
    g = np.asarray(groups)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(g))))
    preds: Dict[str, np.ndarray] = {}
    for name, X in sets.items():
        Xa = X.fillna(X.median()).to_numpy(float)
        p = np.zeros(len(ya))
        for tr, te in gkf.split(Xa, ya, g):
            m = mf().fit(Xa[tr], ya[tr])
            p[te] = m.predict_proba(Xa[te])[:, 1]
        preds[name] = p
    overall = pd.DataFrame({k: _metrics(ya, v) for k, v in preds.items()}).T
    hb = (np.asarray(hour, float) % 24 // (24 / hour_bins)).astype(int)
    rows = []
    for name, p in preds.items():
        for b in range(hour_bins):
            m = hb == b
            if m.sum() == 0:
                continue
            r = _metrics(ya[m], p[m])
            r.update({"feature_set": name, "hour_bin": b, "hour_start": b * 24 / hour_bins})
            rows.append(r)
    by_hour = pd.DataFrame(rows)
    return AblationResult(overall=overall, by_hour=by_hour, predictions=preds,
                          clock_share=float(overall.loc["clock_only", "auroc"] - 0.5),
                          clock_increment=float(overall.loc["physiology_plus_clock", "auroc"] - overall.loc["physiology_only", "auroc"]))


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 0,
                           groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Bootstrap CI for AUROC(a) - AUROC(b), resampling stays (groups) if given."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    deltas = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_g = {u: np.nonzero(groups == u)[0] for u in uniq}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([idx_by_g[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.array(deltas)
    return {"delta": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5))}


def clock_probe(X_phys: pd.DataFrame, hour: pd.Series, groups: Sequence, n_splits: int = 5) -> float:
    """How well can the clock hour be recovered from the physiology features alone? (R^2 of sin/cos ridge probe.)

    A high value means the "no-clock" ablation still leaks the clock through measurement patterns.
    """
    from sklearn.linear_model import RidgeCV

    Xa = X_phys.fillna(X_phys.median()).to_numpy(float)
    h = np.asarray(hour, float) % 24
    Y = np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])
    g = np.asarray(groups)
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(g)))).split(Xa, Y, g):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 10))).fit(Xa[tr], Y[tr])
        pred[te] = m.predict(Xa[te])
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2)
    return float(1 - ss_res / ss_tot)
