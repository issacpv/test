"""Evaluation of dynamic predictions: per-landmark discrimination and calibration, bootstrap, treatment-aware summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = float(sm.Logit(y, sm.add_constant(z)).fit(disp=False).params[1])
    intercept = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0])
    return intercept, slope


def metrics_by_landmark(df: pd.DataFrame, p: np.ndarray, min_events: int = 5) -> pd.DataFrame:
    """AUROC, AUPRC, event count, calibration intercept/slope per landmark day (and pooled)."""
    d = df.assign(_p=np.asarray(p, float))
    rows = []
    for day, g in list(d.groupby("landmark_day")) + [("pooled", d)]:
        y = g["outcome"].to_numpy(int)
        if y.sum() < min_events or y.sum() == len(y):
            rows.append({"landmark_day": day, "n": len(g), "events": int(y.sum()), "auroc": np.nan, "auprc": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan})
            continue
        a, b = calibration_intercept_slope(y, g["_p"].to_numpy())
        rows.append({"landmark_day": day, "n": len(g), "events": int(y.sum()), "auroc": _safe_auc(y, g["_p"]), "auprc": float(average_precision_score(y, g["_p"])), "cal_intercept": a, "cal_slope": b})
    return pd.DataFrame(rows)


def bootstrap_auroc_by_stay(df: pd.DataFrame, p: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Stay-level cluster bootstrap of the pooled AUROC."""
    rng = np.random.default_rng(seed)
    d = df.assign(_p=np.asarray(p, float))
    groups = {s: g for s, g in d.groupby("stay_id")}
    keys = np.array(list(groups))
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(keys, len(keys), replace=True)
        b = pd.concat([groups[k] for k in draw], ignore_index=True)
        vals.append(_safe_auc(b["outcome"].to_numpy(int), b["_p"].to_numpy()))
    vals = np.asarray(vals, float)
    return {"auroc": _safe_auc(d["outcome"].to_numpy(int), d["_p"].to_numpy()), "ci_low": float(np.nanquantile(vals, 0.025)), "ci_high": float(np.nanquantile(vals, 0.975))}


def paired_auroc_difference(df: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Stay-level bootstrap of AUROC(a) - AUROC(b) on the same landmark rows."""
    rng = np.random.default_rng(seed)
    d = df.assign(_a=np.asarray(p_a, float), _b=np.asarray(p_b, float))
    groups = {s: g for s, g in d.groupby("stay_id")}
    keys = np.array(list(groups))
    diffs = []
    for _ in range(n_boot):
        draw = rng.choice(keys, len(keys), replace=True)
        b = pd.concat([groups[k] for k in draw], ignore_index=True)
        y = b["outcome"].to_numpy(int)
        diffs.append(_safe_auc(y, b["_a"]) - _safe_auc(y, b["_b"]))
    diffs = np.asarray(diffs, float)
    y = d["outcome"].to_numpy(int)
    return {"difference": _safe_auc(y, d["_a"]) - _safe_auc(y, d["_b"]), "ci_low": float(np.nanquantile(diffs, 0.025)), "ci_high": float(np.nanquantile(diffs, 0.975))}


def treatment_aware_summary(df: pd.DataFrame, p: np.ndarray, prevention_col: str = "prevention_24h", low_braden: float = 12.0) -> pd.DataFrame:
    """AUROC and event rate stratified by documented prevention, plus the adjusted OR of prevention on the outcome among low-Braden landmarks."""
    d = df.assign(_p=np.asarray(p, float))
    rows = []
    for flag, g in d.groupby(prevention_col):
        y = g["outcome"].to_numpy(int)
        rows.append({"prevention": int(flag), "n": len(g), "event_rate": float(y.mean()), "auroc": _safe_auc(y, g["_p"].to_numpy())})
    out = pd.DataFrame(rows)
    low = d[d["braden_last"] <= low_braden].dropna(subset=["braden_last"])
    if len(low) and low["outcome"].nunique() == 2 and low[prevention_col].nunique() == 2:
        X = sm.add_constant(low[[prevention_col, "braden_last", "landmark_day"]].astype(float))
        res = sm.Logit(low["outcome"].to_numpy(int), X).fit(disp=False)
        ci = res.conf_int().loc[prevention_col]
        out.attrs["prevention_or_low_braden"] = float(np.exp(res.params[prevention_col]))
        out.attrs["prevention_or_ci"] = (float(np.exp(ci[0])), float(np.exp(ci[1])))
        out.attrs["n_low_braden"] = int(len(low))
    return out


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2)) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    n = len(y)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * t / (1 - t), "net_benefit_all": y.mean() - (1 - y.mean()) * t / (1 - t), "alert_rate": float(pred.mean())})
    return pd.DataFrame(rows)
