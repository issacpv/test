"""Clinical warning metrics for glucose forecasts.

The clinically important question is not RMSE but whether a forecaster warns of
hypoglycemia in time. These functions evaluate a forecast as a hypoglycemia
(<70 mg/dL) alarm: sensitivity, false-alarm rate, median lead time, and the
Clarke error-grid zone proportions.
"""
from __future__ import annotations

import numpy as np

HYPO = 70.0    # mg/dL
HYPER = 180.0  # mg/dL


def hypo_warning(y_true: np.ndarray, y_pred: np.ndarray, thresh: float = HYPO) -> dict:
    """Treat 'predicted glucose < thresh' as a hypo alarm; score vs true events.

    Returns sensitivity (recall of true hypo), false-alarm rate (1 - specificity),
    precision and the count of events.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    true_evt = y_true < thresh
    alarm = y_pred < thresh
    tp = np.sum(alarm & true_evt)
    fn = np.sum(~alarm & true_evt)
    fp = np.sum(alarm & ~true_evt)
    tn = np.sum(~alarm & ~true_evt)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (fp + tn) if (fp + tn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    return {"sensitivity": float(sens), "false_alarm_rate": float(far),
            "precision": float(prec), "n_events": int(true_evt.sum())}


def detection_lead_time(times_min: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        horizon_min: int, thresh: float = HYPO) -> float:
    """Median lead time (min) between the alarm and the true hypo crossing.

    ``times_min`` is the timestamp (minutes) of each prediction's *target*; a
    prediction made ``horizon_min`` earlier that fires before the true event
    contributes ``horizon_min`` of lead time (a simplified per-window estimate).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fired_before = (y_pred < thresh) & (y_true < thresh)
    if fired_before.sum() == 0:
        return float("nan")
    # each correctly-anticipated event is warned ~horizon_min ahead
    return float(horizon_min)


def clarke_zones(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Approximate Clarke error-grid zone proportions (A-E).

    A simplified implementation of the standard zones sufficient for benchmark
    comparison; A/B are clinically acceptable, C-E increasingly dangerous.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    zones = {z: 0 for z in "ABCDE"}
    for r, p in zip(y_true, y_pred):
        if (r <= 70 and p <= 70) or abs(p - r) <= 0.2 * r:
            zones["A"] += 1
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            zones["E"] += 1
        elif ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7 / 5) * r - 182):
            zones["C"] += 1
        elif (r >= 240 and (70 <= p <= 180)) or (r <= 70 and (70 <= p <= 180)):
            zones["D"] += 1
        else:
            zones["B"] += 1
    n = max(len(y_true), 1)
    return {z: c / n for z, c in zones.items()}


def calibration_bias_by_range(y_true: np.ndarray, y_pred: np.ndarray,
                              edges=(0, 70, 180, 400)) -> dict:
    """Mean signed error (pred - true) within glucose ranges (detects range bias)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi)
        out[f"{lo}-{hi}"] = float(np.mean(y_pred[m] - y_true[m])) if m.sum() else float("nan")
    return out
