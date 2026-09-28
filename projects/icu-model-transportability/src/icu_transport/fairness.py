"""Subgroup performance parity with bootstrap CIs and permutation nulls.

The metrics follow the "sufficiency / separation" split of the fairness literature
(Barocas, Hardt & Narayanan, 2019): calibration-type metrics per group (intercept, slope, ECE)
test sufficiency; TPR / FPR at a fixed operating point test separation (equalised odds);
AUROC / AUPRC per group describe discrimination. Gaps are max - min across groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .calibration import calibration_slope_intercept, expected_calibration_error
from .shift import weighted_auroc

METRICS = ("n", "prevalence", "auroc", "auprc", "intercept", "slope", "ece", "tpr", "fpr", "ppv", "alert_rate")


def _group_metrics(y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    out = {"n": float(len(y)), "prevalence": float(y.mean()), "alert_rate": float((p >= threshold).mean())}
    if len(np.unique(y)) < 2 or len(y) < 20:
        for k in ("auroc", "auprc", "intercept", "slope", "ece", "tpr", "fpr", "ppv"):
            out[k] = float("nan")
        return out
    out["auroc"] = weighted_auroc(y, p)
    out["auprc"] = float(average_precision_score(y, p))
    si = calibration_slope_intercept(y, p)
    out["intercept"], out["slope"] = si["intercept"], si["slope"]
    out["ece"] = expected_calibration_error(y, p, n_bins)
    alert = p >= threshold
    out["tpr"] = float(alert[y == 1].mean())
    out["fpr"] = float(alert[y == 0].mean())
    out["ppv"] = float(y[alert].mean()) if alert.any() else float("nan")
    return out


def operating_threshold(p_source: np.ndarray, alert_rate: float = 0.10) -> float:
    """Threshold giving a fixed alert rate on the *source* site (kept fixed at target sites)."""
    return float(np.quantile(np.asarray(p_source, dtype=float), 1 - alert_rate))


def subgroup_metrics(y: np.ndarray, p: np.ndarray, group: np.ndarray, threshold: float, n_bins: int = 10,
                     n_boot: int = 200, random_state: int = 0, min_group_n: int = 50) -> pd.DataFrame:
    """Per-group metrics with percentile-bootstrap 95% CIs. Groups below ``min_group_n`` are dropped."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(group)
    rng = np.random.default_rng(random_state)
    rows = []
    for lvl in pd.unique(g):
        m = g == lvl
        if m.sum() < min_group_n:
            continue
        est = _group_metrics(y[m], p[m], threshold, n_bins)
        boots = {k: [] for k in METRICS}
        idx_all = np.flatnonzero(m)
        for _ in range(n_boot):
            idx = rng.choice(idx_all, len(idx_all), replace=True)
            b = _group_metrics(y[idx], p[idx], threshold, n_bins)
            for k in METRICS:
                boots[k].append(b[k])
        row = {"group": lvl}
        for k in METRICS:
            row[k] = est[k]
            if n_boot:
                arr = np.array(boots[k], dtype=float)
                row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def parity_gaps(table: pd.DataFrame, metrics: tuple[str, ...] = ("auroc", "intercept", "slope", "ece", "tpr", "fpr")) -> pd.Series:
    """Max - min gap across groups for each metric (absolute gaps; intercept/slope as-is)."""
    return pd.Series({m: float(table[m].max() - table[m].min()) for m in metrics if m in table})


def permutation_gap_test(y: np.ndarray, p: np.ndarray, group: np.ndarray, threshold: float, metric: str = "ece",
                         n_perm: int = 500, random_state: int = 0, min_group_n: int = 50) -> dict[str, float]:
    """Null distribution of the max-min gap under random group assignment (group labels shuffled,
    group sizes preserved). p-value = P(null gap >= observed gap)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(group).copy()
    rng = np.random.default_rng(random_state)

    def gap(gg: np.ndarray) -> float:
        vals = []
        for lvl in pd.unique(gg):
            m = gg == lvl
            if m.sum() >= min_group_n:
                vals.append(_group_metrics(y[m], p[m], threshold, 10)[metric])
        vals = [v for v in vals if np.isfinite(v)]
        return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")

    obs = gap(g)
    null = np.array([gap(rng.permutation(g)) for _ in range(n_perm)])
    pval = float((np.sum(null >= obs) + 1) / (np.sum(np.isfinite(null)) + 1))
    return {"metric": metric, "observed_gap": obs, "null_mean": float(np.nanmean(null)),
            "null_95": float(np.nanpercentile(null, 95)), "p_value": pval}


def parity_transfer_table(source: dict[str, np.ndarray], targets: dict[str, dict[str, np.ndarray]],
                          threshold: float, metrics: tuple[str, ...] = ("auroc", "intercept", "ece", "tpr", "fpr"),
                          n_boot: int = 100) -> pd.DataFrame:
    """Gap at the source vs gap at each target site (H3: gaps widen under transfer).

    ``source`` and each target dict carry keys ``y``, ``p``, ``group``. Returns one row per site
    with the max-min gap per metric and the number of groups compared.
    """
    rows = []
    sites = {"source": source, **targets}
    for name, d in sites.items():
        t = subgroup_metrics(d["y"], d["p"], d["group"], threshold, n_boot=n_boot)
        gaps = parity_gaps(t, metrics)
        rows.append({"site": name, "n_groups": len(t), **{f"gap_{k}": v for k, v in gaps.items()}})
    return pd.DataFrame(rows).set_index("site")
