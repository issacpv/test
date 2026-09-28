"""Subgroup fairness metrics and the leakage->inequity link.

Given diagnostic-probe scores and subgroup labels, compute per-group AUROC,
the max-min gap, and equalised-odds gap; and correlate label-wise demographic
decodability with label-wise subgroup gaps (RQ4).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def subgroup_auroc_gap(y_true: np.ndarray, scores: np.ndarray, group: np.ndarray) -> dict:
    """AUROC per group and the max-min gap."""
    y_true, scores, group = np.asarray(y_true), np.asarray(scores, float), np.asarray(group)
    per = {}
    for g in np.unique(group):
        m = group == g
        if m.sum() >= 10 and len(np.unique(y_true[m])) == 2:
            per[str(g)] = float(roc_auc_score(y_true[m], scores[m]))
    if len(per) < 2:
        return {"per_group": per, "gap": float("nan")}
    return {"per_group": per, "gap": float(max(per.values()) - min(per.values()))}


def equalised_odds_gap(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    """Max across groups of |TPR - meanTPR| + |FPR - meanFPR| at a fixed threshold."""
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    tprs, fprs = {}, {}
    for g in np.unique(group):
        m = group == g
        pos, neg = (y_true[m] == 1), (y_true[m] == 0)
        if pos.sum() and neg.sum():
            tprs[g] = float((y_pred[m][pos] == 1).mean())
            fprs[g] = float((y_pred[m][neg] == 1).mean())
    if len(tprs) < 2:
        return float("nan")
    mt, mf = np.mean(list(tprs.values())), np.mean(list(fprs.values()))
    return float(max(abs(tprs[g] - mt) + abs(fprs[g] - mf) for g in tprs))


def leakage_gap_correlation(leakage: dict[str, float], gaps: dict[str, float]) -> dict:
    """Spearman correlation between per-label demographic decodability and gap."""
    labels = [k for k in leakage if k in gaps and np.isfinite(leakage[k]) and np.isfinite(gaps[k])]
    if len(labels) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(labels)}
    x = [leakage[k] for k in labels]
    g = [gaps[k] for k in labels]
    rho, p = stats.spearmanr(x, g)
    return {"rho": float(rho), "p": float(p), "n": len(labels)}
