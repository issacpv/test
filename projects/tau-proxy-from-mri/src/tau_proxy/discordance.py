"""T/N discordance analysis: is the MRI 'tau' signal just neurodegeneration?

Given out-of-fold probabilities from a model, tau labels and N status, this
module computes (i) AUC restricted to N− participants (tau signal without
measurable atrophy), (ii) false-positive rates in T−N+ vs T−N− (does atrophy
without tau fool the model?), and (iii) a paired comparison of two models
(e.g., with vs without WMH) on the T−N+ false-positive rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .labels import tn_quadrant


def quadrant_table(t_pos: pd.Series, n_pos: pd.Series) -> pd.DataFrame:
    q = tn_quadrant(t_pos, n_pos)
    tab = q.value_counts(dropna=True).rename_axis("quadrant").reset_index(name="n")
    tab["fraction"] = tab["n"] / tab["n"].sum()
    return tab.sort_values("quadrant").reset_index(drop=True)


def auc_within_stratum(prob: np.ndarray, t_pos: np.ndarray, mask: np.ndarray) -> float:
    """AUC of ``prob`` for T+ vs T− restricted to ``mask`` rows (NaN if one class only)."""
    m = np.asarray(mask, dtype=bool) & ~np.isnan(np.asarray(t_pos, dtype=float))
    yt = np.asarray(t_pos)[m]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, np.asarray(prob)[m]))


def false_positive_rate(prob: np.ndarray, t_pos: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Fraction of T− rows within ``mask`` with prob ≥ threshold."""
    m = np.asarray(mask, dtype=bool) & (np.asarray(t_pos) == 0)
    if m.sum() == 0:
        return float("nan")
    return float((np.asarray(prob)[m] >= threshold).mean())


def threshold_at_sensitivity(prob: np.ndarray, t_pos: np.ndarray, sensitivity: float = 0.9) -> float:
    """Largest threshold achieving at least ``sensitivity`` among T+."""
    pos = np.asarray(prob)[np.asarray(t_pos) == 1]
    if pos.size == 0:
        return float("nan")
    return float(np.quantile(pos, 1 - sensitivity, method="lower"))


def discordance_report(
    prob: np.ndarray,
    t_pos: pd.Series,
    n_pos: pd.Series,
    sensitivity: float = 0.9,
) -> dict[str, float]:
    """Summary of the T/N discordance test for one model."""
    t = np.asarray(t_pos, dtype=float)
    n = np.asarray(n_pos, dtype=float)
    thr = threshold_at_sensitivity(prob, t, sensitivity)
    return {
        "auc_all": auc_within_stratum(prob, t, np.ones_like(t, dtype=bool)),
        "auc_N_neg": auc_within_stratum(prob, t, n == 0),
        "auc_N_pos": auc_within_stratum(prob, t, n == 1),
        "threshold_at_sens": thr,
        "fpr_TnegNneg": false_positive_rate(prob, t, n == 0, thr),
        "fpr_TnegNpos": false_positive_rate(prob, t, n == 1, thr),
    }


def paired_fpr_difference(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    t_pos: pd.Series,
    n_pos: pd.Series,
    groups: np.ndarray,
    sensitivity: float = 0.9,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap (subject-level) difference in T−N+ false-positive rate: model b − model a."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t_pos, dtype=float)
    n = np.asarray(n_pos, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}

    def stat(idx: np.ndarray) -> float:
        ta, tb = threshold_at_sensitivity(prob_a[idx], t[idx], sensitivity), threshold_at_sensitivity(prob_b[idx], t[idx], sensitivity)
        return false_positive_rate(prob_b[idx], t[idx], n[idx] == 1, tb) - false_positive_rate(prob_a[idx], t[idx], n[idx] == 1, ta)

    point = stat(np.arange(len(t)))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        s = stat(idx)
        if not np.isnan(s):
            boots.append(s)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan)
    return {"delta_fpr_TnegNpos": float(point), "ci_low": float(lo), "ci_high": float(hi)}
