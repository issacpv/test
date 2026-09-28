"""Subgroup metrics with stratified bootstrap CIs and label-noise-aware evaluation.

All functions accept ``y`` (n,) binary labels and ``s`` (n,) scores for one
harmonised class; multi-label results are obtained by looping over classes.
Optional ``groups`` (n,) is a patient identifier used to resample *patients*
(cluster bootstrap) so that multiple ECGs from one person are not treated as
independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class CI:
    """Point estimate and percentile bootstrap interval."""

    value: float
    low: float
    high: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.value, self.low, self.high


def _safe_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def _cluster_resample(rng: np.random.Generator, groups: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    pick = rng.integers(0, len(uniq), size=len(uniq))
    counts = np.bincount(pick, minlength=len(uniq))
    return np.repeat(np.arange(len(groups)), counts[inv])


def bootstrap_metric(y: np.ndarray, s: np.ndarray, metric: str = "auroc", n_boot: int = 1000,
                     groups: np.ndarray | None = None, weights: np.ndarray | None = None,
                     alpha: float = 0.05, seed: int = 0) -> CI:
    """Percentile bootstrap CI for AUROC / AUPRC, optionally clustered by ``groups`` and weighted."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    rng = np.random.default_rng(seed)

    def m(idx: np.ndarray) -> float:
        w = None if weights is None else weights[idx]
        if metric == "auroc":
            return _safe_auc(y[idx], s[idx], w)
        if metric == "auprc":
            return float(average_precision_score(y[idx], s[idx], sample_weight=w)) if y[idx].sum() > 0 else float("nan")
        raise ValueError(metric)

    full = np.arange(len(y))
    point = m(full)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _cluster_resample(rng, np.asarray(groups)) if groups is not None else rng.integers(0, len(y), len(y))
        vals[b] = m(idx)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return CI(point, float("nan"), float("nan"), len(y))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(point, float(lo), float(hi), len(y))


def subgroup_auroc(y: np.ndarray, s: np.ndarray, subgroup: np.ndarray, n_boot: int = 1000,
                   groups: np.ndarray | None = None, seed: int = 0) -> dict[str, CI]:
    """AUROC with bootstrap CI for every level of ``subgroup`` (e.g. sex, age band)."""
    out: dict[str, CI] = {}
    subgroup = np.asarray(subgroup)
    for lvl in np.unique(subgroup):
        m = subgroup == lvl
        g = None if groups is None else np.asarray(groups)[m]
        out[str(lvl)] = bootstrap_metric(y[m], s[m], "auroc", n_boot, groups=g, seed=seed)
    return out


def parity_gap(y: np.ndarray, s: np.ndarray, subgroup: np.ndarray, n_boot: int = 1000,
               groups: np.ndarray | None = None, seed: int = 0) -> CI:
    """max-min AUROC gap across subgroup levels, with a bootstrap CI computed on the gap itself."""
    y, s, subgroup = np.asarray(y).astype(int), np.asarray(s, float), np.asarray(subgroup)
    rng = np.random.default_rng(seed)
    levels = np.unique(subgroup)

    def gap(idx: np.ndarray) -> float:
        aucs = [_safe_auc(y[idx][subgroup[idx] == l], s[idx][subgroup[idx] == l]) for l in levels]
        aucs = [a for a in aucs if np.isfinite(a)]
        return float(max(aucs) - min(aucs)) if len(aucs) >= 2 else float("nan")

    point = gap(np.arange(len(y)))
    vals = []
    for _ in range(n_boot):
        idx = _cluster_resample(rng, np.asarray(groups)) if groups is not None else rng.integers(0, len(y), len(y))
        vals.append(gap(idx))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size == 0:
        return CI(point, float("nan"), float("nan"), len(y))
    return CI(point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(y))


def equalized_odds_gap(y: np.ndarray, s: np.ndarray, subgroup: np.ndarray, threshold: float) -> dict[str, float]:
    """TPR and FPR gaps (max-min across subgroup levels) at a fixed operating point chosen on the source."""
    y, s, subgroup = np.asarray(y).astype(int), np.asarray(s, float), np.asarray(subgroup)
    pred = (s >= threshold).astype(int)
    tprs, fprs = [], []
    for l in np.unique(subgroup):
        m = subgroup == l
        pos, neg = (y[m] == 1), (y[m] == 0)
        if pos.sum():
            tprs.append(pred[m][pos].mean())
        if neg.sum():
            fprs.append(pred[m][neg].mean())
    return {
        "tpr_gap": float(max(tprs) - min(tprs)) if len(tprs) > 1 else float("nan"),
        "fpr_gap": float(max(fprs) - min(fprs)) if len(fprs) > 1 else float("nan"),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE with equal-width probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def transfer_drop(auc_in: float, auc_out: float) -> float:
    """Signed drop in AUROC from in-source to out-of-source evaluation."""
    return float(auc_in - auc_out)


# ----------------------------------------------------------------------------
# Label-noise-aware evaluation
# ----------------------------------------------------------------------------
def noise_sensitivity(y: np.ndarray, s: np.ndarray, flip_rates: tuple[float, ...] = (0.05, 0.10, 0.15),
                      n_sim: int = 200, symmetric: bool = False, seed: int = 0) -> dict[float, tuple[float, float, float]]:
    """AUROC under simulated class-conditional label noise.

    For each rate ``r`` a fraction ``r`` of *positive* labels is flipped to
    negative (and, if ``symmetric``, the same fraction of negatives to
    positive).  Returns ``{rate: (mean, 2.5th pct, 97.5th pct)}``.  If model
    rankings change across rates, differences on the noisy target are not
    trustworthy for that class.
    """
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {}
    for r in flip_rates:
        vals = []
        for _ in range(n_sim):
            yy = y.copy()
            k = int(round(r * len(pos)))
            if k:
                yy[rng.choice(pos, k, replace=False)] = 0
            if symmetric:
                k2 = int(round(r * len(neg)))
                if k2:
                    yy[rng.choice(neg, k2, replace=False)] = 1
            vals.append(_safe_auc(yy, s))
        vals = np.asarray([v for v in vals if np.isfinite(v)])
        out[r] = (float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals.size else (np.nan,) * 3
    return out


def agreement_restricted_auroc(y: np.ndarray, s: np.ndarray, agree_mask: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[CI, CI]:
    """AUROC on all records vs on the subset whose labels agree across mapping policies."""
    full = bootstrap_metric(y, s, "auroc", n_boot, seed=seed)
    m = np.asarray(agree_mask, bool)
    sub = bootstrap_metric(np.asarray(y)[m], np.asarray(s)[m], "auroc", n_boot, seed=seed)
    return full, sub


def rank_stability(scores_by_model: dict[str, np.ndarray], y: np.ndarray, flip_rate: float = 0.1,
                   n_sim: int = 100, seed: int = 0) -> float:
    """Kendall-tau-like stability of the model ranking under simulated label noise (1 = never changes)."""
    from scipy.stats import kendalltau

    names = list(scores_by_model)
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    base = np.array([_safe_auc(y, scores_by_model[n]) for n in names])
    pos = np.where(y == 1)[0]
    taus = []
    for _ in range(n_sim):
        yy = y.copy()
        k = int(round(flip_rate * len(pos)))
        if k:
            yy[rng.choice(pos, k, replace=False)] = 0
        noisy = np.array([_safe_auc(yy, scores_by_model[n]) for n in names])
        tau, _ = kendalltau(base, noisy)
        taus.append(tau if np.isfinite(tau) else 1.0)
    return float(np.mean(taus))
