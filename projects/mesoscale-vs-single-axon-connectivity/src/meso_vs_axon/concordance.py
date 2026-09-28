"""Concordance between a bulk (mesoscale) projection vector and single-neuron target vectors.

Inputs are aligned 1-D vectors over the same target labels (e.g. ``"<acronym>_<ipsi|contra>"``):
``bulk`` (non-negative, e.g. normalised projection volume) and a neurons x targets matrix ``M``
of per-neuron axon fractions. Metrics: Jaccard on binarised vectors, Spearman, top-weighted
Kendall tau, AUROC / precision@k / recall@k with the bulk vector as a score, pooled recovery
curves and permutation nulls.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def binarize(v: np.ndarray, threshold: float = 0.01, relative: bool = True) -> np.ndarray:
    """Targets with weight >= threshold (relative: threshold is a fraction of the vector's sum)."""
    v = np.asarray(v, float)
    thr = threshold * v.sum() if relative else threshold
    return v >= max(thr, np.finfo(float).tiny)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union > 0 else float("nan")


def weighted_rank_correlation(bulk: np.ndarray, single: np.ndarray) -> float:
    """Top-weighted Kendall tau (scipy ``weightedtau``): strong targets matter more than weak ones."""
    return float(stats.weightedtau(np.asarray(bulk, float), np.asarray(single, float)).statistic)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC of ``scores`` for binary ``labels`` (nan if one class is missing)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> Tuple[float, float]:
    """Precision/recall of the top-k bulk targets against the neuron's true target set."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.sum() == 0:
        return float("nan"), float("nan")
    top = np.argsort(-scores)[:k]
    hits = labels[top].sum()
    return float(hits / k), float(hits / labels.sum())


def per_neuron_concordance(bulk: np.ndarray, single: np.ndarray, threshold: float = 0.01, k: int = 5) -> Dict[str, float]:
    bulk, single = np.asarray(bulk, float), np.asarray(single, float)
    b_bin, s_bin = binarize(bulk, threshold), binarize(single, threshold)
    prec, rec = precision_recall_at_k(bulk, s_bin, k)
    rho = stats.spearmanr(bulk, single).correlation if single.sum() > 0 else float("nan")
    return {"jaccard": jaccard(b_bin, s_bin), "spearman": float(rho), "weighted_tau": weighted_rank_correlation(bulk, single),
            "auroc": auroc(bulk, s_bin), f"precision_at_{k}": prec, f"recall_at_{k}": rec,
            "n_targets": int(s_bin.sum()), "n_targets_bulk": int(b_bin.sum())}


def concordance_table(bulk: pd.Series, M: pd.DataFrame, threshold: float = 0.01, k: int = 5) -> pd.DataFrame:
    """Per-neuron concordance metrics; ``M`` is aligned to ``bulk.index`` (missing targets = 0)."""
    M = M.reindex(columns=bulk.index, fill_value=0.0)
    rows = [dict(neuron=idx, **per_neuron_concordance(bulk.to_numpy(), row.to_numpy(), threshold, k))
            for idx, row in M.iterrows()]
    return pd.DataFrame(rows).set_index("neuron")


def pooled_vector(M: pd.DataFrame, weights: Optional[np.ndarray] = None) -> pd.Series:
    """Population vector from single neurons (mean of per-neuron fractions, optionally weighted)."""
    w = np.ones(len(M)) if weights is None else np.asarray(weights, float)
    v = (M.to_numpy(float) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
    return pd.Series(v, index=M.columns)


def pooled_concordance(bulk: pd.Series, M: pd.DataFrame) -> Dict[str, float]:
    M = M.reindex(columns=bulk.index, fill_value=0.0)
    pv = pooled_vector(M)
    return {"spearman": float(stats.spearmanr(bulk.to_numpy(), pv.to_numpy()).correlation),
            "pearson_log": float(np.corrcoef(np.log1p(1e3 * bulk.to_numpy()), np.log1p(1e3 * pv.to_numpy()))[0, 1]),
            "weighted_tau": weighted_rank_correlation(bulk.to_numpy(), pv.to_numpy()),
            "jaccard": jaccard(binarize(bulk.to_numpy()), binarize(pv.to_numpy())), "n_neurons": int(len(M))}


def subsampling_curve(bulk: pd.Series, M: pd.DataFrame, n_values: Sequence[int] = (1, 2, 5, 10, 20, 50, 100),
                      n_rep: int = 50, seed: int = 0) -> pd.DataFrame:
    """How many single neurons does it take to recover the bulk vector? Spearman vs. n (with replacement)."""
    rng = np.random.default_rng(seed)
    M = M.reindex(columns=bulk.index, fill_value=0.0)
    rows = []
    for n in n_values:
        if n > len(M):
            continue
        for r in range(n_rep):
            idx = rng.choice(len(M), size=n, replace=False)
            pv = pooled_vector(M.iloc[idx])
            rows.append({"n": n, "rep": r, "spearman": float(stats.spearmanr(bulk.to_numpy(), pv.to_numpy()).correlation),
                         "weighted_tau": weighted_rank_correlation(bulk.to_numpy(), pv.to_numpy())})
    return pd.DataFrame(rows)


def n_for_recovery(curve: pd.DataFrame, rho_target: float = 0.8, metric: str = "spearman") -> Optional[int]:
    """Smallest n whose mean metric reaches ``rho_target`` (None if never reached)."""
    g = curve.groupby("n")[metric].mean()
    hit = g[g >= rho_target]
    return int(hit.index[0]) if len(hit) else None


def label_permutation_null(bulk: pd.Series, M: pd.DataFrame, n_perm: int = 200, seed: int = 0,
                           metric: str = "auroc", threshold: float = 0.01) -> Tuple[float, np.ndarray, float]:
    """Null for the mean per-neuron metric obtained by permuting target labels of the bulk vector."""
    rng = np.random.default_rng(seed)
    obs = float(concordance_table(bulk, M, threshold)[metric].mean())
    b = bulk.to_numpy()
    null = []
    for _ in range(n_perm):
        pb = pd.Series(rng.permutation(b), index=bulk.index)
        null.append(float(concordance_table(pb, M, threshold)[metric].mean()))
    null = np.asarray(null)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, null, p


def cross_region_specificity(bulk_by_region: Dict[str, pd.Series], M_by_region: Dict[str, pd.DataFrame],
                             metric: str = "auroc", threshold: float = 0.01) -> pd.DataFrame:
    """Matrix: rows = neuron source region, columns = bulk vector's region; diagonal should dominate."""
    regions = list(M_by_region)
    out = pd.DataFrame(index=regions, columns=list(bulk_by_region), dtype=float)
    for r in regions:
        for b in bulk_by_region:
            out.loc[r, b] = concordance_table(bulk_by_region[b], M_by_region[r], threshold)[metric].mean()
    return out


__all__ = ["binarize", "jaccard", "weighted_rank_correlation", "auroc", "precision_recall_at_k",
           "per_neuron_concordance", "concordance_table", "pooled_vector", "pooled_concordance", "subsampling_curve",
           "n_for_recovery", "label_permutation_null", "cross_region_specificity"]
