"""Projection heterogeneity index (PHI) per source region, with nulls and confidence intervals.

Components computed on a neurons x targets matrix ``M`` (fractions) binarised at a threshold:

* ``phi_jaccard``: mean pairwise Jaccard *distance* between neurons (0 = all identical target sets).
* ``motif_entropy``: normalised Shannon entropy of distinct target-set motifs.
* ``divergence``: mean number of targets per neuron and the fraction of multi-target neurons.
* ``bulk_explained``: mean per-neuron cosine similarity to the bulk vector (if given).

The key null is **independent sampling from the bulk map**: each neuron draws its observed number
of targets without replacement with probabilities proportional to the bulk vector. Random sampling
maximises motif diversity, so a PHI *below* the null (negative z) means neurons cluster into a
limited set of projection motifs (parallel channels / subpopulations), whereas a PHI *above* the
null (positive z) means targets are combined more exclusively than chance (anti-correlated
pathways). Bootstrap CIs resample neurons; rarefaction equalises n across regions.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .concordance import binarize


def _binary_matrix(M, threshold: float) -> np.ndarray:
    A = np.asarray(M, float)
    if A.ndim != 2:
        raise ValueError("M must be 2-D (neurons x targets)")
    return np.vstack([binarize(row, threshold) for row in A]) if len(A) else np.zeros((0, A.shape[1]), bool)


def mean_pairwise_jaccard_distance(B: np.ndarray) -> float:
    """Mean over neuron pairs of 1 - |A∩B|/|A∪B| (vectorised)."""
    B = np.asarray(B, float)
    n = B.shape[0]
    if n < 2:
        return float("nan")
    inter = B @ B.T
    sizes = B.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    iu = np.triu_indices(n, k=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union[iu] > 0, inter[iu] / union[iu], 1.0)
    return float(np.mean(1.0 - jac))


def mean_nearest_neighbour_distance(B: np.ndarray) -> float:
    """Mean over neurons of the Jaccard distance to its most similar other neuron.

    0 = every neuron has a twin (strong motif structure); random sampling from a bulk map gives
    intermediate values. More sensitive to parallel channels than the mean pairwise distance when
    the bulk map is concentrated on a few targets.
    """
    B = np.asarray(B, float)
    n = B.shape[0]
    if n < 2:
        return float("nan")
    inter = B @ B.T
    sizes = B.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / union, 1.0)
    dist = 1.0 - jac
    np.fill_diagonal(dist, np.inf)
    return float(np.mean(dist.min(axis=1)))


def motif_entropy(B: np.ndarray) -> float:
    """Normalised entropy (0-1) of distinct target-set motifs; 1 = every neuron unique."""
    n = B.shape[0]
    if n < 2:
        return float("nan")
    _, counts = np.unique(B.astype(np.uint8), axis=0, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(n))


def divergence(B: np.ndarray) -> Dict[str, float]:
    k = B.sum(axis=1)
    return {"targets_per_neuron": float(k.mean()) if k.size else float("nan"),
            "multi_target_fraction": float(np.mean(k >= 2)) if k.size else float("nan")}


def bulk_explained(M: np.ndarray, bulk: np.ndarray) -> float:
    """Mean cosine similarity between each neuron's fraction vector and the bulk vector."""
    M, bulk = np.asarray(M, float), np.asarray(bulk, float)
    nb = np.linalg.norm(bulk)
    nm = np.linalg.norm(M, axis=1)
    ok = (nm > 0) & (nb > 0)
    return float(np.mean((M[ok] @ bulk) / (nm[ok] * nb))) if ok.any() else float("nan")


def heterogeneity_index(M, bulk: Optional[Sequence[float]] = None, threshold: float = 0.01) -> Dict[str, float]:
    """All heterogeneity components for one source region."""
    B = _binary_matrix(M, threshold)
    out = {"n_neurons": int(B.shape[0]), "phi_jaccard": mean_pairwise_jaccard_distance(B),
           "nn_distance": mean_nearest_neighbour_distance(B), "motif_entropy": motif_entropy(B), **divergence(B),
           "n_population_targets": int(B.any(axis=0).sum())}
    if bulk is not None:
        out["bulk_explained"] = bulk_explained(np.asarray(M, float), bulk)
    return out


# ---------------------------------------------------------------------- nulls and CIs
def independent_sampling_null(bulk: Sequence[float], n_targets_per_neuron: Sequence[int], n_sim: int = 500,
                              seed: int = 0, stat: Callable[[np.ndarray], float] = mean_pairwise_jaccard_distance
                              ) -> np.ndarray:
    """Null distribution of ``stat`` when each neuron samples its targets independently from ``bulk``.

    Neuron i draws ``n_targets_per_neuron[i]`` distinct targets with probability proportional to
    the bulk vector (without replacement). Returns ``n_sim`` values of the statistic.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(np.asarray(bulk, float), 0, None)
    if p.sum() <= 0:
        raise ValueError("bulk vector has no mass")
    p = p / p.sum()
    T = p.size
    ks = np.clip(np.asarray(n_targets_per_neuron, int), 0, int((p > 0).sum()))
    out = np.empty(n_sim)
    for s in range(n_sim):
        B = np.zeros((ks.size, T), dtype=bool)
        for i, k in enumerate(ks):
            if k > 0:
                B[i, rng.choice(T, size=k, replace=False, p=p)] = True
        out[s] = stat(B)
    return out


def null_deviation(M, bulk: Sequence[float], threshold: float = 0.01, n_sim: int = 500, seed: int = 0,
                   stat: Callable[[np.ndarray], float] = mean_pairwise_jaccard_distance) -> Dict[str, float]:
    """Observed statistic vs the independent-sampling null: z-score and two-sided permutation p.

    ``z < 0``: fewer distinct motifs than random draws from the bulk map (parallel channels);
    ``z > 0``: more mutually exclusive targeting than chance. Use ``stat=mean_nearest_neighbour_distance``
    as the primary structure detector; ``mean_pairwise_jaccard_distance`` (PHI) has low power when
    the bulk map is concentrated on a few targets (random draws then overlap heavily too).
    """
    B = _binary_matrix(M, threshold)
    obs = stat(B)
    null = independent_sampling_null(bulk, B.sum(axis=1), n_sim=n_sim, seed=seed, stat=stat)
    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 0 else float("nan")
    p_upper = (np.sum(null >= obs) + 1) / (n_sim + 1)
    p_lower = (np.sum(null <= obs) + 1) / (n_sim + 1)
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_sd": float(sd), "z": float(z),
            "p": float(min(1.0, 2 * min(p_upper, p_lower))), "stat": getattr(stat, "__name__", "stat")}


excess_heterogeneity = null_deviation  # backwards-compatible alias


def bootstrap_ci(M, stat: Callable[[np.ndarray], float] = mean_pairwise_jaccard_distance, threshold: float = 0.01,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Neuron-level bootstrap (point estimate, lower, upper) of a statistic on the binarised matrix."""
    rng = np.random.default_rng(seed)
    B = _binary_matrix(M, threshold)
    n = B.shape[0]
    est = stat(B)
    boots = np.array([stat(B[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


def rarefied_statistic(M, n_common: int, stat: Callable[[np.ndarray], float] = mean_pairwise_jaccard_distance,
                       threshold: float = 0.01, n_rep: int = 200, seed: int = 0) -> float:
    """Mean statistic over random subsamples of ``n_common`` neurons (equalises n across regions)."""
    rng = np.random.default_rng(seed)
    B = _binary_matrix(M, threshold)
    if B.shape[0] < n_common:
        return float("nan")
    return float(np.mean([stat(B[rng.choice(B.shape[0], size=n_common, replace=False)]) for _ in range(n_rep)]))


def region_table(M_by_region: Dict[str, pd.DataFrame], bulk_by_region: Optional[Dict[str, pd.Series]] = None,
                 threshold: float = 0.01, n_common: Optional[int] = None, n_sim: int = 300, n_boot: int = 500,
                 min_neurons: int = 10, seed: int = 0) -> pd.DataFrame:
    """Per-region heterogeneity summary: PHI with bootstrap CI, rarefied PHI, null-deviation z/p (``dev_*``)."""
    rows = []
    for region, M in M_by_region.items():
        if len(M) < min_neurons:
            continue
        bulk = None
        if bulk_by_region is not None and region in bulk_by_region:
            bulk = bulk_by_region[region].reindex(M.columns, fill_value=0.0).to_numpy()
        row = {"region": region, **heterogeneity_index(M, bulk, threshold)}
        est, lo, hi = bootstrap_ci(M, threshold=threshold, n_boot=n_boot, seed=seed)
        row.update({"phi_ci_low": lo, "phi_ci_high": hi})
        if n_common is not None:
            row["phi_rarefied"] = rarefied_statistic(M, n_common, threshold=threshold, seed=seed)
        if bulk is not None and np.sum(bulk) > 0:
            dev = null_deviation(M, bulk, threshold, n_sim, seed)
            row.update({f"dev_{k}": v for k, v in dev.items() if k != "stat"})
            dev_nn = null_deviation(M, bulk, threshold, n_sim, seed, stat=mean_nearest_neighbour_distance)
            row.update({f"devnn_{k}": v for k, v in dev_nn.items() if k != "stat"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("region") if rows else pd.DataFrame()


__all__ = ["mean_pairwise_jaccard_distance", "mean_nearest_neighbour_distance", "motif_entropy", "divergence",
           "bulk_explained", "heterogeneity_index",
           "independent_sampling_null", "null_deviation", "excess_heterogeneity", "bootstrap_ci", "rarefied_statistic",
           "region_table"]
