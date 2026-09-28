"""Null models for area-level gene-gradient analyses.

- ``variogram_surrogates``: spatial-autocorrelation-preserving surrogates for a target map on
  3-D area centroids (variogram matching after Burt et al., 2020).
- ``stratified_permutation``: permute areas within strata of a covariate (e.g. tertiles of the
  Harris et al., 2019 hierarchy score) to test whether an association survives the hierarchy.
- ``gene_ensemble_null``: random gene sets matched on a nuisance property (Fulcher et al., 2021).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


def morans_i(x: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    x = np.asarray(x, float)
    D = squareform(pdist(np.asarray(coords, float)))
    np.fill_diagonal(D, np.inf)
    k = min(k, len(x) - 1)
    W = np.zeros_like(D)
    for i, nb in enumerate(np.argsort(D, axis=1)[:, :k]):
        W[i, nb] = 1.0
    W = np.maximum(W, W.T)
    z = x - x.mean()
    return float(len(x) / W.sum() * (W * np.outer(z, z)).sum() / (z ** 2).sum())


def _variogram(x: np.ndarray, D: np.ndarray, bins: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    g = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    which = np.digitize(d, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = which == b
        if m.any():
            out[b] = g[m].mean()
    return out


def variogram_surrogates(x: np.ndarray, coords: np.ndarray, n_surr: int = 1000,
                         kernel_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0), n_bins: int = 15,
                         seed: int = 0) -> np.ndarray:
    """Surrogate maps matching the empirical variogram of ``x`` over ``coords`` (n_surr x n)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    coords = np.asarray(coords, float)
    n = len(x)
    D = squareform(pdist(coords))
    Doff = D.copy()
    np.fill_diagonal(Doff, np.inf)
    d0 = float(np.median(Doff.min(axis=1)))
    bins = np.linspace(0, np.percentile(D[np.triu_indices(n, 1)], 60), n_bins + 1)
    gt = _variogram(x, D, bins)
    kernels = []
    for s in kernel_scales:
        K = np.exp(-0.5 * (D / (s * d0)) ** 2)
        kernels.append(K / K.sum(1, keepdims=True))
    out = np.empty((n_surr, n))
    for i in range(n_surr):
        perm = rng.permutation(x)
        best = None
        for K in kernels:
            s = K @ perm
            if s.std() < 1e-12:
                continue
            s = (s - s.mean()) / s.std() * x.std()
            g = _variogram(s, D, bins)
            m = np.isfinite(gt) & np.isfinite(g)
            if m.sum() < 3 or np.std(g[m]) < 1e-12:
                continue
            b, a, _, _, _ = stats.linregress(g[m], gt[m])
            resid = np.mean((gt[m] - (a + b * g[m])) ** 2)
            if best is None or resid < best[0]:
                best = (resid, max(a, 0.0), max(b, 0.0), s)
        if best is None:  # pragma: no cover - degenerate geometry
            out[i] = perm
            continue
        _, a, b, s = best
        y = np.sqrt(b) * s + np.sqrt(a) * rng.standard_normal(n)
        out[i] = (y - y.mean()) / (y.std() + 1e-12) * x.std() + x.mean()
    return out


def stratified_permutation(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute ``values`` only within levels of ``strata``."""
    values = np.asarray(values)
    out = values.copy()
    for s in np.unique(strata):
        idx = np.nonzero(strata == s)[0]
        out[idx] = values[rng.permutation(idx)]
    return out


def permutation_p(observed: float, null: np.ndarray, two_sided: bool = True) -> float:
    null = np.asarray(null, float)
    if two_sided:
        return float((np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean())) + 1) / (len(null) + 1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def spatial_null_r2(
    fit_fn: Callable[[np.ndarray], float],
    y: np.ndarray,
    coords: np.ndarray,
    n_surr: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    """Empirical p-value for a prediction R^2 by refitting on variogram surrogates of the target."""
    obs = fit_fn(np.asarray(y, float))
    surr = variogram_surrogates(y, coords, n_surr=n_surr, seed=seed)
    null = np.array([fit_fn(s) for s in surr])
    return {"r2": float(obs), "p_spatial": permutation_p(obs, null, two_sided=False),
            "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95))}


def hierarchy_null_r2(
    fit_fn: Callable[[np.ndarray], float],
    y: np.ndarray,
    hierarchy: np.ndarray,
    n_perm: int = 500,
    n_strata: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """Empirical p-value for R^2 under permutations of the target within hierarchy strata."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    q = np.quantile(hierarchy, np.linspace(0, 1, n_strata + 1))
    strata = np.clip(np.digitize(hierarchy, q[1:-1]), 0, n_strata - 1)
    obs = fit_fn(y)
    null = np.array([fit_fn(stratified_permutation(y, strata, rng)) for _ in range(n_perm)])
    return {"r2": float(obs), "p_hierarchy": permutation_p(obs, null, two_sided=False), "null_mean": float(null.mean())}


def gene_ensemble_null(
    gene_scores: np.ndarray,
    gene_set_mask: np.ndarray,
    match_on: Optional[np.ndarray] = None,
    n_perm: int = 5000,
    n_bins: int = 10,
    statistic: Callable[[np.ndarray], float] = lambda v: float(np.mean(np.abs(v))),
    seed: int = 0,
) -> Dict[str, float]:
    """Gene-set statistic vs random gene sets matched on ``match_on`` quantile bins."""
    rng = np.random.default_rng(seed)
    gene_scores = np.asarray(gene_scores, float)
    mask = np.asarray(gene_set_mask, bool)
    k = int(mask.sum())
    if k == 0:
        raise ValueError("empty gene set")
    obs = statistic(gene_scores[mask])
    idx = np.arange(len(gene_scores))
    if match_on is None:
        null = np.array([statistic(gene_scores[rng.choice(idx, k, replace=False)]) for _ in range(n_perm)])
    else:
        mo = np.asarray(match_on, float)
        edges = np.quantile(mo, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        b = np.clip(np.digitize(mo, edges) - 1, 0, n_bins - 1)
        need = np.bincount(b[mask], minlength=n_bins)
        pools = [idx[b == j] for j in range(n_bins)]
        null = np.empty(n_perm)
        for i in range(n_perm):
            d = [rng.choice(pools[j], min(need[j], len(pools[j])), replace=len(pools[j]) < need[j])
                 for j in range(n_bins) if need[j] > 0]
            null[i] = statistic(gene_scores[np.concatenate(d)])
    return {"stat": float(obs), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_ensemble": permutation_p(obs, null, two_sided=False), "z": float((obs - null.mean()) / (null.std() + 1e-12))}


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out
