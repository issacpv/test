"""Gene-set enrichment with random-gene and ensemble (spatial) nulls.

Standard GSEA-style tests ask whether the genes in a category score higher
than *randomly chosen genes*.  Fulcher, Arnatkeviciute & Fornito (2021,
*Nat Commun*) showed this is badly miscalibrated for spatial transcriptomic
atlases: some categories (e.g. containing many spatially smooth, highly
co-expressed genes) are enriched for almost *any* smooth brain map.  Their
remedy is an *ensemble-based* null: recompute category scores for an ensemble
of random phenotypes that share the spatial structure of real maps (spin or
variogram surrogates of the target map, or random SA-matched maps), and
compare the observed category score to that distribution.

This module implements both nulls on top of a fast matrix formulation:
with z-scored expression ``Z`` (regions x genes) and z-scored map ``y``,
gene scores are ``Z.T @ y / (n-1)``; for ``S`` surrogate maps this is a single
``Z.T @ Y`` product, and category scores follow from an indicator matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _zscore(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    mu = a.mean(axis=axis, keepdims=True)
    sd = a.std(axis=axis, ddof=1, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (a - mu) / sd


def _rank(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.apply_along_axis(stats.rankdata, axis, np.asarray(a, dtype=float))


def gene_scores(expr: pd.DataFrame, y: Sequence[float], method: str = "spearman") -> pd.Series:
    """Correlation of every gene's regional expression with map ``y``."""
    X = expr.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    if method == "spearman":
        X, yv = _rank(X, 0), stats.rankdata(yv)
    Z = _zscore(X, 0)
    zy = _zscore(yv)
    r = (Z.T @ zy) / (len(yv) - 1)
    return pd.Series(r, index=expr.columns, name="r")


def _scores_matrix(expr: pd.DataFrame, Y: np.ndarray, method: str) -> np.ndarray:
    """(n_genes, n_maps) correlations for many maps at once."""
    X = expr.to_numpy(dtype=float)
    if method == "spearman":
        X, Y = _rank(X, 0), _rank(Y, 1)
    Z = _zscore(X, 0)
    ZY = _zscore(Y, 1)  # maps are rows -> z-score each row across regions
    return (Z.T @ ZY.T) / (X.shape[0] - 1)


def category_statistic(gene_r: np.ndarray, members: np.ndarray, statistic: str = "mean") -> np.ndarray:
    """Category score from gene scores.

    ``members`` is (n_sets, n_genes) boolean; ``gene_r`` is (n_genes,) or
    (n_genes, n_maps).  ``"mean"`` averages r; ``"mean_abs"`` averages |r|.
    """
    g = np.abs(gene_r) if statistic == "mean_abs" else gene_r
    M = members.astype(float)
    return (M @ g) / M.sum(axis=1, keepdims=True if g.ndim == 2 else False).reshape(-1, *([1] if g.ndim == 2 else []))


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def enrichment_with_nulls(
    expr: pd.DataFrame,
    y: Sequence[float],
    gene_sets: Dict[str, Sequence[str]],
    null_maps: Optional[np.ndarray] = None,
    n_gene_null: int = 2000,
    method: str = "spearman",
    statistic: str = "mean",
    min_size: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Category enrichment under a random-gene null and an ensemble null.

    Parameters
    ----------
    expr
        regions x genes expression (already gene-filtered).
    y
        parcellated map aligned to ``expr.index``.
    gene_sets
        ``{category: [gene symbols]}``; genes absent from ``expr`` are ignored.
    null_maps
        (n_null, n_regions) surrogate maps (spin / variogram / Moran) for the
        ensemble null.  If None, only the gene null is computed.
    n_gene_null
        Number of random gene resamples (size-matched) per category.

    Returns
    -------
    DataFrame with ``category, size, score, p_gene, q_gene, p_ensemble,
    q_ensemble, z_ensemble``.
    """
    rng = np.random.default_rng(seed)
    genes = np.asarray(expr.columns)
    gidx = {g: i for i, g in enumerate(genes)}
    names, members = [], []
    for cat, gl in gene_sets.items():
        m = np.zeros(len(genes), dtype=bool)
        for g in gl:
            if g in gidx:
                m[gidx[g]] = True
        if m.sum() >= min_size:
            names.append(cat)
            members.append(m)
    if not names:
        return pd.DataFrame(columns=["category", "size", "score", "p_gene", "q_gene", "p_ensemble", "q_ensemble", "z_ensemble"])
    M = np.vstack(members)
    sizes = M.sum(axis=1)

    r_obs = gene_scores(expr, y, method).to_numpy()
    score = category_statistic(r_obs, M, statistic)

    # random-gene null: for each category, resample size-matched gene sets
    g_for_null = np.abs(r_obs) if statistic == "mean_abs" else r_obs
    p_gene = np.empty(len(names))
    for i, s in enumerate(sizes):
        draws = rng.choice(g_for_null, size=(n_gene_null, int(s)), replace=False if s <= len(genes) else True).mean(axis=1) \
            if n_gene_null * s <= 5_000_000 else np.array([rng.choice(g_for_null, int(s), replace=False).mean() for _ in range(n_gene_null)])
        p_gene[i] = (np.sum(np.abs(draws) >= abs(score[i])) + 1) / (n_gene_null + 1)
    out = pd.DataFrame({"category": names, "size": sizes, "score": score, "p_gene": p_gene, "q_gene": bh_fdr(p_gene)})

    if null_maps is not None:
        Y = np.asarray(null_maps, dtype=float)
        R = _scores_matrix(expr, Y, method)              # genes x n_null
        null_scores = category_statistic(R, M, statistic)  # sets x n_null
        p_ens = (np.sum(np.abs(null_scores) >= np.abs(score)[:, None], axis=1) + 1) / (Y.shape[0] + 1)
        mu, sd = null_scores.mean(axis=1), null_scores.std(axis=1, ddof=1)
        out["p_ensemble"] = p_ens
        out["q_ensemble"] = bh_fdr(p_ens)
        out["z_ensemble"] = (score - mu) / np.where(sd == 0, np.nan, sd)
    else:
        out["p_ensemble"] = np.nan
        out["q_ensemble"] = np.nan
        out["z_ensemble"] = np.nan
    return out.sort_values("p_gene").reset_index(drop=True)


def category_false_positive_rate(
    expr: pd.DataFrame, gene_sets: Dict[str, Sequence[str]], random_maps: np.ndarray,
    null_maps_per_random: Optional[np.ndarray] = None, alpha: float = 0.05, method: str = "spearman",
) -> pd.DataFrame:
    """How often each category is "significant" for random (SA-matched) maps.

    This reproduces the core diagnostic of Fulcher et al. (2021): under the
    random-gene null, some categories are flagged for a large fraction of
    random phenotypes.  ``random_maps`` is (n_random, n_regions).
    """
    rows = []
    for k in range(random_maps.shape[0]):
        res = enrichment_with_nulls(expr, random_maps[k], gene_sets, null_maps=null_maps_per_random,
                                    n_gene_null=500, method=method, seed=k)
        rows.append(res.set_index("category")[["p_gene", "p_ensemble"]] < alpha)
    fp = pd.concat(rows, axis=1, keys=range(len(rows)))
    return pd.DataFrame({
        "fpr_gene_null": fp.xs("p_gene", axis=1, level=1).mean(axis=1),
        "fpr_ensemble_null": fp.xs("p_ensemble", axis=1, level=1).mean(axis=1),
    })


def make_gene_sets(genes: Sequence[str], n_sets: int = 20, size: int = 30, seed: int = 0,
                   planted: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """Random gene categories (plus optional planted ones) for simulations."""
    rng = np.random.default_rng(seed)
    genes = list(genes)
    sets = {f"SET{i:02d}": list(rng.choice(genes, size=size, replace=False)) for i in range(n_sets)}
    if planted:
        sets.update(planted)
    return sets
