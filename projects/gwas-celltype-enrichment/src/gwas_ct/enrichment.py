"""Permutation / bootstrap enrichment tests, multiple-testing correction and estimator concordance.

Two complementary tests are implemented:

* `bootstrap_gene_z_enrichment` — "GWAS→cell type": mean MAGMA gene Z of a cell type's top-decile
  genes versus random gene sets matched on a covariate (gene length or expression bin).
* `ewce_bootstrap` — "cell type→GWAS" (Skene et al. 2018): mean specificity of GWAS hit genes in each
  cell type versus random gene lists of the same size matched on expression level.

Plus `spatial_permutation` for domain-restricted specificity (permute domain labels within cell type),
`bh_fdr`, `cauchy_combination`, `concordance` and `resolution_curve`.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------------- utilities
def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (monotone), NaN-safe."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(q, 0, 1)
    out[ok] = adj
    return out


def cauchy_combination(pvals: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Cauchy combination test (Liu & Xie 2020) for possibly dependent p-values."""
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1 - 1e-16)
    w = np.ones_like(p) / len(p) if weights is None else np.asarray(weights, float) / np.sum(weights)
    t = np.sum(w * np.tan((0.5 - p) * np.pi))
    return float(0.5 - np.arctan(t) / np.pi)


def _bins(values: pd.Series, n_bins: int) -> pd.Series:
    return pd.qcut(values.rank(method="first"), n_bins, labels=False)


def _matched_random(rng: np.random.Generator, target: Sequence[str], bins: pd.Series, n_draw: int) -> np.ndarray:
    """(n_draw, len(target)) random genes with the same bin composition as `target`."""
    pools = {b: bins.index[bins == b].to_numpy() for b in bins.unique()}
    tb = bins.reindex(target)
    out = np.empty((n_draw, len(target)), dtype=object)
    for j, b in enumerate(tb.to_numpy()):
        out[:, j] = rng.choice(pools[b], size=n_draw, replace=True)
    return out


# ------------------------------------------------------------------------------- GWAS -> cell type
def bootstrap_gene_z_enrichment(gene_z: pd.Series, gene_set: Iterable[str], n_boot: int = 10_000,
                                match_on: Optional[pd.Series] = None, n_bins: int = 20, seed: int = 0,
                                min_genes: int = 20) -> Dict[str, float]:
    """Is the mean gene Z of `gene_set` higher than for random gene sets (matched on `match_on`)?

    Parameters
    ----------
    gene_z : gene-level statistics indexed by gene symbol (e.g. MAGMA ZSTAT).
    gene_set : cell-type-specific genes (e.g. top decile).
    match_on : optional covariate (gene length, mean expression) indexed like `gene_z`;
        random sets keep the same covariate-bin composition.

    Returns dict with n_genes, mean_z, null_mean, null_sd, effect_sd (standardised), p (one-sided).
    """
    rng = np.random.default_rng(seed)
    z = gene_z.dropna()
    genes = [g for g in dict.fromkeys(gene_set) if g in z.index]
    if len(genes) < min_genes:
        return {"n_genes": len(genes), "mean_z": np.nan, "null_mean": np.nan, "null_sd": np.nan,
                "effect_sd": np.nan, "p": np.nan}
    obs = float(z.loc[genes].mean())
    if match_on is not None:
        cov = match_on.reindex(z.index).fillna(match_on.median())
        bins = _bins(cov, n_bins)
        draws = _matched_random(rng, genes, bins, n_boot)
        zval = z.to_dict()
        null = np.array([np.mean([zval[g] for g in row]) for row in draws])
    else:
        arr = z.to_numpy()
        idx = rng.integers(0, len(arr), size=(n_boot, len(genes)))
        null = arr[idx].mean(axis=1)
    p = (1 + np.sum(null >= obs)) / (1 + n_boot)
    return {"n_genes": len(genes), "mean_z": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "effect_sd": float((obs - null.mean()) / (null.std() + 1e-12)),
            "p": float(p)}


def enrichment_table(gene_z: pd.Series, gene_sets: Mapping[str, Iterable[str]], **kw) -> pd.DataFrame:
    """Run `bootstrap_gene_z_enrichment` for every gene set and add BH-FDR."""
    rows = {name: bootstrap_gene_z_enrichment(gene_z, gs, **kw) for name, gs in gene_sets.items()}
    df = pd.DataFrame(rows).T
    df["fdr"] = bh_fdr(df["p"].to_numpy())
    return df.sort_values("p")


# ------------------------------------------------------------------------------- cell type -> GWAS
def ewce_bootstrap(spec: pd.DataFrame, hit_genes: Iterable[str], n_boot: int = 10_000,
                   expr_level: Optional[pd.Series] = None, n_bins: int = 20, seed: int = 0,
                   min_hits: int = 20) -> pd.DataFrame:
    """EWCE-style bootstrap: for every cell type, mean specificity of hit genes vs random gene lists.

    `expr_level` (mean expression across all cells, indexed by gene) enables expression-matched
    sampling as recommended by Skene et al. 2018; without it sampling is uniform over genes.
    Returns per cell type: n_hits, mean_spec, fold_change, sd_from_mean, p, fdr.
    """
    rng = np.random.default_rng(seed)
    hits = [g for g in dict.fromkeys(hit_genes) if g in spec.columns]
    out = pd.DataFrame(index=spec.index, columns=["n_hits", "mean_spec", "fold_change", "sd_from_mean", "p"],
                       dtype=float)
    if len(hits) < min_hits:
        out["n_hits"] = len(hits)
        out["fdr"] = np.nan
        return out
    obs = spec[hits].mean(axis=1)  # per cell type
    if expr_level is not None:
        cov = expr_level.reindex(spec.columns).fillna(expr_level.median())
        bins = _bins(cov, n_bins)
        draws = _matched_random(rng, hits, bins, n_boot)
        col_pos = {g: i for i, g in enumerate(spec.columns)}
        S = spec.to_numpy()
        null = np.stack([S[:, [col_pos[g] for g in row]].mean(axis=1) for row in draws], axis=1)
    else:
        S = spec.to_numpy()
        idx = rng.integers(0, S.shape[1], size=(n_boot, len(hits)))
        null = np.stack([S[:, row].mean(axis=1) for row in idx], axis=1)  # (celltypes, n_boot)
    null_mean = null.mean(axis=1)
    null_sd = null.std(axis=1) + 1e-12
    p = (1 + (null >= obs.to_numpy()[:, None]).sum(axis=1)) / (1 + n_boot)
    out["n_hits"] = len(hits)
    out["mean_spec"] = obs
    out["fold_change"] = obs / null_mean
    out["sd_from_mean"] = (obs - null_mean) / null_sd
    out["p"] = p
    out["fdr"] = bh_fdr(p)
    return out.sort_values("p")


# ------------------------------------------------------------------------------- spatial domains
def spatial_permutation(pb_cells_fn, celltype: np.ndarray, domain: np.ndarray, gene_z: pd.Series,
                        target_celltype: str, target_domains: Sequence[str], n_perm: int = 1000,
                        q: float = 0.9, seed: int = 0) -> Dict[str, float]:
    """Domain-restricted enrichment with a within-cell-type label permutation null.

    `pb_cells_fn(mask) -> pd.Series` must return the mean expression (indexed by gene) over the cells
    selected by the boolean `mask`. The statistic is the mean gene Z of the top-decile genes of
    `target_celltype` when specificity is computed from cells of that type *inside* `target_domains`
    relative to the same type *outside*. Domain labels are permuted within the cell type so cell-type
    composition is preserved (H2 in the README).
    """
    rng = np.random.default_rng(seed)
    ct_mask = celltype == target_celltype
    dom = np.asarray(domain)
    in_dom = np.isin(dom, list(target_domains))

    def stat(in_mask: np.ndarray) -> float:
        inside = pb_cells_fn(ct_mask & in_mask)
        outside = pb_cells_fn(ct_mask & ~in_mask)
        tot = inside + outside
        spec = (inside / tot.replace(0, np.nan)).dropna()
        top = spec[spec >= spec.quantile(q)].index
        top = [g for g in top if g in gene_z.index]
        return float(gene_z.loc[top].mean()) if top else np.nan

    obs = stat(in_dom)
    idx_ct = np.flatnonzero(ct_mask)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = in_dom.copy()
        perm[idx_ct] = rng.permutation(in_dom[idx_ct])
        null[k] = stat(perm)
    null = null[~np.isnan(null)]
    p = (1 + np.sum(null >= obs)) / (1 + len(null)) if len(null) else np.nan
    return {"obs": obs, "null_mean": float(null.mean()) if len(null) else np.nan,
            "null_sd": float(null.std()) if len(null) else np.nan, "p": float(p), "n_perm": int(len(null))}


# ------------------------------------------------------------------------------- comparisons
def concordance(a: pd.Series, b: pd.Series, top_k: int = 10) -> Dict[str, float]:
    """Spearman correlation of two −log10(p) (or effect) rankings and Jaccard of their top-k sets."""
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return {"n": len(common), "spearman": np.nan, "jaccard_topk": np.nan}
    rho = stats.spearmanr(a.loc[common], b.loc[common]).correlation
    ta = set(a.loc[common].sort_values(ascending=False).head(top_k).index)
    tb = set(b.loc[common].sort_values(ascending=False).head(top_k).index)
    return {"n": int(len(common)), "spearman": float(rho), "jaccard_topk": len(ta & tb) / len(ta | tb)}


def resolution_curve(results_by_level: Mapping[str, pd.DataFrame], p_col: str = "p",
                     alpha: float = 0.05) -> pd.DataFrame:
    """Summarise enrichment across taxonomy levels: #tests, #FDR-significant, best −log10 p."""
    rows = []
    for lvl, df in results_by_level.items():
        p = df[p_col].astype(float)
        fdr = bh_fdr(p.to_numpy())
        rows.append({"level": lvl, "n_tests": int(p.notna().sum()), "n_sig_fdr": int(np.nansum(fdr < alpha)),
                     "best_neglog10p": float(-np.log10(np.nanmin(p))) if p.notna().any() else np.nan})
    return pd.DataFrame(rows).set_index("level")
