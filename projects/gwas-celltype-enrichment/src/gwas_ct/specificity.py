"""Cell-type specificity scores (EWCE-style), top-decile gene sets, hierarchy aggregation,
spatial-domain specificity and mouse->human orthology mapping.

Conventions: pseudobulk / specificity frames are (labels x genes) `pandas.DataFrame`s.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def filter_genes(pb: pd.DataFrame, min_mean: float = 0.0, min_labels_expressed: int = 1,
                 expr_threshold: float = 0.0) -> pd.DataFrame:
    """Drop genes with total mean expression <= `min_mean` or expressed (> threshold) in too few labels."""
    tot = pb.sum(axis=0)
    n_expr = (pb > expr_threshold).sum(axis=0)
    keep = (tot > min_mean) & (n_expr >= min_labels_expressed)
    return pb.loc[:, keep]


def specificity_matrix(pb: pd.DataFrame, pseudocount: float = 0.0) -> pd.DataFrame:
    """EWCE specificity: expression of gene g in label c divided by its sum over all labels.

    Rows sum to nothing in particular; *columns* (genes) sum to 1. Genes with zero total are dropped.
    """
    tot = pb.sum(axis=0) + pseudocount * pb.shape[0]
    keep = tot > 0
    spec = (pb.loc[:, keep] + pseudocount) / tot[keep]
    return spec


def top_decile_genes(spec: pd.DataFrame, label: str, q: float = 0.9, min_expr: Optional[pd.Series] = None,
                     min_expr_value: float = 0.0) -> List[str]:
    """Genes in the top (1-q) quantile of specificity for `label` (default: top decile).

    Optionally require `min_expr[g] > min_expr_value` (e.g. mean expression in that label) so that
    highly specific but essentially unexpressed genes are excluded, as MAGMA_Celltyping does.
    """
    s = spec.loc[label]
    if min_expr is not None:
        s = s[min_expr.reindex(s.index).fillna(0) > min_expr_value]
    thr = s.quantile(q)
    return list(s.index[s >= thr])


def all_top_decile_sets(spec: pd.DataFrame, q: float = 0.9, pb: Optional[pd.DataFrame] = None,
                        min_expr_value: float = 0.0) -> Dict[str, List[str]]:
    """Top-decile gene set for every label."""
    return {lab: top_decile_genes(spec, lab, q, None if pb is None else pb.loc[lab], min_expr_value)
            for lab in spec.index}


def specificity_quantiles(spec: pd.DataFrame, n_bins: int = 40) -> pd.DataFrame:
    """Bin specificity into `n_bins` quantile bins per label (MAGMA_Celltyping 'quantiles' style).

    Returns integer bins 0..n_bins (0 = not expressed / zero specificity) as a (labels x genes) frame.
    Genes with specificity 0 get bin 0; the remaining genes are ranked within the label.
    """
    out = pd.DataFrame(0, index=spec.index, columns=spec.columns, dtype=int)
    for lab in spec.index:
        s = spec.loc[lab]
        nz = s > 0
        if nz.sum() == 0:
            continue
        ranks = s[nz].rank(method="average")
        bins = np.ceil(ranks / ranks.max() * n_bins).astype(int).clip(1, n_bins)
        out.loc[lab, nz.index[nz]] = bins.values
    return out


def aggregate_hierarchy(pb_fine: pd.DataFrame, fine_to_coarse: Mapping[str, str],
                        n_cells: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Aggregate fine-level pseudobulk (e.g. clusters) to a coarser level (e.g. subclass).

    Cell-count weighted mean when `n_cells` is supplied (so that the result equals the mean over
    all cells of the coarse label); unweighted mean of fine labels otherwise.
    """
    coarse = pd.Series({k: fine_to_coarse.get(k) for k in pb_fine.index}).dropna()
    pb = pb_fine.loc[coarse.index]
    w = pd.Series(1.0, index=pb.index) if n_cells is None else pd.Series(n_cells).reindex(pb.index).fillna(0)
    num = (pb.mul(w, axis=0)).groupby(coarse.values).sum()
    den = w.groupby(coarse.values).sum()
    return num.div(den, axis=0)


def specificity_by_level(pb_cluster: pd.DataFrame, hier: pd.DataFrame,
                         levels: Sequence[str] = ("class", "subclass", "supertype", "cluster"),
                         n_cells_col: str = "n_cells") -> Dict[str, pd.DataFrame]:
    """Specificity matrices at every hierarchy level from a cluster-level pseudobulk.

    `hier` is indexed by cluster label and has one column per coarser level (see
    `abc_loader.hierarchy_table`).
    """
    out: Dict[str, pd.DataFrame] = {}
    n_cells = hier[n_cells_col] if n_cells_col in hier.columns else None
    for lvl in levels:
        if lvl == hier.index.name or lvl == levels[-1]:
            pb = pb_cluster
        else:
            pb = aggregate_hierarchy(pb_cluster, hier[lvl].to_dict(), None if n_cells is None else n_cells.to_dict())
        out[lvl] = specificity_matrix(pb)
    return out


def spatial_specificity(pb_combined: pd.DataFrame, sep: str = "|") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """From a pseudobulk over combined 'celltype|domain' labels return

    (i) specificity across *all* combined labels (cell type x domain), and
    (ii) within-cell-type specificity across domains (rows renormalised within each cell type),
        which isolates *where* a cell type expresses a gene more than its counterparts elsewhere.
    """
    spec_all = specificity_matrix(pb_combined)
    ct = pd.Series([s.split(sep, 1)[0] for s in pb_combined.index], index=pb_combined.index)
    within = pb_combined.copy()
    for c in ct.unique():
        rows = ct.index[ct == c]
        tot = pb_combined.loc[rows].sum(axis=0)
        tot[tot == 0] = np.nan
        within.loc[rows] = pb_combined.loc[rows] / tot
    return spec_all, within.fillna(0.0)


def map_orthologs(spec_mouse: pd.DataFrame, orth: pd.DataFrame, mouse_col: str = "mouse_symbol",
                  human_col: str = "human_symbol", one_to_one: bool = True) -> pd.DataFrame:
    """Rename mouse genes to human symbols using an orthology table; drops unmapped genes.

    With `one_to_one=True` any mouse or human symbol appearing more than once in the table is removed.
    """
    o = orth[[mouse_col, human_col]].dropna().drop_duplicates()
    if one_to_one:
        o = o[~o[mouse_col].duplicated(keep=False) & ~o[human_col].duplicated(keep=False)]
    m = dict(zip(o[mouse_col], o[human_col]))
    cols = [g for g in spec_mouse.columns if g in m]
    out = spec_mouse[cols].copy()
    out.columns = [m[g] for g in cols]
    return out


def match_labels_across_species(mouse_labels: Iterable[str], human_labels: Iterable[str],
                                mapping: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Pairs (mouse_label, human_label) present in both atlases according to a curated mapping."""
    hs = set(human_labels)
    return [(m, mapping[m]) for m in mouse_labels if m in mapping and mapping[m] in hs]
