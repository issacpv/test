"""Regional aggregation of cell-level (MERFISH) expression and composition decomposition.

The functions operate on plain pandas/numpy objects so they can be tested with synthetic
data; the loaders that turn ABC Atlas files into these objects are thin wrappers at the
bottom of the module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def normalize_counts(counts: np.ndarray, target_sum: float = 1e4, log: bool = True) -> np.ndarray:
    """Per-cell library-size normalisation followed by log1p.

    Parameters
    ----------
    counts : (n_cells, n_genes) array of raw counts.
    target_sum : counts are scaled so each cell sums to this value.
    log : apply ``log1p`` after scaling.
    """
    counts = np.asarray(counts, dtype=float)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    x = counts / lib * target_sum
    return np.log1p(x) if log else x


def aggregate_cells_to_regions(
    expr: np.ndarray,
    cell_meta: pd.DataFrame,
    genes: Sequence[str],
    region_col: str = "parcellation_structure",
    min_cells: int = 100,
    statistic: str = "mean",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Average normalised expression per region.

    Returns
    -------
    region_expr : DataFrame (regions x genes) of mean (or median) expression.
    n_cells : Series with the number of cells per retained region.
    Regions with fewer than ``min_cells`` cells are dropped.
    """
    if expr.shape[0] != len(cell_meta):
        raise ValueError("expr rows must match cell_meta rows")
    if expr.shape[1] != len(genes):
        raise ValueError("expr columns must match genes")
    labels = cell_meta[region_col].astype(str).to_numpy()
    df = pd.DataFrame(np.asarray(expr, dtype=float), columns=list(genes))
    df[region_col] = labels
    grp = df.groupby(region_col)
    counts = grp.size()
    keep = counts[counts >= min_cells].index
    agg = grp.mean() if statistic == "mean" else grp.median()
    agg = agg.loc[keep]
    return agg, counts.loc[keep]


def composition_matrix(
    cell_meta: pd.DataFrame,
    region_col: str = "parcellation_structure",
    type_col: str = "subclass",
    min_cells: int = 100,
) -> pd.DataFrame:
    """Region x cell-type matrix of proportions (rows sum to 1)."""
    ct = pd.crosstab(cell_meta[region_col].astype(str), cell_meta[type_col].astype(str))
    ct = ct[ct.sum(axis=1) >= min_cells]
    return ct.div(ct.sum(axis=1), axis=0)


@dataclass
class CompositionDecomposition:
    """Result of regressing regional gene maps on cell-type proportions."""

    fitted: pd.DataFrame        # composition-explained part (regions x genes)
    residual: pd.DataFrame      # within-type part (regions x genes)
    r2: pd.Series               # per-gene cross-validated R^2 of the composition model
    alpha: float                # ridge penalty used


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with intercept handled by centring. Returns coefficients (p x q)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    p = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(p)
    return np.linalg.solve(A, Xc.T @ Yc)


def decompose_composition(
    region_expr: pd.DataFrame,
    composition: pd.DataFrame,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
    seed: int = 0,
) -> CompositionDecomposition:
    """Split each gene's regional profile into a composition-predicted part and a residual.

    A single ridge penalty is chosen by K-fold cross-validation over regions (pooled over
    genes), then the model is refit on all regions. Cross-validated R^2 per gene is
    reported so that "explained by composition" is not an in-sample statement.
    """
    common = region_expr.index.intersection(composition.index)
    if len(common) < 5:
        raise ValueError("need at least 5 shared regions")
    X = composition.loc[common].to_numpy(float)
    Y = region_expr.loc[common].to_numpy(float)
    n = len(common)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    def cv_pred(alpha: float) -> np.ndarray:
        pred = np.zeros_like(Y)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            B = _ridge_fit(X[tr], Y[tr], alpha)
            pred[te] = (X[te] - X[tr].mean(0)) @ B + Y[tr].mean(0)
        return pred

    best_alpha, best_score, best_pred = None, -np.inf, None
    for a in alphas:
        pred = cv_pred(a)
        ss_res = ((Y - pred) ** 2).sum()
        ss_tot = ((Y - Y.mean(0)) ** 2).sum()
        score = 1 - ss_res / ss_tot
        if score > best_score:
            best_alpha, best_score, best_pred = a, score, pred
    assert best_pred is not None and best_alpha is not None
    ss_res_g = ((Y - best_pred) ** 2).sum(0)
    ss_tot_g = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = pd.Series(1 - ss_res_g / np.where(ss_tot_g == 0, np.nan, ss_tot_g), index=region_expr.columns)
    B = _ridge_fit(X, Y, best_alpha)
    fitted = (X - X.mean(0)) @ B + Y.mean(0)
    fitted_df = pd.DataFrame(fitted, index=common, columns=region_expr.columns)
    resid_df = region_expr.loc[common] - fitted_df
    return CompositionDecomposition(fitted=fitted_df, residual=resid_df, r2=r2, alpha=float(best_alpha))


def zscore_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene across regions (columns), guarding constant columns."""
    sd = df.std(axis=0, ddof=1).replace(0, np.nan)
    return (df - df.mean(axis=0)) / sd


def region_centroids(cell_meta: pd.DataFrame, region_col: str = "parcellation_structure",
                     coord_cols: Sequence[str] = ("x_ccf", "y_ccf", "z_ccf")) -> pd.DataFrame:
    """Centroid (mm) of each region's cells in CCF coordinates, used for spatial nulls."""
    return cell_meta.groupby(cell_meta[region_col].astype(str))[list(coord_cols)].mean()


# ------------------------------------------------------------------ ABC Atlas loaders
def load_abc_cell_metadata(abc_dir: Path, dataset: str = "MERFISH-C57BL6J-638850",
                           release: str = "20230830") -> pd.DataFrame:
    """Join the cluster-annotated cell metadata with CCF coordinates and parcellation labels.

    Expects the layout produced by ``scripts/download_data.py`` (see data/README.md).
    """
    meta_dir = abc_dir / "metadata" / dataset / release
    ccf_dir = abc_dir / "metadata" / f"{dataset}-CCF" / release
    cells = pd.read_csv(meta_dir / "views" / "cell_metadata_with_cluster_annotation.csv", index_col="cell_label")
    parc = pd.read_csv(ccf_dir / "views" / "cell_metadata_with_parcellation_annotation.csv", index_col="cell_label")
    parc_cols = [c for c in parc.columns if c.startswith("parcellation_") or c.endswith("_ccf")]
    return cells.join(parc[parc_cols], how="inner")


def load_abc_expression(h5ad_path: Path, cell_labels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, list]:
    """Load a cell-by-gene h5ad (optionally subset to ``cell_labels``); returns (dense array, gene symbols)."""
    import anndata as ad  # local import: optional dependency

    adata = ad.read_h5ad(h5ad_path, backed="r")
    if cell_labels is not None:
        adata = adata[list(cell_labels)]
    X = adata.to_memory().X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    genes = list(adata.var["gene_symbol"]) if "gene_symbol" in adata.var else list(adata.var_names)
    return X, genes


def ish_energy_matrix(ish_csv: Path, plane_preference: str = "coronal") -> pd.DataFrame:
    """Gene x structure expression-energy matrix from the script's ISH CSV (coronal preferred)."""
    df = pd.read_csv(ish_csv)
    pref = df[df["plane"].str.lower() == plane_preference]
    fallback = df[~df["gene"].isin(pref["gene"].unique())]
    use = pd.concat([pref, fallback])
    return use.pivot_table(index="gene", columns="structure_id", values="expression_energy", aggfunc="mean")
