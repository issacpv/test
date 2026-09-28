"""Parcellated Allen Human Brain Atlas expression via abagen.

``abagen`` (Markello et al., 2021, *eLife*) downloads the six-donor
microarray data (~4 GB) and implements the processing choices reviewed in
Arnatkeviciute et al. (2019, *NeuroImage*).  This module wraps it with the
defaults recommended there and adds a pure-numpy differential-stability
filter (Hawrylycz et al., 2015, *Nat Neurosci*) so that gene filtering can be
audited and tested without the atlas present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

PathLike = Union[str, Path]

ABAGEN_DEFAULTS = dict(
    ibf_threshold=0.5,               # intensity-based filtering of probes
    probe_selection="diff_stability",
    lr_mirror="bidirectional",       # mirror samples across hemispheres
    missing="interpolate",           # fill parcels with no samples from nearest samples
    tolerance=2,                     # mm sample-to-parcel matching
    sample_norm="srs",
    gene_norm="srs",
    norm_matched=True,
    region_agg="donors",
    agg_metric="mean",
)


def fetch_parcellated_expression(
    atlas: PathLike,
    atlas_info: Optional[PathLike] = None,
    data_dir: Optional[PathLike] = None,
    donors: Union[str, Sequence[str]] = "all",
    return_donors: bool = False,
    cache: Optional[PathLike] = None,
    **overrides,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Regions x genes expression for a parcellation using abagen.

    Parameters
    ----------
    atlas
        Volumetric NIfTI or a tuple of (lh, rh) GIFTI label files; for the
        Glasser MMP1.0 use the fsaverage/fsLR label GIFTIs from BALSA or
        ``neuromaps``; for Schaefer use ``abagen.fetch_desikan_killiany``-style
        helpers or ``nilearn.datasets.fetch_atlas_schaefer_2018``.
    atlas_info
        CSV with ``id, label, hemisphere, structure`` (required for surface
        atlases; see abagen docs).
    return_donors
        If True, returns ``{donor: DataFrame}`` (needed for differential
        stability); otherwise a single donor-averaged DataFrame.
    cache
        Optional ``.parquet``/``.csv`` path to reuse a previous download.
    overrides
        Any ``abagen.get_expression_data`` keyword (see :data:`ABAGEN_DEFAULTS`).
    """
    if cache is not None and Path(cache).exists() and not return_donors:
        return pd.read_parquet(cache) if str(cache).endswith(".parquet") else pd.read_csv(cache, index_col=0)
    try:
        import abagen  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install abagen  (downloads AHBA microarray on first use)") from e
    kwargs = {**ABAGEN_DEFAULTS, **overrides}
    expr = abagen.get_expression_data(
        atlas, atlas_info=atlas_info, data_dir=data_dir, donors=donors, return_donors=return_donors, **kwargs
    )
    if cache is not None and not return_donors:
        if str(cache).endswith(".parquet"):
            expr.to_parquet(cache)
        else:
            expr.to_csv(cache)
    return expr


def differential_stability(donor_expr: Dict[str, pd.DataFrame], min_regions: int = 10) -> pd.Series:
    """Mean pairwise Spearman correlation of each gene's regional profile across donors.

    Parameters
    ----------
    donor_expr
        ``{donor: regions x genes}`` with identical columns; regions missing
        in a donor are NaN and are dropped pairwise.
    min_regions
        Minimum number of regions shared by a donor pair for the correlation
        to count.

    Returns
    -------
    pd.Series indexed by gene (DS in [-1, 1]); genes with no valid donor pair
    are NaN.  Keep e.g. the top 50 % or DS > 0.1 (Arnatkeviciute et al., 2019).
    """
    donors = list(donor_expr)
    genes = donor_expr[donors[0]].columns
    acc = np.zeros(len(genes))
    cnt = np.zeros(len(genes))
    for i in range(len(donors)):
        for j in range(i + 1, len(donors)):
            a, b = donor_expr[donors[i]].align(donor_expr[donors[j]], join="inner", axis=0)
            if len(a) < min_regions:
                continue
            for g_idx, g in enumerate(genes):
                x, y = a[g].to_numpy(dtype=float), b[g].to_numpy(dtype=float)
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() < min_regions or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    continue
                r = stats.spearmanr(x[ok], y[ok]).statistic
                if not np.isnan(r):
                    acc[g_idx] += r
                    cnt[g_idx] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        ds = np.where(cnt > 0, acc / cnt, np.nan)
    return pd.Series(ds, index=genes, name="differential_stability")


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score (per gene across regions), NaN-safe."""
    return (expr - expr.mean(axis=0)) / expr.std(axis=0, ddof=1)


def simulate_expression(
    coords: np.ndarray,
    n_genes: int = 500,
    n_spatial_genes: int = 100,
    length_scale: float = 30.0,
    noise: float = 0.5,
    seed: int = 0,
    n_donors: int = 1,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Synthetic regions x genes matrix with spatially autocorrelated genes.

    The first ``n_spatial_genes`` genes are Gaussian-process draws with an
    exponential kernel on ``coords`` (so they have spatial autocorrelation);
    the remainder are white noise.  With ``n_donors > 1`` each donor gets the
    same signal plus independent noise (for testing differential stability).
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    K = np.exp(-D / length_scale) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K)
    signal = np.hstack([L @ rng.normal(size=(n, n_spatial_genes)), np.zeros((n, n_genes - n_spatial_genes))])
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    regions = [f"R{i:03d}" for i in range(n)]

    def _one(k):
        e = signal + rng.normal(0, noise, size=(n, n_genes)) + (rng.normal(size=(n, n_genes)) if k == "noise_only" else 0)
        return pd.DataFrame(e, index=regions, columns=genes)

    if n_donors == 1:
        return _one(0)
    return {f"donor{d}": _one(d) for d in range(n_donors)}
