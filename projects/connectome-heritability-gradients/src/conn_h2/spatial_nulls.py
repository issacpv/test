"""Spatial null models for comparing regional heritability maps with gene
expression and other annotations.

* :func:`spin_permutations` - parcel-level spin test (Alexander-Bloch et al.,
  2018, NeuroImage; Váša et al., 2018, Cereb Cortex): random rotations of
  parcel centroids on the sphere, mirrored across hemispheres, nearest-
  centroid reassignment.
* :func:`spatial_correlation_test` - observed correlation vs. spun nulls.
* :func:`random_gene_set_null` - Fulcher et al. (2021, Nat Commun) style
  ensemble null for gene-set scores: compares a gene set's spatial
  correlation with the map against random gene sets of the same size.
* :func:`fetch_expression` - thin abagen wrapper (optional dependency).

Coordinates are expected as spherical-surface centroids (e.g. from the
fsaverage / fsLR ``sphere`` surfaces), *not* volumetric MNI coordinates.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# Spins
# --------------------------------------------------------------------------- #
def spin_permutations(
    coords_lh: np.ndarray, coords_rh: np.ndarray, n_perm: int = 1000, seed: int = 0
) -> np.ndarray:
    """Generate ``(n_perm, n_lh + n_rh)`` index arrays of spun parcel assignments.

    A random rotation is applied to the left-hemisphere sphere; the mirrored
    rotation (x -> -x) is applied to the right hemisphere so that the two
    hemispheres rotate symmetrically. Each original parcel is mapped to the
    nearest rotated centroid (with replacement, so some parcels may be
    duplicated - the standard 'vasa'/'spin' variant for parcellated data).
    """
    L = np.asarray(coords_lh, dtype=float)
    R = np.asarray(coords_rh, dtype=float)
    rng = np.random.default_rng(seed)
    n_l = len(L)
    out = np.empty((n_perm, n_l + len(R)), dtype=int)
    tree_l = cKDTree(L)
    tree_r = cKDTree(R)
    flip = np.diag([-1.0, 1.0, 1.0])
    for p in range(n_perm):
        rot = Rotation.random(random_state=rng.integers(0, 2**32 - 1)).as_matrix()
        rot_r = flip @ rot @ flip
        _, idx_l = tree_l.query(L @ rot.T)
        _, idx_r = tree_r.query(R @ rot_r.T)
        out[p, :n_l] = idx_l
        out[p, n_l:] = idx_r + n_l
    return out


def spherical_coords_from_labels(labels: np.ndarray, sphere_vertices: np.ndarray) -> Dict[int, np.ndarray]:
    """Parcel centroids on a sphere: mean vertex coordinate projected back to the sphere."""
    labels = np.asarray(labels)
    V = np.asarray(sphere_vertices, dtype=float)
    radius = np.linalg.norm(V, axis=1).mean()
    centroids = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        c = V[labels == lab].mean(axis=0)
        centroids[int(lab)] = c / np.linalg.norm(c) * radius
    return centroids


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(a[ok], b[ok]).correlation)
    return float(stats.pearsonr(a[ok], b[ok])[0])


def spatial_correlation_test(
    map_a: np.ndarray, map_b: np.ndarray, spins: np.ndarray, method: str = "spearman"
) -> Dict[str, float]:
    """Correlation of two parcel maps with a spin-based null (map_a is spun).

    p-value is two-sided, ``(k + 1) / (n_perm + 1)``.
    """
    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    obs = _corr(a, b, method)
    null = np.array([_corr(a[idx], b, method) for idx in spins])
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {"r": obs, "p_spin": float(p), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "n_null": int(len(null))}


def random_gene_set_null(
    map_a: np.ndarray,
    expression: pd.DataFrame,
    gene_set: list,
    n_perm: int = 1000,
    seed: int = 0,
    method: str = "spearman",
    spins: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Gene-set score = mean z-scored expression across the set's genes.

    Two nulls are computed: (i) random gene sets of the same size (controls
    for the generic spatial structure of expression, Fulcher et al., 2021);
    (ii) optionally spins of the brain map (controls for spatial
    autocorrelation). Report both; a finding should survive both.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(map_a, dtype=float)
    E = expression.copy()
    E = (E - E.mean(axis=0)) / E.std(axis=0, ddof=1)
    genes = [g for g in gene_set if g in E.columns]
    if len(genes) == 0:
        raise ValueError("no genes from gene_set are present in expression")
    score = E[genes].mean(axis=1).to_numpy()
    obs = _corr(a, score, method)
    all_genes = np.asarray(E.columns)
    null_genes = []
    for _ in range(n_perm):
        pick = rng.choice(all_genes, size=len(genes), replace=False)
        null_genes.append(_corr(a, E[pick].mean(axis=1).to_numpy(), method))
    null_genes = np.asarray(null_genes)
    p_genes = (np.sum(np.abs(null_genes) >= abs(obs)) + 1) / (n_perm + 1)
    out = {"r": obs, "p_random_genes": float(p_genes), "n_genes_used": len(genes)}
    if spins is not None:
        out["p_spin"] = spatial_correlation_test(a, score, spins, method)["p_spin"]
    return out


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


# --------------------------------------------------------------------------- #
# abagen wrapper
# --------------------------------------------------------------------------- #
def fetch_expression(atlas, atlas_info=None, **kwargs) -> pd.DataFrame:
    """Regional AHBA gene expression via abagen (Markello et al., 2021, eLife).

    Parameters
    ----------
    atlas
        Path to a volumetric/surface parcellation, or a tuple of GIFTI label
        files for fsaverage/fsLR surfaces (see ``abagen.get_expression_data``).
    atlas_info
        Optional CSV with ``id, label, hemisphere, structure``.
    kwargs
        Passed to ``abagen.get_expression_data`` (e.g. ``lr_mirror='bidirectional'``,
        ``missing='interpolate'``, ``norm_matched=True``).

    The first call downloads the AHBA microarray data (~4 GB) to
    ``~/abagen-data`` (override with ``ABAGEN_DATA``).
    """
    try:
        import abagen  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("abagen not installed: pip install abagen") from exc
    return abagen.get_expression_data(atlas, atlas_info=atlas_info, **kwargs)
