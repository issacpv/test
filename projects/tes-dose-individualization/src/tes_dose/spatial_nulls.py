"""Null models for E-field to behaviour maps.

Two complementary nulls are implemented:

* :func:`spin_test_correlation` - a spatial null for *map-level* claims: one
  map is rotated on the sphere (Alexander-Bloch et al., 2018) and re-assigned
  by nearest neighbour, preserving spatial autocorrelation.
* :func:`montage_null` - a subject-level null for *dose-response maps*: the
  behavioural labels are permuted across subjects while each subject's E-field
  map is kept intact, so the null preserves the (montage-driven) common shape
  of the maps and only breaks the subject-to-outcome link.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(pearsonr(x, y)[0])
    if method == "spearman":
        return float(spearmanr(x, y)[0])
    raise ValueError("method must be 'pearson' or 'spearman'")


def spin_permutations(coords: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Nearest-neighbour vertex permutations from random rotations of spherical coordinates.

    Parameters
    ----------
    coords : (V, 3) spherical coordinates (e.g. fsaverage sphere vertices).
    n_perm : number of rotations.
    rng : numpy random generator.

    Returns
    -------
    (n_perm, V) integer array; row k maps each vertex to its rotated neighbour.
    """
    coords = np.asarray(coords, float)
    unit = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    tree = cKDTree(unit)
    perms = np.empty((n_perm, len(unit)), dtype=int)
    rots = Rotation.random(n_perm, random_state=int(rng.integers(0, 2**31 - 1)))
    for k in range(n_perm):
        rotated = rots[k].apply(unit)
        perms[k] = tree.query(rotated, k=1)[1]
    return perms


def spin_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    n_perm: int = 1000,
    method: str = "pearson",
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Correlate two surface maps with a spin-test null.

    Returns ``(r_observed, p_spin, null_distribution)``; ``p_spin`` is two-sided
    with the +1 correction of Phipson & Smyth.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    r_obs = _corr(x, y, method)
    perms = spin_permutations(coords, n_perm, rng)
    null = np.array([_corr(x[p], y, method) for p in perms])
    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, float(p), null


def vertexwise_dose_correlation(efield_maps: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Per-vertex Pearson correlation between subjects' |E| and their behavioural effect.

    Parameters
    ----------
    efield_maps : (S, V) |E| per subject and vertex.
    effect : (S,) behavioural effect per subject.
    """
    e = np.asarray(efield_maps, float)
    b = np.asarray(effect, float)
    ez = (e - e.mean(0)) / (e.std(0) + 1e-12)
    bz = (b - b.mean()) / (b.std() + 1e-12)
    return (ez * bz[:, None]).mean(0)


def montage_null(
    efield_maps: np.ndarray,
    effect: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Subject-permutation null for a vertex-wise dose-response map with max-statistic correction.

    Returns ``(r_map, p_fwe_map, threshold)`` where ``p_fwe_map`` is the
    family-wise-error-corrected p per vertex (max |r| over vertices under the
    null) and ``threshold`` is the 95th percentile of the max-|r| null.
    """
    rng = np.random.default_rng(seed)
    r_map = vertexwise_dose_correlation(efield_maps, effect)
    eff = np.asarray(effect, float)
    max_null = np.empty(n_perm)
    for k in range(n_perm):
        max_null[k] = np.abs(vertexwise_dose_correlation(efield_maps, rng.permutation(eff))).max()
    p_fwe = (np.array([(max_null >= abs(r)).sum() for r in r_map]) + 1) / (n_perm + 1)
    return r_map, p_fwe, float(np.percentile(max_null, 95))


def survival_fraction(p_naive: np.ndarray, p_null: np.ndarray, alpha: float = 0.05) -> float:
    """Fraction of naive-significant vertices that remain significant under the corrected null (H3)."""
    naive = np.asarray(p_naive) < alpha
    if naive.sum() == 0:
        return float("nan")
    return float(((np.asarray(p_null) < alpha) & naive).sum() / naive.sum())
