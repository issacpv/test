"""Parcel-level lesion network maps and symptom association.

Functional connectome representation: ``fc_vp`` is an ``(n_voxels, n_parcels)`` matrix of
Fisher-z connectivity between each grid voxel (flattened index of the 3-D grid, brain voxels
only or all voxels) and each parcel. A lesion network map is the mean over lesion voxels.

Structural representation: ``tract_voxels`` is a list of voxel-index arrays visited by each
template tract/bundle and ``tract_parcels`` an ``(n_tracts, n_parcels)`` 0/1 matrix of which
parcels each tract connects; a parcel's disconnection is the fraction of its tracts that pass
through the lesion.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def flat_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, bool).ravel())


def lesion_network_map(mask: np.ndarray, fc_vp: np.ndarray, voxel_index: np.ndarray | None = None) -> np.ndarray:
    """Mean voxel-to-parcel connectivity over the lesion voxels (functional LNM seed map).

    ``voxel_index`` maps flat grid index -> row of ``fc_vp`` (-1 for voxels without a row);
    when ``None`` the rows of ``fc_vp`` are assumed to be in flat grid order.
    """
    flat = flat_indices(mask)
    rows = flat if voxel_index is None else voxel_index[flat]
    rows = rows[rows >= 0]
    if rows.size == 0:
        return np.full(fc_vp.shape[1], np.nan)
    return fc_vp[rows].mean(axis=0)


def lesion_network_maps(lesions: list[np.ndarray], fc_vp: np.ndarray,
                        voxel_index: np.ndarray | None = None) -> np.ndarray:
    """Stack of maps, ``(n_lesions, n_parcels)``."""
    return np.stack([lesion_network_map(m, fc_vp, voxel_index) for m in lesions])


def disconnection_map(mask: np.ndarray, tract_voxels: list[np.ndarray], tract_parcels: np.ndarray) -> np.ndarray:
    """Fraction of each parcel's tracts intersected by the lesion (structural LNM)."""
    flat = set(flat_indices(mask).tolist())
    hit = np.array([bool(flat.intersection(tv.tolist())) for tv in tract_voxels], float)
    n_tracts = tract_parcels.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (hit @ tract_parcels) / n_tracts
    return np.where(n_tracts > 0, frac, 0.0)


def sensitivity_map(maps: np.ndarray, symptomatic: np.ndarray, threshold: float) -> np.ndarray:
    """Classic LNM 'sensitivity': fraction of symptomatic lesions whose map exceeds ``threshold``."""
    s = maps[np.asarray(symptomatic, bool)] > threshold
    return s.mean(axis=0)


def symptom_association(maps: np.ndarray, scores: np.ndarray, covariates: np.ndarray | None = None) -> np.ndarray:
    """Per-parcel t statistic of ``scores ~ map_value + covariates`` (regression LNM).

    Vectorised OLS across parcels; returns t for the map-value coefficient. NaN columns give NaN.
    """
    maps = np.asarray(maps, float)
    y = np.asarray(scores, float)
    n, P = maps.shape
    C = np.ones((n, 1)) if covariates is None else np.column_stack([np.ones(n), np.asarray(covariates, float)])
    # residualise y and each map column on covariates (Frisch-Waugh-Lovell)
    Q, _ = np.linalg.qr(C)
    proj = lambda v: v - Q @ (Q.T @ v)
    yr = proj(y)
    Mr = proj(maps)
    ssx = np.sum(Mr**2, axis=0)
    beta = (Mr.T @ yr) / np.where(ssx > 0, ssx, np.nan)
    resid = yr[:, None] - Mr * beta[None, :]
    dof = n - C.shape[1] - 1
    sigma2 = np.sum(resid**2, axis=0) / dof
    se = np.sqrt(sigma2 / np.where(ssx > 0, ssx, np.nan))
    return beta / se


def t_to_p(t: np.ndarray, dof: int) -> np.ndarray:
    return 2 * sps.t.sf(np.abs(t), dof)


def max_stat_threshold(null_max: np.ndarray, alpha: float = 0.05) -> float:
    """FWER-controlling critical value from the null distribution of the maximum |t|."""
    return float(np.quantile(np.asarray(null_max), 1 - alpha))
