"""Cortical depth (layering) from WM / pial distance maps.

* Equidistant depth: ``d_wm / (d_wm + d_pial)`` (0 at WM boundary, 1 at pial surface).
* Equivolume depth (Bok, 1929; Waehnert et al., 2014, NeuroImage): the depth fraction that
  encloses a fixed fraction of cortical *volume* between the boundaries, which depends on local
  curvature. For a locally annular cortex with WM radius ``R`` and thickness ``T`` the
  equivolume fraction at radius ``r`` is ``(r² - R²) / ((R+T)² - R²)`` on a gyral crown
  (WM inside), and the mirrored expression on a sulcal fundus (WM outside).

The production reference is LayNii ``LN2_LAYERS -equivol``; this module reproduces the
principle on phantoms and small ROIs so that analysis choices can be simulated.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def equidistant_depth(d_wm: np.ndarray, d_pial: np.ndarray) -> np.ndarray:
    """Depth fraction in [0, 1] from distances to the WM and pial boundaries."""
    d_wm, d_pial = np.asarray(d_wm, float), np.asarray(d_pial, float)
    tot = d_wm + d_pial
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, d_wm / tot, np.nan)


def equivolume_depth(d_wm: np.ndarray, d_pial: np.ndarray, curvature: np.ndarray, max_radius: float = 1e6) -> np.ndarray:
    """Equivolume depth fraction given signed local curvature (1/voxels).

    ``curvature > 0``: gyral crown (WM boundary is the inner surface of the annulus).
    ``curvature < 0``: sulcal fundus (WM boundary is the outer surface). ``0`` reduces to equidistant.
    """
    d_wm, d_pial, k = (np.asarray(v, float) for v in (d_wm, d_pial, curvature))
    T = d_wm + d_pial
    R = np.where(np.abs(k) > 1.0 / max_radius, 1.0 / np.maximum(np.abs(k), 1e-12), max_radius)
    out = np.full(d_wm.shape, np.nan)
    ok = T > 0
    # gyrus: r_wm = R, r_pial = R + T, r = R + d_wm
    gy = ok & (k > 0)
    r, rw, rp = R + d_wm, R, R + T
    out[gy] = ((r**2 - rw**2) / (rp**2 - rw**2))[gy]
    # sulcus: r_wm = R, r_pial = R - T (clamped), r = R - d_wm
    su = ok & (k < 0)
    rp_s = np.maximum(R - T, 1e-6)
    r_s = np.maximum(R - d_wm, 1e-6)
    out[su] = ((R**2 - r_s**2) / (R**2 - rp_s**2))[su]
    fl = ok & (k == 0)
    out[fl] = (d_wm / T)[fl]
    return np.clip(out, 0.0, 1.0)


def assign_bins(depth: np.ndarray, n_bins: int) -> np.ndarray:
    """Integer depth bins 1..n_bins (0 where depth is NaN); bin 1 = deepest (WM side)."""
    depth = np.asarray(depth, float)
    b = np.floor(np.nan_to_num(depth, nan=-1.0) * n_bins).astype(int) + 1
    b[np.isnan(depth)] = 0
    b[depth >= 1.0] = n_bins
    return b


def distances_from_masks(gm: np.ndarray, wm: np.ndarray, pial_outside: np.ndarray | None = None,
                         spacing: tuple[float, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean distances (voxels or mm) from each GM voxel to the WM mask and to the outside (CSF)."""
    gm, wm = np.asarray(gm, bool), np.asarray(wm, bool)
    outside = ~(gm | wm) if pial_outside is None else np.asarray(pial_outside, bool)
    d_wm = ndimage.distance_transform_edt(~wm, sampling=spacing)
    d_pial = ndimage.distance_transform_edt(~outside, sampling=spacing)
    d_wm = np.where(gm, d_wm, np.nan)
    d_pial = np.where(gm, d_pial, np.nan)
    return d_wm, d_pial


def annulus_phantom(size: int = 121, r_wm: float = 20.0, r_pial: float = 35.0) -> dict[str, np.ndarray]:
    """2-D gyral-crown phantom: WM disc of radius ``r_wm``, GM annulus up to ``r_pial``.

    Returns masks, distance maps, the analytic curvature (1/r_wm) and the analytic depths.
    """
    y, x = np.indices((size, size)).astype(float)
    c = (size - 1) / 2.0
    r = np.hypot(x - c, y - c)
    wm = r <= r_wm
    gm = (r > r_wm) & (r <= r_pial)
    d_wm, d_pial = distances_from_masks(gm, wm)
    curv = np.where(gm, 1.0 / r_wm, 0.0)
    eqd_true = np.where(gm, (r - r_wm) / (r_pial - r_wm), np.nan)
    eqv_true = np.where(gm, (r**2 - r_wm**2) / (r_pial**2 - r_wm**2), np.nan)
    return {"gm": gm, "wm": wm, "radius": r, "d_wm": d_wm, "d_pial": d_pial, "curvature": curv,
            "equidist_true": eqd_true, "equivol_true": eqv_true}


def bin_volume_fractions(bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Fraction of GM voxels per bin (equivolume layering should give ≈ equal fractions)."""
    counts = np.bincount(bins[bins > 0].ravel(), minlength=n_bins + 1)[1:]
    return counts / counts.sum()
