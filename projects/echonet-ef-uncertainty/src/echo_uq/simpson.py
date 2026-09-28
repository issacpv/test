"""Ejection fraction from segmentation masks: Simpson's method of disks.

Given a binary LV-cavity mask in an apical view, the long axis is taken as the
mask's principal axis; the cavity is sliced into ``n_disks`` slabs of equal
height ``h`` along that axis and each slab is treated as a disk (single-plane)
or an ellipse whose two diameters come from the A4C and A2C views (biplane):

    V_single  = sum_i  pi * (w_i / 2)^2 * h
    V_biplane = sum_i  pi / 4 * w_i^A4C * w_i^A2C * h

Slab widths use the slab *area / h*, which is robust to pixelisation.  The
functions are pure numpy; ``pixel_spacing`` converts to mL when known
(EchoNet-Dynamic: 112 px frames, physical spacing unknown -> EF only; CAMUS
``.mhd`` headers give spacing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal as sps


@dataclass
class DiskVolume:
    volume: float
    length: float
    widths: np.ndarray
    n_disks: int


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (y, x) and unit long-axis direction (y, x) from the mask's second moments."""
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        raise ValueError("mask too small")
    pts = np.stack([ys, xs], axis=1).astype(float)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    return c, axis / np.linalg.norm(axis)


def slab_widths(mask: np.ndarray, n_disks: int = 20, axis: np.ndarray | None = None, centroid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Width of each of ``n_disks`` slabs along the long axis (area / slab height) and the axis length."""
    mask = largest_component(mask)
    if centroid is None or axis is None:
        centroid, axis = principal_axis(mask)
    ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs], axis=1).astype(float) - centroid
    proj = pts @ axis
    lo, hi = proj.min() - 0.5, proj.max() + 0.5
    length = hi - lo
    h = length / n_disks
    edges = lo + h * np.arange(n_disks + 1)
    idx = np.clip(((proj - lo) / h).astype(int), 0, n_disks - 1)
    areas = np.bincount(idx, minlength=n_disks).astype(float)
    return areas / h, float(length)


def disk_volume_single_plane(mask: np.ndarray, n_disks: int = 20, pixel_spacing: float = 1.0) -> DiskVolume:
    """Single-plane Simpson volume assuming circular cross-sections (units: pixel_spacing^3)."""
    w, L = slab_widths(mask, n_disks)
    h = L / n_disks
    vol = float(np.sum(np.pi * (w / 2.0) ** 2 * h)) * pixel_spacing ** 3
    return DiskVolume(vol, L * pixel_spacing, w * pixel_spacing, n_disks)


def disk_volume_biplane(mask_a4c: np.ndarray, mask_a2c: np.ndarray, n_disks: int = 20, pixel_spacing: float = 1.0) -> DiskVolume:
    """Biplane Simpson volume from A4C and A2C masks; slab height uses the longer of the two axes."""
    w4, L4 = slab_widths(mask_a4c, n_disks)
    w2, L2 = slab_widths(mask_a2c, n_disks)
    L = max(L4, L2)
    h = L / n_disks
    vol = float(np.sum(np.pi / 4.0 * w4 * w2 * h)) * pixel_spacing ** 3
    return DiskVolume(vol, L * pixel_spacing, np.sqrt(w4 * w2) * pixel_spacing, n_disks)


def ef_from_volumes(edv: float, esv: float) -> float:
    """EF in [0, 1]; NaN if EDV is not positive."""
    return float(1.0 - esv / edv) if edv > 0 else float("nan")


def area_curve(masks: np.ndarray) -> np.ndarray:
    """Per-frame cavity area (pixels) of a (T, H, W) mask stack."""
    return masks.reshape(masks.shape[0], -1).sum(axis=1).astype(float)


def select_ed_es(areas: np.ndarray) -> tuple[int, int]:
    """Global end-diastolic (max area) and end-systolic (min area) frames."""
    return int(np.argmax(areas)), int(np.argmin(areas))


def beat_segments(areas: np.ndarray, fps: float, min_hr: float = 40.0, max_hr: float = 180.0) -> list[tuple[int, int, int]]:
    """Split an area curve into beats: (ED_i, ES_i, ED_{i+1}) triples from area peaks."""
    if areas.size < 3:
        return []
    smooth = ndimage.uniform_filter1d(areas, size=max(3, int(fps * 0.06)))
    distance = max(2, int(fps * 60.0 / max_hr))
    peaks, _ = sps.find_peaks(smooth, distance=distance, prominence=0.05 * (smooth.max() - smooth.min() + 1e-9))
    segs = []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        if p1 - p0 > fps * 60.0 / min_hr:
            continue
        es = p0 + int(np.argmin(smooth[p0:p1]))
        segs.append((int(p0), int(es), int(p1)))
    return segs


def beat_to_beat_ef(masks: np.ndarray, fps: float, n_disks: int = 20) -> dict[str, float | list[float]]:
    """Per-beat EF from single-plane Simpson volumes; SD and RR-interval CV are rhythm-irregularity proxies."""
    areas = area_curve(masks)
    segs = beat_segments(areas, fps)
    efs, rrs = [], []
    for ed, es, ed2 in segs:
        edv = disk_volume_single_plane(masks[ed], n_disks).volume
        esv = disk_volume_single_plane(masks[es], n_disks).volume
        efs.append(ef_from_volumes(edv, esv))
        rrs.append((ed2 - ed) / fps)
    efs_arr, rrs_arr = np.asarray(efs), np.asarray(rrs)
    return {
        "n_beats": len(efs),
        "ef_beats": efs,
        "ef_mean": float(np.mean(efs_arr)) if efs else float("nan"),
        "ef_sd": float(np.std(efs_arr)) if len(efs) > 1 else float("nan"),
        "rr_cv": float(np.std(rrs_arr) / np.mean(rrs_arr)) if len(rrs) > 1 and np.mean(rrs_arr) > 0 else float("nan"),
    }


def ef_from_mask_stack(masks: np.ndarray, n_disks: int = 20, masks_a2c: np.ndarray | None = None) -> dict[str, float]:
    """Global ED/ES EF from a mask stack (single-plane), or biplane when a matching A2C stack is provided."""
    ed, es = select_ed_es(area_curve(masks))
    if masks_a2c is None:
        edv = disk_volume_single_plane(masks[ed], n_disks).volume
        esv = disk_volume_single_plane(masks[es], n_disks).volume
    else:
        ed2, es2 = select_ed_es(area_curve(masks_a2c))
        edv = disk_volume_biplane(masks[ed], masks_a2c[ed2], n_disks).volume
        esv = disk_volume_biplane(masks[es], masks_a2c[es2], n_disks).volume
    return {"ef": ef_from_volumes(edv, esv), "edv": edv, "esv": esv, "ed_frame": ed, "es_frame": es}


def mask_quality(mask: np.ndarray) -> dict[str, float]:
    """Gate features for abstention: area fraction, solidity, border contact, elongation."""
    m = largest_component(mask)
    area = float(m.sum())
    if area == 0:
        return {"area_fraction": 0.0, "solidity": 0.0, "touches_border": 1.0, "elongation": float("nan")}
    hull_area = _convex_hull_area(m)
    c, axis = principal_axis(m)
    ys, xs = np.nonzero(m)
    pts = np.stack([ys, xs], 1).astype(float) - c
    proj = pts @ axis
    perp = pts @ np.array([-axis[1], axis[0]])
    border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    return {
        "area_fraction": area / m.size,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "touches_border": float(border),
        "elongation": float((proj.max() - proj.min()) / max(perp.max() - perp.min(), 1e-6)),
    }


def _convex_hull_area(mask: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return float(ys.size)
    try:
        return float(ConvexHull(np.stack([xs, ys], 1)).volume)
    except Exception:  # degenerate (collinear) masks
        return float(ys.size)
