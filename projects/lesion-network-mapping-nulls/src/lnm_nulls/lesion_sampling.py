"""Lesion representation, features and null-lesion generators.

Lesions are boolean 3-D arrays on a common grid (e.g. MNI 2 mm). Anatomical context is
given by a boolean ``brain_mask`` and an integer ``territory`` map (0 = outside brain,
1..K = arterial territories, e.g. ACA/MCA/PCA/deep, left and right coded separately).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LesionFeatures:
    volume: int
    hemisphere: int            # 0 = left, 1 = right, 2 = bilateral/midline
    centroid: tuple[float, float, float]
    territory_fractions: np.ndarray  # length K+1 (index 0 = outside any territory)
    dominant_territory: int


def hemisphere_of(mask: np.ndarray, midline_axis: int = 0, tol: float = 0.9) -> int:
    """0 if >= ``tol`` of voxels are left of the midline, 1 if right, else 2 (bilateral)."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return 2
    mid = mask.shape[midline_axis] / 2.0
    left = np.mean(idx[:, midline_axis] < mid)
    if left >= tol:
        return 0
    if left <= 1 - tol:
        return 1
    return 2


def lesion_features(mask: np.ndarray, territory: np.ndarray | None = None, n_territories: int | None = None,
                    midline_axis: int = 0) -> LesionFeatures:
    """Volume, hemisphere, centroid and arterial-territory composition of a lesion mask."""
    mask = np.asarray(mask, bool)
    idx = np.argwhere(mask)
    vol = int(idx.shape[0])
    centroid = tuple(float(v) for v in (idx.mean(0) if vol else np.full(3, np.nan)))
    if territory is None:
        fr = np.array([1.0])
        dom = 0
    else:
        K = int(territory.max()) if n_territories is None else n_territories
        counts = np.bincount(territory[mask].astype(int), minlength=K + 1).astype(float)
        fr = counts / max(vol, 1)
        dom = int(np.argmax(counts[1:]) + 1) if vol and counts[1:].sum() > 0 else 0
    return LesionFeatures(vol, hemisphere_of(mask, midline_axis), centroid, fr, dom)


def sphere_lesion(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float) -> np.ndarray:
    """Boolean sphere on a grid of ``shape``."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    d2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return d2 <= radius**2


def random_sphere_lesions(n: int, brain_mask: np.ndarray, target_volumes: np.ndarray,
                          rng: np.random.Generator, max_tries: int = 200) -> list[np.ndarray]:
    """N2 null: volume-matched spheres placed at random inside the brain (>= 80 % of voxels in mask)."""
    brain_mask = np.asarray(brain_mask, bool)
    coords = np.argwhere(brain_mask)
    out: list[np.ndarray] = []
    for v in np.asarray(target_volumes):
        r = (3.0 * float(v) / (4.0 * np.pi)) ** (1.0 / 3.0)
        for _ in range(max_tries):
            c = coords[rng.integers(coords.shape[0])]
            s = sphere_lesion(brain_mask.shape, tuple(c), r) & brain_mask
            if s.sum() >= 0.8 * v:
                out.append(s)
                break
        else:
            out.append(s)
    return out


def shift_lesion(mask: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Rigidly translate a boolean mask by integer ``offset`` (voxels shifted outside are dropped)."""
    idx = np.argwhere(mask) + np.asarray(offset, int)
    keep = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
    out = np.zeros_like(mask, dtype=bool)
    out[tuple(idx[keep].T)] = True
    return out


def shuffle_lesion_locations(lesions: list[np.ndarray], brain_mask: np.ndarray, rng: np.random.Generator,
                             same_hemisphere: bool = True, midline_axis: int = 0, min_inside: float = 0.9,
                             max_tries: int = 200) -> list[np.ndarray]:
    """N3 null: translate each real lesion to a random location, keeping >= ``min_inside`` in the brain
    and (optionally) the original hemisphere."""
    brain_mask = np.asarray(brain_mask, bool)
    coords = np.argwhere(brain_mask)
    out = []
    for les in lesions:
        idx = np.argwhere(les)
        cen = idx.mean(0)
        hemi = hemisphere_of(les, midline_axis)
        best = les
        for _ in range(max_tries):
            target = coords[rng.integers(coords.shape[0])]
            cand = shift_lesion(les, np.round(target - cen).astype(int))
            inside = (cand & brain_mask).sum() / max(les.sum(), 1)
            if inside < min_inside:
                continue
            cand &= brain_mask
            if same_hemisphere and hemisphere_of(cand, midline_axis) != hemi:
                continue
            best = cand
            break
        out.append(best)
    return out


def matched_resample(pool: list[np.ndarray], pool_features: list[LesionFeatures], targets: list[LesionFeatures],
                     rng: np.random.Generator, volume_tol: float = 0.2, match_territory: bool = True,
                     match_hemisphere: bool = True) -> list[int]:
    """N4 null: for each target lesion, draw a pool index with similar volume (±``volume_tol``),
    same hemisphere and same dominant territory. Falls back to relaxing territory, then hemisphere,
    then nearest volume if no candidate exists. Returns pool indices (with replacement)."""
    vols = np.array([f.volume for f in pool_features], float)
    hemis = np.array([f.hemisphere for f in pool_features])
    doms = np.array([f.dominant_territory for f in pool_features])
    picks = []
    for t in targets:
        ok_vol = np.abs(vols - t.volume) <= volume_tol * t.volume
        cand = ok_vol.copy()
        if match_hemisphere:
            cand &= hemis == t.hemisphere
        if match_territory:
            cand &= doms == t.dominant_territory
        if not cand.any() and match_territory:
            cand = ok_vol & (hemis == t.hemisphere) if match_hemisphere else ok_vol
        if not cand.any():
            cand = ok_vol
        if not cand.any():
            cand = np.zeros_like(ok_vol)
            cand[np.argmin(np.abs(vols - t.volume))] = True
        picks.append(int(rng.choice(np.flatnonzero(cand))))
    return picks
