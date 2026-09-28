"""SWC loading, axon-path extraction, descriptors, scaling and placement around a DBS lead.

SWC columns: id, type (1 soma, 2 axon, 3 basal, 4 apical), x, y, z (um), radius, parent.
Paths are returned in mm. Radii are ignored for excitability (fibre diameter is a model factor).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AXON = 2


def load_swc(path_or_text: str) -> pd.DataFrame:
    """Parse an SWC file (path) or SWC text into a DataFrame with the seven standard columns."""
    text = path_or_text
    if "\n" not in path_or_text and len(path_or_text) < 4096:
        with open(path_or_text) as fh:
            text = fh.read()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append([int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                     float(parts[5]), int(parts[6])])
    df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
    return df.set_index("id", drop=False)


def axon_paths(swc: pd.DataFrame, min_length_mm: float = 0.2, structure: int = AXON,
               fallback_all: bool = True) -> List[np.ndarray]:
    """Root-to-terminal paths (mm) through nodes of the requested structure type.

    If the reconstruction has no such nodes and ``fallback_all`` is True, dendritic paths are
    returned instead (useful for smoke tests; flagged by the caller).
    """
    sub = swc[swc["type"] == structure]
    if sub.empty and fallback_all:
        sub = swc[swc["type"] != 1]
    if sub.empty:
        return []
    ids = set(sub["id"])
    children: Dict[int, List[int]] = {}
    for i, p in zip(sub["id"], sub["parent"]):
        if p in ids:
            children.setdefault(p, []).append(i)
    terminals = [i for i in sub["id"] if i not in children]
    coords = swc[["x", "y", "z"]]
    paths: List[np.ndarray] = []
    for t in terminals:
        chain = []
        cur = t
        while cur in ids:
            chain.append(cur)
            cur = int(swc.at[cur, "parent"])
        if cur in swc.index and swc.at[cur, "type"] == 1:
            chain.append(cur)  # include soma attachment point
        chain = chain[::-1]
        pts = coords.loc[chain].to_numpy() * 1e-3
        if len(pts) >= 2 and np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) >= min_length_mm:
            paths.append(pts)
    return paths


def branch_points(swc: pd.DataFrame, structure: int = AXON) -> np.ndarray:
    """Coordinates (mm) of nodes with >= 2 children within the structure."""
    sub = swc[swc["type"] == structure]
    counts = sub["parent"].value_counts()
    bp_ids = [i for i, c in counts.items() if c >= 2 and i in sub.index]
    return sub.loc[bp_ids, ["x", "y", "z"]].to_numpy() * 1e-3 if bp_ids else np.zeros((0, 3))


def path_descriptors(path_mm: np.ndarray, bend_deg: float = 30.0) -> Dict[str, float]:
    """Length, tortuosity, mean absolute curvature and number of bends of a polyline."""
    p = np.asarray(path_mm, dtype=float)
    seg = np.diff(p, axis=0)
    ls = np.linalg.norm(seg, axis=1)
    length = float(ls.sum())
    chord = float(np.linalg.norm(p[-1] - p[0]))
    tort = length / chord if chord > 1e-9 else np.inf
    if len(seg) >= 2:
        u = seg[:-1] / np.maximum(ls[:-1, None], 1e-12)
        w = seg[1:] / np.maximum(ls[1:, None], 1e-12)
        ang = np.degrees(np.arccos(np.clip((u * w).sum(axis=1), -1, 1)))
        curv = float(np.mean(np.radians(ang) / np.maximum(0.5 * (ls[:-1] + ls[1:]), 1e-9)))
        bends = int(np.sum(ang > bend_deg))
    else:
        curv, bends = 0.0, 0
    return {"length_mm": length, "chord_mm": chord, "tortuosity": float(tort), "mean_curvature_per_mm": curv,
            "n_bends": bends, "n_points": int(len(p))}


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation matrix (Haar) via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def place_path(path_mm: np.ndarray, distance_mm: float, z_mm: float, rng: Optional[np.random.Generator] = None,
               scale: float = 1.0, azimuth_rad: Optional[float] = None, rotate: bool = True,
               anchor: str = "centroid") -> np.ndarray:
    """Rigidly place a (scaled) path so that its anchor sits at perpendicular distance ``distance_mm``
    from the lead axis (z) at height ``z_mm``; random orientation unless ``rotate=False``."""
    rng = rng or np.random.default_rng()
    p = np.asarray(path_mm, dtype=float) * scale
    a = p.mean(axis=0) if anchor == "centroid" else p[0]
    p = p - a
    if rotate:
        p = p @ random_rotation(rng).T
    az = rng.uniform(0, 2 * np.pi) if azimuth_rad is None else azimuth_rad
    target = np.array([distance_mm * np.cos(az), distance_mm * np.sin(az), z_mm])
    return p + target


def straight_fiber(length_mm: float, distance_mm: float, z_mm: float, orientation: str = "parallel",
                   azimuth_rad: float = 0.0) -> np.ndarray:
    """Straight axon at perpendicular distance from the lead axis: ``parallel`` (along z) or
    ``perpendicular`` (tangential, in the x-y plane) - the two canonical PAM cases."""
    c = np.array([distance_mm * np.cos(azimuth_rad), distance_mm * np.sin(azimuth_rad), z_mm])
    if orientation == "parallel":
        d = np.array([0.0, 0.0, 1.0])
    elif orientation == "perpendicular":
        d = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0])
    else:
        raise ValueError(orientation)
    return np.vstack([c - d * length_mm / 2, c + d * length_mm / 2])


def scale_factor_for_species(species: str) -> float:
    """Geometric scale applied to reconstructions to approximate human dimensions (design axis, not a fact)."""
    return {"mouse": 2.0, "rat": 1.6, "monkey": 1.2, "human": 1.0}.get(species.lower(), 1.0)
