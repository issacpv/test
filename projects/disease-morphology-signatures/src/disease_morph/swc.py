"""SWC parsing and morphometrics recomputed with a single definition for all archives.

Only the dendritic compartments (SWC types 3 = basal, 4 = apical) are measured by
default, because axons are reconstructed inconsistently across disease datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
DENDRITE_TYPES = (BASAL, APICAL)


@dataclass
class Morphology:
    """Arrays of an SWC file; ``parent_index`` is -1 for roots."""

    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent_index: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.ids))

    def children(self) -> List[List[int]]:
        ch: List[List[int]] = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent_index):
            if p >= 0:
                ch[p].append(i)
        return ch

    def soma_center(self) -> np.ndarray:
        m = self.types == SOMA
        return self.xyz[m].mean(axis=0) if m.any() else self.xyz[self.parent_index < 0].mean(axis=0)


def parse_swc(path: Path | str) -> Morphology:
    """Parse an SWC file (comments with '#', 7 columns) into a :class:`Morphology`."""
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            rows.append([float(v) for v in parts[:7]])
    if not rows:
        raise ValueError(f"no SWC rows in {path}")
    arr = np.asarray(rows, dtype=float)
    return morphology_from_array(arr)


def morphology_from_array(arr: np.ndarray) -> Morphology:
    """Build a Morphology from an (n, 7) array of [id, type, x, y, z, r, parent]."""
    ids = arr[:, 0].astype(int)
    id_to_idx = {int(i): k for k, i in enumerate(ids)}
    parent = np.array([id_to_idx.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=int)
    return Morphology(ids=ids, types=arr[:, 1].astype(int), xyz=arr[:, 2:5].astype(float),
                      radius=arr[:, 5].astype(float), parent_index=parent)


def segment_lengths(m: Morphology) -> np.ndarray:
    """Length of the segment from each node to its parent (0 for roots)."""
    out = np.zeros(m.n)
    has_parent = m.parent_index >= 0
    out[has_parent] = np.linalg.norm(m.xyz[has_parent] - m.xyz[m.parent_index[has_parent]], axis=1)
    return out


def branch_orders(m: Morphology) -> np.ndarray:
    """Centrifugal branch order: 0 at the soma/root, +1 after every bifurcation."""
    ch = m.children()
    order = np.zeros(m.n, dtype=int)
    # process nodes in an order guaranteeing parents first (BFS from roots)
    roots = [i for i, p in enumerate(m.parent_index) if p < 0]
    stack = list(roots)
    while stack:
        i = stack.pop()
        kids = ch[i]
        bump = 1 if (len(kids) > 1 and m.types[i] != SOMA) else 0
        for k in kids:
            order[k] = order[i] + bump
            stack.append(k)
    return order


def path_distance_to_root(m: Morphology) -> np.ndarray:
    seg = segment_lengths(m)
    dist = np.zeros(m.n)
    ch = m.children()
    stack = [i for i, p in enumerate(m.parent_index) if p < 0]
    while stack:
        i = stack.pop()
        for k in ch[i]:
            dist[k] = dist[i] + seg[k]
            stack.append(k)
    return dist


def sholl_profile(m: Morphology, radii: Sequence[float], types: Sequence[int] = DENDRITE_TYPES) -> np.ndarray:
    """Number of segments (of the given types) crossing each sphere centred on the soma."""
    c = m.soma_center()
    mask = np.isin(m.types, types) & (m.parent_index >= 0)
    d_node = np.linalg.norm(m.xyz[mask] - c, axis=1)
    d_par = np.linalg.norm(m.xyz[m.parent_index[mask]] - c, axis=1)
    lo, hi = np.minimum(d_node, d_par), np.maximum(d_node, d_par)
    return np.array([int(((lo <= r) & (hi > r)).sum()) for r in radii])


def effective_dimensionality(xyz: np.ndarray) -> float:
    """Participation ratio of the coordinate covariance eigenvalues (1 = line, 3 = isotropic)."""
    if len(xyz) < 3:
        return float("nan")
    ev = np.linalg.eigvalsh(np.cov(xyz.T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else float("nan")


def morphometrics(m: Morphology, types: Sequence[int] = DENDRITE_TYPES,
                  sholl_radii: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Dendritic morphometrics used as meta-analysis outcomes.

    Returns total length, number of bifurcations and tips, max branch order,
    mean branch (inter-bifurcation) length, mean diameter, max Euclidean and path
    extents, convex-hull volume, effective dimensionality and the Sholl profile
    (as ``sholl_<radius>`` keys plus ``sholl_auc`` and ``sholl_peak``).
    """
    mask = np.isin(m.types, types)
    seg = segment_lengths(m)
    ch = m.children()
    n_children = np.array([len(c) for c in ch])
    is_bif = mask & (n_children > 1)
    is_tip = mask & (n_children == 0)
    order = branch_orders(m)
    pdist = path_distance_to_root(m)
    c = m.soma_center()
    edist = np.linalg.norm(m.xyz - c, axis=1)
    total_length = float(seg[mask].sum())
    n_bif = int(is_bif.sum())
    n_tips = int(is_tip.sum())
    n_branches = n_bif + n_tips
    out: Dict[str, float] = {
        "total_length": total_length,
        "n_bifurcations": n_bif,
        "n_tips": n_tips,
        "max_branch_order": int(order[mask].max()) if mask.any() else 0,
        "mean_branch_length": total_length / n_branches if n_branches else float("nan"),
        "mean_diameter": float((2 * m.radius[mask]).mean()) if mask.any() else float("nan"),
        "max_euclid_extent": float(edist[mask].max()) if mask.any() else 0.0,
        "max_path_extent": float(pdist[mask].max()) if mask.any() else 0.0,
        "effective_dimensionality": effective_dimensionality(m.xyz[mask]) if mask.any() else float("nan"),
    }
    try:
        from scipy.spatial import ConvexHull

        pts = m.xyz[mask]
        out["hull_volume"] = float(ConvexHull(pts).volume) if len(pts) >= 4 else float("nan")
    except Exception:  # noqa: BLE001 - degenerate (planar) point sets
        out["hull_volume"] = float("nan")
    radii = list(sholl_radii) if sholl_radii is not None else list(range(20, 501, 20))
    prof = sholl_profile(m, radii, types)
    for r, v in zip(radii, prof):
        out[f"sholl_{int(r)}"] = float(v)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    out["sholl_auc"] = float(_trapz(prof, radii)) if len(radii) > 1 else float(prof.sum())
    out["sholl_peak"] = float(prof.max()) if len(prof) else 0.0
    return out


def morphometrics_table(paths: Iterable[Path | str], **kw) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Morphometrics for many files as a DataFrame indexed by file stem (errors are logged, not raised)."""
    import logging

    import pandas as pd

    rows = {}
    for p in paths:
        try:
            rows[Path(p).stem.replace(".CNG", "")] = morphometrics(parse_swc(p), **kw)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("morphometrics failed for %s: %s", p, exc)
    return pd.DataFrame.from_dict(rows, orient="index")
