"""SWC parsing and morphometrics for allometric scaling analyses.

SWC columns: ``id type x y z radius parent`` with types 1 = soma, 2 = axon, 3 = basal dendrite,
4 = apical dendrite (other integers are custom). Parent ``-1`` marks a root.

All functions are pure numpy so they run on hundreds of thousands of files without extra
dependencies; ``scipy.spatial.ConvexHull`` is used for the spanning volume when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
DENDRITE_TYPES: Tuple[int, ...] = (BASAL, APICAL)


@dataclass
class SWCTree:
    """Array representation of an SWC morphology (nodes in file order).

    ``parent_idx`` holds the *row index* of the parent (``-1`` for roots), not the SWC id.
    """

    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @property
    def roots(self) -> np.ndarray:
        return np.flatnonzero(self.parent_idx < 0)

    def soma_center(self) -> np.ndarray:
        """Centroid of soma-typed nodes, falling back to the first root."""
        mask = self.types == SOMA
        if mask.any():
            return self.xyz[mask].mean(axis=0)
        return self.xyz[self.roots[0]]

    def subtree_mask(self, neurite_types: Optional[Iterable[int]]) -> np.ndarray:
        """Boolean mask of nodes whose type is in ``neurite_types`` (all nodes if ``None``)."""
        if neurite_types is None:
            return np.ones(len(self), dtype=bool)
        return np.isin(self.types, list(neurite_types))


def parse_swc_text(text: str) -> SWCTree:
    """Parse SWC content from a string. Lines starting with ``#`` are ignored."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 7:
            continue
        rows.append([float(p) for p in parts[:7]])
    if not rows:
        raise ValueError("no SWC rows found")
    arr = np.asarray(rows, dtype=float)
    ids = arr[:, 0].astype(np.int64)
    parents = arr[:, 6].astype(np.int64)
    id_to_idx = {int(i): k for k, i in enumerate(ids)}
    parent_idx = np.array([id_to_idx.get(int(p), -1) if p >= 0 else -1 for p in parents], dtype=np.int64)
    return SWCTree(ids=ids, types=arr[:, 1].astype(np.int64), xyz=arr[:, 2:5].copy(),
                   radius=arr[:, 5].copy(), parent_idx=parent_idx)


def read_swc(path: Union[str, Path]) -> SWCTree:
    """Read an SWC file from disk."""
    return parse_swc_text(Path(path).read_text(errors="ignore"))


# ---------------------------------------------------------------------- topology
def segment_lengths(tree: SWCTree) -> np.ndarray:
    """Euclidean length of the segment from each node to its parent (0 for roots)."""
    out = np.zeros(len(tree), dtype=float)
    has_parent = tree.parent_idx >= 0
    d = tree.xyz[has_parent] - tree.xyz[tree.parent_idx[has_parent]]
    out[has_parent] = np.linalg.norm(d, axis=1)
    return out


def child_counts(tree: SWCTree) -> np.ndarray:
    """Number of children per node."""
    counts = np.zeros(len(tree), dtype=np.int64)
    has_parent = tree.parent_idx >= 0
    np.add.at(counts, tree.parent_idx[has_parent], 1)
    return counts


def topological_order(tree: SWCTree) -> np.ndarray:
    """Node indices ordered so that every parent precedes its children (BFS from roots)."""
    counts = child_counts(tree)
    children: Dict[int, list] = {}
    for i, p in enumerate(tree.parent_idx):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    order = []
    stack = list(int(r) for r in tree.roots)
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children.get(node, ()))
    if len(order) != len(tree):  # orphan nodes (broken parent links) are appended
        seen = set(order)
        order.extend(i for i in range(len(tree)) if i not in seen)
    del counts
    return np.asarray(order, dtype=np.int64)


def path_distances(tree: SWCTree) -> np.ndarray:
    """Path length from each node to its root along the tree."""
    seg = segment_lengths(tree)
    dist = np.zeros(len(tree), dtype=float)
    for i in topological_order(tree):
        p = tree.parent_idx[i]
        if p >= 0:
            dist[i] = dist[p] + seg[i]
    return dist


def branch_orders(tree: SWCTree, ignore_soma: bool = True) -> np.ndarray:
    """Centrifugal branch order: number of bifurcations between the root and each node.

    Soma nodes are treated as order 0 and do not count as bifurcations when ``ignore_soma``.
    """
    counts = child_counts(tree)
    order = np.zeros(len(tree), dtype=np.int64)
    for i in topological_order(tree):
        p = tree.parent_idx[i]
        if p < 0:
            continue
        parent_is_bif = counts[p] >= 2 and not (ignore_soma and tree.types[p] == SOMA)
        order[i] = order[p] + (1 if parent_is_bif else 0)
    return order


# ---------------------------------------------------------------------- geometry
def spanning_volume(points: np.ndarray, method: str = "hull") -> float:
    """Spanning volume of a point cloud: convex hull (default) or axis-aligned bounding box."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 4:
        return 0.0
    if method == "bbox":
        ext = pts.max(axis=0) - pts.min(axis=0)
        return float(np.prod(ext))
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore

        try:
            return float(ConvexHull(pts).volume)
        except QhullError:  # degenerate (planar) cloud
            return 0.0
    except ImportError:  # pragma: no cover - scipy missing
        ext = pts.max(axis=0) - pts.min(axis=0)
        return float(np.prod(ext))


def spanning_area(points: np.ndarray) -> float:
    """Area of the convex hull of the point cloud projected on its two principal axes."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt[:2].T
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore

        try:
            return float(ConvexHull(proj).volume)  # 2-D "volume" is area
        except QhullError:
            return 0.0
    except ImportError:  # pragma: no cover
        ext = proj.max(axis=0) - proj.min(axis=0)
        return float(np.prod(ext))


def effective_dimension(points: np.ndarray) -> float:
    """Participation-ratio dimensionality of the point cloud (1 = line, 2 = plane, 3 = volume).

    ``D_eff = (sum lambda_i)^2 / sum lambda_i^2`` for PCA eigenvalues ``lambda_i``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 1.0
    cov = np.cov(pts - pts.mean(axis=0), rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = lam.sum()
    if s <= 0:
        return 1.0
    return float(s ** 2 / np.sum(lam ** 2))


def planarity(points: np.ndarray) -> float:
    """Ratio of the smallest to the largest PCA standard deviation (0 = perfectly planar)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    lam = np.clip(np.linalg.eigvalsh(np.cov(pts - pts.mean(axis=0), rowvar=False)), 0, None)
    return float(np.sqrt(lam.min() / lam.max())) if lam.max() > 0 else 0.0


# ---------------------------------------------------------------------- Sholl
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def sholl_profile(tree: SWCTree, radii: Sequence[float], neurite_types: Optional[Iterable[int]] = DENDRITE_TYPES,
                  center: Optional[np.ndarray] = None) -> np.ndarray:
    """Number of segments crossing concentric spheres of the given radii (classic Sholl).

    A segment (parent -> child) crosses radius ``r`` if ``min(d_p, d_c) < r <= max(d_p, d_c)``
    where ``d`` is the Euclidean distance of the node from ``center`` (default: soma centroid).
    """
    center = tree.soma_center() if center is None else np.asarray(center, dtype=float)
    mask = tree.subtree_mask(neurite_types) & (tree.parent_idx >= 0)
    if not mask.any():
        return np.zeros(len(radii), dtype=np.int64)
    d = np.linalg.norm(tree.xyz - center, axis=1)
    d_child = d[mask]
    d_parent = d[tree.parent_idx[mask]]
    lo = np.minimum(d_child, d_parent)
    hi = np.maximum(d_child, d_parent)
    r = np.asarray(radii, dtype=float)[:, None]
    return ((lo[None, :] < r) & (r <= hi[None, :])).sum(axis=1)


def sholl_summary(tree: SWCTree, step: float = 10.0, neurite_types: Optional[Iterable[int]] = DENDRITE_TYPES,
                  ) -> Dict[str, float]:
    """Peak crossings, radius of the peak, critical radius fraction and Sholl-integral (AUC)."""
    center = tree.soma_center()
    mask = tree.subtree_mask(neurite_types)
    if not mask.any():
        return {"sholl_peak": 0.0, "sholl_peak_radius": 0.0, "sholl_auc": 0.0, "sholl_radius_max": 0.0}
    rmax = float(np.linalg.norm(tree.xyz[mask] - center, axis=1).max())
    radii = np.arange(step, rmax + step, step)
    prof = sholl_profile(tree, radii, neurite_types, center)
    k = int(np.argmax(prof)) if prof.size else 0
    return {
        "sholl_peak": float(prof.max()) if prof.size else 0.0,
        "sholl_peak_radius": float(radii[k]) if prof.size else 0.0,
        "sholl_auc": float(_trapezoid(prof, radii)) if prof.size > 1 else 0.0,
        "sholl_radius_max": rmax,
    }


# ---------------------------------------------------------------------- morphometrics
def morphometrics(tree: SWCTree, neurite_types: Optional[Iterable[int]] = DENDRITE_TYPES,
                  sholl_step: float = 10.0) -> Dict[str, float]:
    """Compute the morphometrics used by the scaling models for one neurite domain.

    Returns total length, branch points, tips, stems, max branch order, extents, hull volume,
    projected area, effective dimensionality, planarity, mean segment length and Sholl summaries.
    """
    mask = tree.subtree_mask(neurite_types)
    seg = segment_lengths(tree)
    counts = child_counts(tree)
    has_parent = tree.parent_idx >= 0
    # a segment belongs to the domain if its child node is in the domain and it has a parent
    seg_mask = mask & has_parent
    total_length = float(seg[seg_mask].sum())
    n_nodes = int(mask.sum())
    # branch points: domain nodes with >= 2 children (soma excluded)
    bif_mask = mask & (counts >= 2) & (tree.types != SOMA)
    n_branch_points = int(bif_mask.sum())
    tip_mask = mask & (counts == 0) & (tree.types != SOMA)
    n_tips = int(tip_mask.sum())
    # stems: domain nodes whose parent is soma or a root
    parent_types = np.where(has_parent, tree.types[np.clip(tree.parent_idx, 0, None)], SOMA)
    stem_mask = mask & (tree.types != SOMA) & ((~has_parent) | (parent_types == SOMA))
    n_stems = int(stem_mask.sum())
    orders = branch_orders(tree)
    max_order = int(orders[mask].max()) if n_nodes else 0
    pdist = path_distances(tree)
    center = tree.soma_center()
    pts = tree.xyz[mask]
    if pts.shape[0] == 0:
        pts = center[None, :]
    ext = pts.max(axis=0) - pts.min(axis=0)
    euclid = np.linalg.norm(pts - center, axis=1)
    out: Dict[str, float] = {
        "n_nodes": float(n_nodes),
        "total_length": total_length,
        "n_branch_points": float(n_branch_points),
        "n_tips": float(n_tips),
        "n_stems": float(n_stems),
        "max_branch_order": float(max_order),
        "mean_branch_order": float(orders[mask].mean()) if n_nodes else 0.0,
        "max_path_distance": float(pdist[mask].max()) if n_nodes else 0.0,
        "max_euclidean_distance": float(euclid.max()),
        "width": float(ext[0]),
        "height": float(ext[1]),
        "depth": float(ext[2]),
        "hull_volume": spanning_volume(pts),
        "bbox_volume": spanning_volume(pts, method="bbox"),
        "hull_area_2d": spanning_area(pts),
        "effective_dimension": effective_dimension(pts),
        "planarity": planarity(pts),
        "mean_segment_length": float(seg[seg_mask].mean()) if seg_mask.any() else 0.0,
        "mean_radius": float(tree.radius[mask].mean()) if n_nodes else 0.0,
    }
    out.update(sholl_summary(tree, step=sholl_step, neurite_types=neurite_types))
    return out


def wiring_law_prediction(n_branch_points: np.ndarray, volume: np.ndarray, dim: int = 3) -> np.ndarray:
    """Optimal-wiring scaling prediction (up to a constant): ``L ∝ n^((d-1)/d) * V^(1/d)``.

    For 3-D arbors this is the 2/3 power law of Cuntz et al. (2012); for planar arbors
    (``dim=2`` with ``volume`` an area) the exponents are 1/2 and 1/2.
    """
    n = np.asarray(n_branch_points, dtype=float)
    v = np.asarray(volume, dtype=float)
    return n ** ((dim - 1) / dim) * v ** (1.0 / dim)


__all__ = [
    "SWCTree", "parse_swc_text", "read_swc", "segment_lengths", "child_counts", "path_distances",
    "branch_orders", "spanning_volume", "spanning_area", "effective_dimension", "planarity",
    "sholl_profile", "sholl_summary", "morphometrics", "wiring_law_prediction", "DENDRITE_TYPES",
]
