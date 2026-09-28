"""SWC parsing, SWC -> graph conversion and hand-crafted morphometrics.

The graph representation is framework-agnostic (numpy arrays); ``to_torch_geometric`` converts to a
``torch_geometric.data.Data`` object when that package is installed.

SWC types: 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite. Allen SWC files use µm with the
y axis along the pial-depth direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
NODE_FEATURE_NAMES: Tuple[str, ...] = (
    "is_soma", "is_axon", "is_basal", "is_apical", "radius", "path_dist", "euclid_dist", "branch_order",
    "seg_length", "is_bifurcation", "is_tip", "dir_x", "dir_y", "dir_z", "rel_depth",
)


@dataclass
class SWCTree:
    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent_idx: np.ndarray  # row index of parent, -1 for roots

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def soma_center(self) -> np.ndarray:
        m = self.types == SOMA
        return self.xyz[m].mean(axis=0) if m.any() else self.xyz[np.flatnonzero(self.parent_idx < 0)[0]]


def parse_swc_text(text: str) -> SWCTree:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 7:
            rows.append([float(p) for p in parts[:7]])
    if not rows:
        raise ValueError("empty SWC")
    arr = np.asarray(rows, dtype=float)
    ids = arr[:, 0].astype(np.int64)
    id_to_idx = {int(i): k for k, i in enumerate(ids)}
    parent_idx = np.array([id_to_idx.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=np.int64)
    return SWCTree(ids=ids, types=arr[:, 1].astype(np.int64), xyz=arr[:, 2:5].copy(), radius=arr[:, 5].copy(),
                   parent_idx=parent_idx)


def read_swc(path: Union[str, Path]) -> SWCTree:
    return parse_swc_text(Path(path).read_text(errors="ignore"))


# ---------------------------------------------------------------------- tree utilities
def _children_lists(tree: SWCTree) -> List[List[int]]:
    ch: List[List[int]] = [[] for _ in range(len(tree))]
    for i, p in enumerate(tree.parent_idx):
        if p >= 0:
            ch[p].append(i)
    return ch


def _topo_order(tree: SWCTree, children: List[List[int]]) -> List[int]:
    order: List[int] = []
    stack = [int(r) for r in np.flatnonzero(tree.parent_idx < 0)]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(children[n])
    if len(order) < len(tree):
        seen = set(order)
        order.extend(i for i in range(len(tree)) if i not in seen)
    return order


def node_attributes(tree: SWCTree) -> Dict[str, np.ndarray]:
    """Per-node segment length, path distance, Euclidean distance, branch order, child count."""
    n = len(tree)
    children = _children_lists(tree)
    n_children = np.array([len(c) for c in children], dtype=np.int64)
    seg = np.zeros(n)
    has_p = tree.parent_idx >= 0
    seg[has_p] = np.linalg.norm(tree.xyz[has_p] - tree.xyz[tree.parent_idx[has_p]], axis=1)
    path = np.zeros(n)
    order = np.zeros(n, dtype=np.int64)
    for i in _topo_order(tree, children):
        p = tree.parent_idx[i]
        if p >= 0:
            path[i] = path[p] + seg[i]
            bif_parent = n_children[p] >= 2 and tree.types[p] != SOMA
            order[i] = order[p] + (1 if bif_parent else 0)
    euclid = np.linalg.norm(tree.xyz - tree.soma_center(), axis=1)
    return {"seg_length": seg, "path_dist": path, "euclid_dist": euclid, "branch_order": order,
            "n_children": n_children}


# ---------------------------------------------------------------------- graph
@dataclass
class NeuronGraph:
    """Graph of an SWC morphology.

    ``x`` is ``(n_nodes, n_features)`` float32, ``edge_index`` is ``(2, n_edges)`` int64 with both
    directions of every parent-child edge, ``pos`` holds soma-centred coordinates.
    """

    x: np.ndarray
    edge_index: np.ndarray
    pos: np.ndarray
    node_type: np.ndarray
    feature_names: Tuple[str, ...] = NODE_FEATURE_NAMES
    globals: Optional[Dict[str, float]] = None
    name: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    def to_torch_geometric(self):  # pragma: no cover - optional dependency
        import torch
        from torch_geometric.data import Data  # type: ignore

        return Data(x=torch.as_tensor(self.x), edge_index=torch.as_tensor(self.edge_index),
                    pos=torch.as_tensor(self.pos, dtype=torch.float32))


def swc_to_graph(tree: SWCTree, include_axon: bool = False, keep_types: Optional[Iterable[int]] = None,
                 scale_um: float = 100.0, name: str = "") -> NeuronGraph:
    """Convert an SWC tree to a ``NeuronGraph`` with normalised node features.

    Parameters
    ----------
    include_axon : keep axon nodes (type 2). Default drops them (Allen axons are often truncated).
    keep_types : explicit set of SWC types to keep (overrides ``include_axon``).
    scale_um : distances are divided by this value so features are O(1).
    """
    if keep_types is None:
        keep_types = {SOMA, BASAL, APICAL} | ({AXON} if include_axon else set())
    attrs = node_attributes(tree)
    keep = np.isin(tree.types, list(keep_types))
    # drop nodes whose parent chain leaves the kept set (e.g. dendrite hanging off an axon node)
    idx_map = -np.ones(len(tree), dtype=np.int64)
    idx_map[keep] = np.arange(int(keep.sum()))
    center = tree.soma_center()
    pos = (tree.xyz[keep] - center).astype(np.float32)
    types = tree.types[keep]
    parent = tree.parent_idx[keep]
    parent_new = np.where(parent >= 0, idx_map[np.clip(parent, 0, None)], -1)
    # direction of the segment (unit vector), zero for roots
    direction = np.zeros((int(keep.sum()), 3), dtype=np.float32)
    has_p = parent_new >= 0
    d = pos[has_p] - pos[parent_new[has_p]]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    direction[has_p] = np.divide(d, norms, out=np.zeros_like(d), where=norms > 0)
    extent_y = max(float(np.ptp(pos[:, 1])), 1e-6)
    rel_depth = (pos[:, 1] - pos[:, 1].min()) / extent_y
    n_children = attrs["n_children"][keep]
    feats = np.column_stack([
        (types == SOMA), (types == AXON), (types == BASAL), (types == APICAL),
        tree.radius[keep] / 5.0,
        attrs["path_dist"][keep] / scale_um,
        attrs["euclid_dist"][keep] / scale_um,
        attrs["branch_order"][keep] / 10.0,
        attrs["seg_length"][keep] / 10.0,
        (n_children >= 2) & (types != SOMA),
        (n_children == 0) & (types != SOMA),
        direction[:, 0], direction[:, 1], direction[:, 2],
        rel_depth,
    ]).astype(np.float32)
    src = np.flatnonzero(has_p)
    dst = parent_new[has_p]
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    glob = {"n_nodes": float(feats.shape[0]), "total_length_um": float(attrs["seg_length"][keep].sum()),
            "soma_radius_um": float(tree.radius[tree.types == SOMA].mean()) if (tree.types == SOMA).any() else 0.0}
    return NeuronGraph(x=feats, edge_index=edge_index, pos=pos, node_type=types.astype(np.int64), globals=glob, name=name)


def rotate_about_y(graph: NeuronGraph, angle_rad: float) -> NeuronGraph:
    """Augmentation: rotate positions and direction features about the pial (y) axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    x = graph.x.copy()
    names = graph.feature_names
    di = [names.index("dir_x"), names.index("dir_y"), names.index("dir_z")]
    x[:, di] = x[:, di] @ R.T
    return NeuronGraph(x=x, edge_index=graph.edge_index, pos=graph.pos @ R.T, node_type=graph.node_type,
                       feature_names=names, globals=graph.globals, name=graph.name)


def subtree_dropout(graph: NeuronGraph, tree: SWCTree, p_drop: float, rng: np.random.Generator) -> NeuronGraph:
    """Augmentation: rebuild the graph after removing random terminal subtrees (simulates cut dendrites)."""
    children = _children_lists(tree)
    n_children = np.array([len(c) for c in children])
    tips = np.flatnonzero((n_children == 0) & (tree.types != SOMA))
    drop_tips = tips[rng.random(tips.size) < p_drop]
    remove = np.zeros(len(tree), dtype=bool)
    for t in drop_tips:
        node = int(t)
        # walk up until a bifurcation, removing the terminal branch
        while node >= 0 and not remove[node]:
            remove[node] = True
            p = int(tree.parent_idx[node])
            if p < 0 or n_children[p] >= 2 or tree.types[p] == SOMA:
                break
            node = p
    keep = ~remove
    sub = SWCTree(ids=tree.ids[keep], types=tree.types[keep], xyz=tree.xyz[keep], radius=tree.radius[keep],
                  parent_idx=tree.parent_idx[keep])
    # re-index parents
    idx_map = -np.ones(len(tree), dtype=np.int64)
    idx_map[keep] = np.arange(int(keep.sum()))
    sub.parent_idx = np.where(sub.parent_idx >= 0, idx_map[np.clip(sub.parent_idx, 0, None)], -1)
    return swc_to_graph(sub, keep_types=set(np.unique(graph.node_type).tolist()), name=graph.name)


# ---------------------------------------------------------------------- morphometrics (ridge baseline)
def _hull_volume(pts: np.ndarray) -> float:
    if pts.shape[0] < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError

        try:
            return float(ConvexHull(pts).volume)
        except QhullError:
            return 0.0
    except ImportError:  # pragma: no cover
        return float(np.prod(np.ptp(pts, axis=0)))


def sholl_crossings(tree: SWCTree, radii: Sequence[float], mask: np.ndarray) -> np.ndarray:
    center = tree.soma_center()
    d = np.linalg.norm(tree.xyz - center, axis=1)
    m = mask & (tree.parent_idx >= 0)
    lo = np.minimum(d[m], d[tree.parent_idx[m]])
    hi = np.maximum(d[m], d[tree.parent_idx[m]])
    r = np.asarray(radii, dtype=float)[:, None]
    return ((lo[None, :] < r) & (r <= hi[None, :])).sum(axis=1)


def morphometric_vector(tree: SWCTree, include_axon: bool = False) -> Dict[str, float]:
    """~25 hand-crafted morphometrics for the ridge baseline (dendrites by default)."""
    attrs = node_attributes(tree)
    types = tree.types
    dend = np.isin(types, [BASAL, APICAL]) | (np.isin(types, [AXON]) if include_axon else False)
    seg = attrs["seg_length"]
    nch = attrs["n_children"]
    out: Dict[str, float] = {}
    for label, m in (("dend", dend), ("basal", types == BASAL), ("apical", types == APICAL)):
        out[f"{label}_length"] = float(seg[m & (tree.parent_idx >= 0)].sum())
        out[f"{label}_n_bif"] = float(((nch >= 2) & m).sum())
        out[f"{label}_n_tips"] = float(((nch == 0) & m).sum())
    if not dend.any():
        pts = tree.xyz
    else:
        pts = tree.xyz[dend]
    center = tree.soma_center()
    ext = np.ptp(pts, axis=0) if pts.shape[0] else np.zeros(3)
    euclid = np.linalg.norm(pts - center, axis=1) if pts.shape[0] else np.zeros(1)
    euclid_all = np.linalg.norm(tree.xyz - center, axis=1)
    m_tort = dend & (euclid_all > 1e-6)
    tortuosity = float(np.mean(attrs["path_dist"][m_tort] / euclid_all[m_tort])) if m_tort.any() else 1.0
    out.update({
        "max_branch_order": float(attrs["branch_order"][dend].max()) if dend.any() else 0.0,
        "max_path_dist": float(attrs["path_dist"][dend].max()) if dend.any() else 0.0,
        "max_euclid_dist": float(euclid.max()),
        "width": float(ext[0]), "height": float(ext[1]), "depth": float(ext[2]),
        "hull_volume": _hull_volume(pts),
        "mean_radius": float(tree.radius[dend].mean()) if dend.any() else 0.0,
        "soma_radius": float(tree.radius[types == SOMA].mean()) if (types == SOMA).any() else 0.0,
        "mean_seg_length": float(seg[dend & (tree.parent_idx >= 0)].mean()) if (dend & (tree.parent_idx >= 0)).any() else 0.0,
        "n_stems": float(((types != SOMA) & dend & np.isin(tree.parent_idx, np.flatnonzero(types == SOMA))).sum()),
        "tortuosity": tortuosity,
        "apical_fraction": float(out["apical_length"] / out["dend_length"]) if out["dend_length"] > 0 else 0.0,
    })
    rmax = float(euclid.max()) if euclid.size else 0.0
    radii = np.arange(10.0, rmax + 10.0, 10.0) if rmax > 10 else np.array([10.0])
    prof = sholl_crossings(tree, radii, dend)
    out["sholl_peak"] = float(prof.max()) if prof.size else 0.0
    out["sholl_peak_radius"] = float(radii[int(np.argmax(prof))]) if prof.size else 0.0
    out["sholl_auc"] = float(np.sum(prof) * 10.0)
    return out


__all__ = ["SWCTree", "NeuronGraph", "NODE_FEATURE_NAMES", "parse_swc_text", "read_swc", "node_attributes",
           "swc_to_graph", "rotate_about_y", "subtree_dropout", "morphometric_vector", "sholl_crossings"]
