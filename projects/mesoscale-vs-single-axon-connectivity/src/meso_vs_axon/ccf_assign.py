"""Assign single-neuron axon nodes and terminals to CCFv3 structures and build target vectors.

Coordinate conventions: CCFv3 in µm with axes (x = anterior→posterior, y = dorsal→ventral,
z = left→right); the 25 µm annotation volume has shape (528, 320, 456), so the midline is at
z ≈ 456/2 * 25 = 5700 µm. MouseLight JSON and SEU-ALLEN CCF-registered SWC files follow this
convention; use ``axis_order``/``scale`` for datasets that do not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
CCF_MIDLINE_UM = 5700.0


@dataclass
class SWCTree:
    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent_idx: np.ndarray
    allen_ids: Optional[np.ndarray] = None  # per-node CCF structure ids when the source provides them

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def soma_xyz(self) -> np.ndarray:
        m = self.types == SOMA
        return self.xyz[m].mean(axis=0) if m.any() else self.xyz[np.flatnonzero(self.parent_idx < 0)[0]]


def parse_swc_text(text: str, scale: float = 1.0, axis_order: Sequence[int] = (0, 1, 2)) -> SWCTree:
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
    xyz = arr[:, 2:5][:, list(axis_order)] * scale
    return SWCTree(ids=ids, types=arr[:, 1].astype(np.int64), xyz=xyz, radius=arr[:, 5].copy(), parent_idx=parent_idx)


def read_swc(path: Union[str, Path], **kw) -> SWCTree:
    return parse_swc_text(Path(path).read_text(errors="ignore"), **kw)


# ---------------------------------------------------------------------- MouseLight JSON
def mouselight_json_neurons(path: Union[str, Path]) -> List[SWCTree]:
    """Parse a MouseLight neuron-browser JSON export into SWC trees (CCF µm, with ``allen_ids``).

    Expected structure: ``{"neurons": [{"idString": ..., "soma": {...}, "axon": [node, ...],
    "dendrite": [node, ...]}]}`` where a node has ``sampleNumber, structureIdentifier, x, y, z,
    radius, parentNumber, allenId``.
    """
    data = json.loads(Path(path).read_text())
    neurons = data["neurons"] if isinstance(data, dict) else data
    out = []
    for nrn in neurons:
        rows, allen = [], []
        offset = 0
        soma = nrn.get("soma")
        if soma:
            rows.append([1, SOMA, soma["x"], soma["y"], soma["z"], soma.get("radius", 5.0), -1])
            allen.append(soma.get("allenId", 0))
            offset = 1
        for key, default_type in (("axon", AXON), ("dendrite", BASAL)):
            nodes = nrn.get(key) or []
            if not nodes:
                continue
            base = len(rows)
            local = {}
            for k, nd in enumerate(nodes):
                local[nd["sampleNumber"]] = base + k + 1
            for nd in nodes:
                parent = nd.get("parentNumber", -1)
                if parent in local:
                    p = local[parent]
                elif offset:
                    p = 1  # attach the neurite root to the soma
                else:
                    p = -1
                st = nd.get("structureIdentifier", default_type)
                st = default_type if st in (None, 0, 1) else st
                rows.append([local[nd["sampleNumber"]], st, nd["x"], nd["y"], nd["z"], nd.get("radius", 1.0), p])
                allen.append(nd.get("allenId", 0))
        arr = np.asarray(rows, dtype=float)
        ids = arr[:, 0].astype(np.int64)
        id_to_idx = {int(i): k for k, i in enumerate(ids)}
        parent_idx = np.array([id_to_idx.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=np.int64)
        tree = SWCTree(ids=ids, types=arr[:, 1].astype(np.int64), xyz=arr[:, 2:5].copy(), radius=arr[:, 5].copy(),
                       parent_idx=parent_idx, allen_ids=np.asarray(allen, dtype=np.int64))
        tree.name = str(nrn.get("idString", nrn.get("id", len(out))))  # type: ignore[attr-defined]
        out.append(tree)
    return out


def tree_to_swc_text(tree: SWCTree) -> str:
    lines = ["# id type x y z radius parent  (CCFv3 um)"]
    for k in range(len(tree)):
        p = tree.ids[tree.parent_idx[k]] if tree.parent_idx[k] >= 0 else -1
        lines.append(f"{tree.ids[k]} {tree.types[k]} {tree.xyz[k,0]:.2f} {tree.xyz[k,1]:.2f} {tree.xyz[k,2]:.2f} "
                     f"{tree.radius[k]:.3f} {p}")
    return "\n".join(lines) + "\n"


def mouselight_json_to_swc_files(json_path: Union[str, Path], out_dir: Union[str, Path]) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trees = mouselight_json_neurons(json_path)
    meta = []
    for t in trees:
        name = getattr(t, "name", "neuron")
        (out_dir / f"{name}.swc").write_text(tree_to_swc_text(t))
        sx, sy, sz = t.soma_xyz()
        meta.append({"name": name, "soma_x": sx, "soma_y": sy, "soma_z": sz, "n_nodes": len(t),
                     "n_axon_nodes": int((t.types == AXON).sum())})
    pd.DataFrame(meta).to_csv(out_dir / "metadata.csv", index=False)
    return len(trees)


# ---------------------------------------------------------------------- geometry -> structures
def axon_terminals(tree: SWCTree) -> np.ndarray:
    """Coordinates of axon tips (axon nodes without children)."""
    has_child = np.zeros(len(tree), dtype=bool)
    has_child[tree.parent_idx[tree.parent_idx >= 0]] = True
    m = (tree.types == AXON) & ~has_child
    return tree.xyz[m]


def axon_segments(tree: SWCTree) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoints and lengths of axon segments (child node of type axon with a parent)."""
    m = (tree.types == AXON) & (tree.parent_idx >= 0)
    child = tree.xyz[m]
    parent = tree.xyz[tree.parent_idx[m]]
    return 0.5 * (child + parent), np.linalg.norm(child - parent, axis=1)


def points_to_voxels(points_um: np.ndarray, resolution_um: float, shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel indices (n x 3) and an in-bounds mask for points in µm."""
    ijk = np.floor(np.asarray(points_um, float) / resolution_um).astype(np.int64)
    shape = np.asarray(shape)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    return ijk, ok


def assign_points_to_structures(points_um: np.ndarray, annotation: np.ndarray, resolution_um: float) -> np.ndarray:
    """Structure id of the annotation voxel containing each point (0 = outside brain / volume)."""
    ijk, ok = points_to_voxels(points_um, resolution_um, annotation.shape)
    out = np.zeros(len(ijk), dtype=np.int64)
    if ok.any():
        out[ok] = annotation[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def collapse_to_summary(struct_ids: np.ndarray, ancestor_map: Mapping[int, Sequence[int]],
                        summary_ids: Iterable[int]) -> np.ndarray:
    """Map fine structure ids to the summary structure that contains them (0 if none)."""
    summary = set(int(s) for s in summary_ids)
    cache: Dict[int, int] = {}
    out = np.zeros(len(struct_ids), dtype=np.int64)
    for k, sid in enumerate(struct_ids):
        sid = int(sid)
        if sid not in cache:
            hit = 0
            for anc in ancestor_map.get(sid, [sid]):
                if int(anc) in summary:
                    hit = int(anc)
                    break
            cache[sid] = hit
        out[k] = cache[sid]
    return out


def hemisphere_labels(points_um: np.ndarray, soma_xyz: np.ndarray, midline_um: float = CCF_MIDLINE_UM) -> np.ndarray:
    """'ipsi' / 'contra' for each point relative to the soma's hemisphere (z axis = left-right)."""
    soma_right = soma_xyz[2] >= midline_um
    pts_right = np.asarray(points_um)[:, 2] >= midline_um
    return np.where(pts_right == soma_right, "ipsi", "contra")


def per_neuron_target_vector(tree: SWCTree, annotation: np.ndarray, resolution_um: float,
                             ancestor_map: Mapping[int, Sequence[int]], summary_ids: Sequence[int],
                             acronyms: Mapping[int, str], weight: str = "length", split_hemisphere: bool = True,
                             exclude_soma_structure: bool = False, midline_um: Optional[float] = None) -> pd.Series:
    """Fraction of axon (length or terminals) per target summary structure (optionally x hemisphere).

    Index labels are ``"<acronym>_<ipsi|contra>"`` (matching ``allen_connectivity.projection_matrix``)
    or just ``"<acronym>"`` when ``split_hemisphere`` is False. The left-right midline defaults to
    half the annotation's z extent (5700 µm for the 25 µm CCFv3 volume).
    """
    if midline_um is None:
        midline_um = annotation.shape[2] * resolution_um / 2.0
    if weight == "length":
        pts, w = axon_segments(tree)
    elif weight == "terminals":
        pts = axon_terminals(tree)
        w = np.ones(len(pts))
    else:
        raise ValueError("weight must be 'length' or 'terminals'")
    if len(pts) == 0:
        return pd.Series(dtype=float)
    fine = assign_points_to_structures(pts, annotation, resolution_um)
    summ = collapse_to_summary(fine, ancestor_map, summary_ids)
    soma = tree.soma_xyz()
    labels = np.array([acronyms.get(int(s), "") for s in summ], dtype=object)
    if split_hemisphere:
        hemi = hemisphere_labels(pts, soma, midline_um=midline_um)
        labels = np.array([f"{a}_{h}" if a else "" for a, h in zip(labels, hemi)], dtype=object)
    keep = labels != ""
    if exclude_soma_structure:
        soma_struct = collapse_to_summary(assign_points_to_structures(soma[None, :], annotation, resolution_um),
                                          ancestor_map, summary_ids)[0]
        soma_acr = acronyms.get(int(soma_struct), "")
        if soma_acr:
            keep &= ~pd.Series(labels).str.startswith(soma_acr + "_").to_numpy() & (labels != soma_acr)
    vec = pd.Series(w[keep]).groupby(labels[keep]).sum()
    total = vec.sum()
    return (vec / total if total > 0 else vec).sort_index()


def neuron_matrix(vectors: Sequence[pd.Series], names: Optional[Sequence[str]] = None,
                  columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Stack per-neuron target vectors into a neurons x targets matrix (missing = 0)."""
    df = pd.DataFrame(list(vectors)).fillna(0.0)
    if names is not None:
        df.index = list(names)
    if columns is not None:
        df = df.reindex(columns=list(columns), fill_value=0.0)
    return df


def soma_summary_structure(tree: SWCTree, annotation: np.ndarray, resolution_um: float,
                           ancestor_map: Mapping[int, Sequence[int]], summary_ids: Sequence[int]) -> int:
    fine = assign_points_to_structures(tree.soma_xyz()[None, :], annotation, resolution_um)
    return int(collapse_to_summary(fine, ancestor_map, summary_ids)[0])


def allen_id_agreement(tree: SWCTree, annotation: np.ndarray, resolution_um: float) -> float:
    """Fraction of nodes whose own ``allen_ids`` (e.g. MouseLight) equal our annotation lookup."""
    if tree.allen_ids is None:
        return float("nan")
    ours = assign_points_to_structures(tree.xyz, annotation, resolution_um)
    m = tree.allen_ids > 0
    return float(np.mean(ours[m] == tree.allen_ids[m])) if m.any() else float("nan")


__all__ = ["SWCTree", "parse_swc_text", "read_swc", "mouselight_json_neurons", "mouselight_json_to_swc_files",
           "tree_to_swc_text", "axon_terminals", "axon_segments", "points_to_voxels", "assign_points_to_structures",
           "collapse_to_summary", "hemisphere_labels", "per_neuron_target_vector", "neuron_matrix",
           "soma_summary_structure", "allen_id_agreement", "CCF_MIDLINE_UM"]
