"""Detect spine-like structures in SWC reconstructions and audit spine annotation.

Spines can be encoded in SWC in two ways: (1) custom type codes (> 4; NeuroMorpho keeps them in
some files, and tools such as Neurolucida export spines as separate types), and (2) very short
terminal branches (one to a few nodes, a few micrometres) hanging off dendritic segments. The
audit reports both, plus a spine-like density per micrometre of dendrite excluding the spine
segments themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
DENDRITE_TYPES = (BASAL, APICAL)


def read_swc(source: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Parse an SWC path or text into arrays with ids remapped to 0..n-1."""
    p = Path(str(source))
    text = p.read_text() if p.exists() else str(source)
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 7:
            continue
        rows.append([float(v) for v in parts[:7]])
    if not rows:
        raise ValueError("no SWC rows")
    arr = np.asarray(rows, dtype=float)
    remap = {int(i): k for k, i in enumerate(arr[:, 0].astype(int))}
    parent = np.array([remap.get(int(v), -1) if v >= 0 else -1 for v in arr[:, 6]], dtype=int)
    return {"type": arr[:, 1].astype(int), "xyz": arr[:, 2:5], "radius": arr[:, 5], "parent": parent}


def _children(parent: np.ndarray) -> List[List[int]]:
    ch: List[List[int]] = [[] for _ in range(len(parent))]
    for i, p in enumerate(parent):
        if p >= 0:
            ch[p].append(i)
    return ch


def _seg_len(swc: Dict[str, np.ndarray]) -> np.ndarray:
    par = swc["parent"]
    d = np.zeros(len(par))
    ok = par >= 0
    d[ok] = np.linalg.norm(swc["xyz"][ok] - swc["xyz"][par[ok]], axis=1)
    return d


def detect_spines(swc: Dict[str, np.ndarray], max_length_um: float = 3.5, max_nodes: int = 3,
                  min_parent_radius_ratio: float = 0.0) -> Dict[str, np.ndarray]:
    """Flag spine-like nodes.

    Returns boolean masks ``by_type`` (custom type codes > 4 attached to dendrites), ``by_geometry``
    (terminal branches of <= ``max_nodes`` nodes and total length <= ``max_length_um`` whose
    attachment node is a dendrite with more than one child, i.e. the branch is a side protrusion,
    not a dendritic ending), and the union ``spine``. ``attach`` holds the attachment node of each
    spine node (-1 for non-spine nodes).
    """
    typ, par = swc["type"], swc["parent"]
    n = len(typ)
    children = _children(par)
    seg = _seg_len(swc)
    by_type = np.zeros(n, dtype=bool)
    by_geom = np.zeros(n, dtype=bool)
    attach = np.full(n, -1, dtype=int)
    # custom types: walk up to the first non-custom ancestor
    for i in range(n):
        if typ[i] > 4:
            a = i
            while a >= 0 and typ[a] > 4:
                a = par[a]
            if a >= 0 and typ[a] in DENDRITE_TYPES:
                by_type[i] = True
                attach[i] = a
    # geometry: terminal branches
    tips = [i for i in range(n) if not children[i] and typ[i] in DENDRITE_TYPES + (5, 6, 7)]
    for tip in tips:
        path = [tip]
        cur = tip
        while par[cur] >= 0 and len(children[par[cur]]) == 1 and typ[par[cur]] != SOMA:
            cur = par[cur]
            path.append(cur)
        a = par[cur]
        if a < 0 or typ[a] not in DENDRITE_TYPES or len(children[a]) < 2:
            continue  # a dendritic ending, not a side protrusion
        length = float(np.sum(seg[path]))
        if len(path) <= max_nodes and length <= max_length_um:
            if min_parent_radius_ratio > 0 and swc["radius"][a] > 0:
                if np.mean(swc["radius"][path]) / swc["radius"][a] > 1.0 / min_parent_radius_ratio:
                    continue
            by_geom[path] = True
            attach[path] = a
    spine = by_type | by_geom
    return {"by_type": by_type, "by_geometry": by_geom, "spine": spine, "attach": attach}


def audit_tree(swc: Dict[str, np.ndarray], **kw) -> Dict[str, float]:
    """Per-reconstruction audit record: counts, dendritic length (spines excluded), spine-like density."""
    det = detect_spines(swc, **kw)
    typ = swc["type"]
    seg = _seg_len(swc)
    dend = np.isin(typ, DENDRITE_TYPES) & ~det["spine"]
    dend_len = float(seg[dend].sum())
    # count spines as distinct attachment events (a multi-node spine counts once)
    n_type = len(np.unique(det["attach"][det["by_type"]] * 1_000_003 + _root_of_spine(det, swc)[det["by_type"]]))
    n_geom = len(np.unique(_root_of_spine(det, swc)[det["by_geometry"]]))
    n_spines = len(np.unique(_root_of_spine(det, swc)[det["spine"]]))
    density = n_spines / dend_len if dend_len > 0 else np.nan
    return {
        "n_nodes": int(len(typ)),
        "n_custom_type_nodes": int(np.sum(typ > 4)),
        "n_spines_by_type": int(n_type),
        "n_spines_by_geometry": int(n_geom),
        "n_spines": int(n_spines),
        "dendrite_length_um": dend_len,
        "spine_density_per_um": float(density),
        "has_spine_annotation": bool(n_spines >= 10 and 0.05 <= density <= 8.0),
        "fraction_nodes_spine": float(det["spine"].mean()),
    }


def _root_of_spine(det: Dict[str, np.ndarray], swc: Dict[str, np.ndarray]) -> np.ndarray:
    """For each node, the first spine node on the path from the attachment (spine 'root'); -1 otherwise."""
    par = swc["parent"]
    root = np.full(len(par), -1, dtype=int)
    for i in np.flatnonzero(det["spine"]):
        cur = i
        while par[cur] >= 0 and det["spine"][par[cur]]:
            cur = par[cur]
        root[i] = cur
    return root


def audit_file(path: Union[str, Path], **kw) -> Dict[str, object]:
    rec: Dict[str, object] = {"path": str(path)}
    try:
        swc = read_swc(path)
        rec.update(audit_tree(swc, **kw))
        rec["error"] = ""
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)
    return rec


__all__ = ["read_swc", "detect_spines", "audit_tree", "audit_file", "SOMA", "AXON", "BASAL", "APICAL"]
