"""SWC parsing and morphology features relevant to the extracellular waveform."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
TYPE_NAMES = {SOMA: "soma", AXON: "axon", BASAL: "basal", APICAL: "apical"}


def parse_swc_text(text: str) -> pd.DataFrame:
    """Parse SWC text into a DataFrame with columns id, type, x, y, z, r, parent (ids as ints)."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        rows.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                     float(parts[5]), int(parts[6])))
    df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
    if df.empty:
        raise ValueError("no SWC rows parsed")
    return df.set_index("id", drop=False)


def read_swc(path: Path) -> pd.DataFrame:
    return parse_swc_text(Path(path).read_text())


def soma_centre(tree: pd.DataFrame) -> np.ndarray:
    soma = tree[tree["type"] == SOMA]
    if soma.empty:
        root = tree[tree["parent"] == -1].iloc[0]
        return root[["x", "y", "z"]].to_numpy(float)
    return soma[["x", "y", "z"]].mean().to_numpy(float)


def centre_on_soma(tree: pd.DataFrame) -> pd.DataFrame:
    c = soma_centre(tree)
    out = tree.copy()
    out[["x", "y", "z"]] = out[["x", "y", "z"]].to_numpy(float) - c
    return out


@dataclass
class Segment:
    """A frustum segment between a node and its parent."""

    start: np.ndarray   # (3,)
    end: np.ndarray     # (3,)
    radius: float
    seg_type: int
    path_dist: float    # path distance of the segment midpoint from the soma (um)


def segments(tree: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised segment arrays: starts (n,3), ends (n,3), radii (n,), types (n,), path distance to soma (n,).

    Segments are child-parent pairs excluding soma-soma pairs. Path distance is measured at
    the segment midpoint along the tree from the root.
    """
    t = tree
    has_parent = t["parent"] != -1
    child = t[has_parent]
    parent = t.loc[child["parent"].to_numpy()]
    starts = parent[["x", "y", "z"]].to_numpy(float)
    ends = child[["x", "y", "z"]].to_numpy(float)
    radii = 0.5 * (child["r"].to_numpy(float) + parent["r"].to_numpy(float))
    types = child["type"].to_numpy(int)
    keep = ~((child["type"].to_numpy() == SOMA) & (parent["type"].to_numpy() == SOMA))
    # path distances via cumulative sum along parent chain
    length = np.linalg.norm(ends - starts, axis=1)
    dist_to_root: Dict[int, float] = {}
    order = child.index.to_numpy()
    seg_len = dict(zip(order, length))
    for nid in t.index:
        if t.at[nid, "parent"] == -1:
            dist_to_root[nid] = 0.0
    # iterate until all resolved (parents precede children in valid SWC, but be safe)
    pending = [nid for nid in order]
    while pending:
        remaining = []
        for nid in pending:
            p = int(t.at[nid, "parent"])
            if p in dist_to_root:
                dist_to_root[nid] = dist_to_root[p] + seg_len[nid]
            else:
                remaining.append(nid)
        if len(remaining) == len(pending):
            raise ValueError("SWC has orphan nodes")
        pending = remaining
    mid = np.array([dist_to_root[n] - 0.5 * seg_len[n] for n in order])
    return starts[keep], ends[keep], radii[keep], types[keep], mid[keep]


def compartmentalize(tree: pd.DataFrame, max_len: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split long segments so no compartment exceeds ``max_len`` um (for the line-source model)."""
    s, e, r, ty, d = segments(tree)
    out_s, out_e, out_r, out_t, out_d = [], [], [], [], []
    for i in range(len(s)):
        L = np.linalg.norm(e[i] - s[i])
        n = max(1, int(np.ceil(L / max_len)))
        for k in range(n):
            a = s[i] + (e[i] - s[i]) * k / n
            b = s[i] + (e[i] - s[i]) * (k + 1) / n
            out_s.append(a)
            out_e.append(b)
            out_r.append(r[i])
            out_t.append(ty[i])
            out_d.append(d[i] - 0.5 * L + L * (k + 0.5) / n)
    return (np.array(out_s), np.array(out_e), np.array(out_r), np.array(out_t), np.array(out_d))


def morphology_features(tree: pd.DataFrame, shells: Sequence[float] = (50.0, 100.0, 200.0, 400.0)) -> Dict[str, float]:
    """Interpretable morphology features hypothesised to shape the extracellular waveform.

    Returns a flat dict: soma radius, number of stems, per-type length/surface area, dendritic
    surface area within radial (Euclidean) shells around the soma, dendritic polarity
    (|mean unit vector| of dendritic segment positions; 1 = fully one-sided), vertical
    asymmetry (apical direction), apical presence and trunk radius, and the axon origin
    distance from the soma centre.
    """
    t = centre_on_soma(tree)
    soma = t[t["type"] == SOMA]
    feats: Dict[str, float] = {}
    if len(soma) == 1:
        feats["soma_radius"] = float(soma["r"].iloc[0])
    elif len(soma) > 1:
        pts = soma[["x", "y", "z"]].to_numpy(float)
        feats["soma_radius"] = float(np.linalg.norm(pts - pts.mean(0), axis=1).mean() + soma["r"].mean())
    else:
        feats["soma_radius"] = float("nan")
    soma_ids = set(soma.index)
    feats["n_stems"] = float(((t["parent"].isin(soma_ids)) & (~t["type"].isin([SOMA]))).sum())

    s, e, r, ty, d = segments(t)
    length = np.linalg.norm(e - s, axis=1)
    area = 2 * np.pi * r * length
    mid = 0.5 * (s + e)
    rad = np.linalg.norm(mid, axis=1)
    for code, name in TYPE_NAMES.items():
        if code == SOMA:
            continue
        m = ty == code
        feats[f"{name}_length"] = float(length[m].sum())
        feats[f"{name}_area"] = float(area[m].sum())
    dend = (ty == BASAL) | (ty == APICAL)
    prev = 0.0
    for sh in shells:
        m = dend & (rad > prev) & (rad <= sh)
        feats[f"dend_area_{int(prev)}_{int(sh)}"] = float(area[m].sum())
        prev = sh
    feats["dend_area_total"] = float(area[dend].sum())
    if dend.any():
        w = area[dend] / max(area[dend].sum(), 1e-9)
        unit = mid[dend] / np.maximum(rad[dend][:, None], 1e-9)
        pol = (w[:, None] * unit).sum(0)
        feats["dend_polarity"] = float(np.linalg.norm(pol))
        feats["dend_vertical_asymmetry"] = float(pol[1])  # y is the pia-white-matter axis in Allen SWCs
        feats["dend_mean_radial_extent"] = float((w * rad[dend]).sum())
    else:
        feats["dend_polarity"] = feats["dend_vertical_asymmetry"] = feats["dend_mean_radial_extent"] = float("nan")
    api = ty == APICAL
    feats["has_apical"] = float(api.any())
    feats["apical_trunk_radius"] = float(r[api & (d < 30.0)].mean()) if (api & (d < 30.0)).any() else 0.0
    ax = ty == AXON
    if ax.any():
        first = np.argmin(d[ax])
        feats["axon_origin_dist"] = float(rad[ax][first])
        feats["axon_origin_vertical"] = float(mid[ax][first][1])
    else:
        feats["axon_origin_dist"] = feats["axon_origin_vertical"] = float("nan")
    feats["n_nodes"] = float(len(t))
    return feats


def features_table(swc_paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for p in swc_paths:
        try:
            f = morphology_features(read_swc(p))
        except Exception as exc:  # noqa: BLE001 - keep going over a large corpus
            f = {"error": str(exc)}
        f["path"] = str(p)
        rows.append(f)
    return pd.DataFrame(rows).set_index("path")


def rotate_about_y(tree: pd.DataFrame, angle_rad: float) -> pd.DataFrame:
    """Rotate the morphology about the vertical (y) axis, e.g. to vary the probe-facing side."""
    out = tree.copy()
    c, s_ = np.cos(angle_rad), np.sin(angle_rad)
    x, z = out["x"].to_numpy(float), out["z"].to_numpy(float)
    out["x"], out["z"] = c * x + s_ * z, -s_ * x + c * z
    return out
