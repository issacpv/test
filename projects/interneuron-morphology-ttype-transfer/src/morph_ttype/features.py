"""Morphology representations: morphometrics, density maps, persistence-style summaries.

Every representation is available in an absolute and a *scale-normalised* form,
because human interneurons are larger than mouse ones and absolute scale is the
first thing that breaks cross-species transfer.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .swc import AXON, DENDRITES, SOMA, Neuron

POSITION_FEATURES = ("soma_depth_norm", "layer_index")


def _branch_orders(n: Neuron) -> np.ndarray:
    ch = n.children()
    order = np.zeros(n.n, dtype=int)
    stack = [i for i, p in enumerate(n.parent) if p < 0]
    while stack:
        i = stack.pop()
        bump = 1 if (len(ch[i]) > 1 and n.types[i] != SOMA) else 0
        for k in ch[i]:
            order[k] = order[i] + bump
            stack.append(k)
    return order


def _path_dist(n: Neuron) -> np.ndarray:
    seg = n.seg_length()
    ch = n.children()
    d = np.zeros(n.n)
    stack = [i for i, p in enumerate(n.parent) if p < 0]
    while stack:
        i = stack.pop()
        for k in ch[i]:
            d[k] = d[i] + seg[k]
            stack.append(k)
    return d


def morphometrics(n: Neuron, types: Sequence[int] = DENDRITES, prefix: str = "dend",
                  sholl_step: float = 25.0, sholl_max: float = 500.0) -> Dict[str, float]:
    """Per-neurite-class morphometrics (``prefix_*``)."""
    mask = np.isin(n.types, types)
    seg = n.seg_length()
    ch = n.children()
    nch = np.array([len(c) for c in ch])
    order = _branch_orders(n)
    pdist = _path_dist(n)
    c = n.soma_center()
    rel = n.xyz - c
    edist = np.linalg.norm(rel, axis=1)
    L = float(seg[mask].sum())
    n_bif = int((mask & (nch > 1)).sum())
    n_tip = int((mask & (nch == 0)).sum())
    out = {
        f"{prefix}_total_length": L,
        f"{prefix}_n_bifurcations": n_bif,
        f"{prefix}_n_tips": n_tip,
        f"{prefix}_n_stems": int((mask & (n.parent >= 0) & np.isin(n.parent, np.where(n.types == SOMA)[0])).sum()),
        f"{prefix}_max_order": int(order[mask].max()) if mask.any() else 0,
        f"{prefix}_mean_branch_length": L / (n_bif + n_tip) if (n_bif + n_tip) else float("nan"),
        f"{prefix}_max_euclid": float(edist[mask].max()) if mask.any() else 0.0,
        f"{prefix}_max_path": float(pdist[mask].max()) if mask.any() else 0.0,
        f"{prefix}_tortuosity": float(np.nanmean(pdist[mask & (nch == 0)] / np.maximum(edist[mask & (nch == 0)], 1e-6)))
        if (mask & (nch == 0)).any() else float("nan"),
        f"{prefix}_extent_x": float(np.ptp(rel[mask, 0])) if mask.any() else 0.0,
        f"{prefix}_extent_y": float(np.ptp(rel[mask, 1])) if mask.any() else 0.0,
        f"{prefix}_extent_z": float(np.ptp(rel[mask, 2])) if mask.any() else 0.0,
        f"{prefix}_centroid_y": float(rel[mask, 1].mean()) if mask.any() else 0.0,  # up/down bias along depth
        f"{prefix}_frac_above_soma": float((rel[mask, 1] < 0).mean()) if mask.any() else float("nan"),
    }
    radii = np.arange(sholl_step, sholl_max + 1e-9, sholl_step)
    m2 = mask & (n.parent >= 0)
    d1 = edist[m2]
    d0 = np.linalg.norm(n.xyz[n.parent[m2]] - c, axis=1)
    lo, hi = np.minimum(d0, d1), np.maximum(d0, d1)
    prof = np.array([((lo <= r) & (hi > r)).sum() for r in radii], float)
    for r, v in zip(radii, prof):
        out[f"{prefix}_sholl_{int(r)}"] = float(v)
    out[f"{prefix}_sholl_peak_radius"] = float(radii[int(np.argmax(prof))]) if prof.any() else 0.0
    return out


def density_map(n: Neuron, types: Sequence[int] = DENDRITES, bins: int = 20, half_width_um: float = 300.0,
                depth_norm: Optional[float] = None, prefix: str = "dmap") -> Dict[str, float]:
    """2-D (lateral x depth) density of neurite length around the soma, flattened to ``bins*bins`` features.

    Lateral axis = radial distance in xz; depth axis = y. When ``depth_norm`` (cortical
    thickness in um) is given the depth axis is expressed in thickness units (window
    +-0.5 thickness), which makes maps comparable across species; otherwise it uses
    +-``half_width_um``.
    """
    mask = np.isin(n.types, types) & (n.parent >= 0)
    if not mask.any():
        return {f"{prefix}_{i}": 0.0 for i in range(bins * bins)}
    c = n.soma_center()
    mid = 0.5 * (n.xyz[mask] + n.xyz[n.parent[mask]]) - c
    w = n.seg_length()[mask]
    lateral = np.sqrt(mid[:, 0] ** 2 + mid[:, 2] ** 2)
    depth = mid[:, 1]
    if depth_norm:
        depth = depth / depth_norm
        drange = (-0.5, 0.5)
        lrange = (0.0, 0.5)
        lateral = lateral / depth_norm
    else:
        drange = (-half_width_um, half_width_um)
        lrange = (0.0, half_width_um)
    H, _, _ = np.histogram2d(lateral, depth, bins=bins, range=[lrange, drange], weights=w)
    H = H / max(H.sum(), 1e-9)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(H.ravel())}


def persistence_summary(n: Neuron, types: Sequence[int] = DENDRITES, n_bins: int = 16, max_r: float = 500.0,
                        prefix: str = "pers") -> Dict[str, float]:
    """Branch-level (birth, death) radial-distance summary in the spirit of TMD persistence barcodes.

    For each unbranched section we record the radial distance at its start (birth) and
    end (death); the features are histograms of births, deaths and lifetimes.
    """
    ch = n.children()
    c = n.soma_center()
    edist = np.linalg.norm(n.xyz - c, axis=1)
    births: List[float] = []
    deaths: List[float] = []
    for i in range(n.n):
        p = n.parent[i]
        if n.types[i] not in types or p < 0:
            continue
        starts = (n.types[p] == SOMA) or (len(ch[p]) > 1)
        if not starts:
            continue
        j = i
        while len(ch[j]) == 1 and n.types[ch[j][0]] in types:
            j = ch[j][0]
        births.append(edist[i])
        deaths.append(edist[j])
    edges = np.linspace(0, max_r, n_bins + 1)
    out: Dict[str, float] = {}
    b = np.asarray(births)
    d = np.asarray(deaths)
    for name, arr in (("birth", b), ("death", d), ("life", np.abs(d - b) if len(b) else b)):
        h, _ = np.histogram(arr, bins=edges)
        h = h / max(h.sum(), 1)
        for k, v in enumerate(h):
            out[f"{prefix}_{name}_{k}"] = float(v)
    out[f"{prefix}_n_sections"] = float(len(b))
    return out


def scale_normalise(feats: Dict[str, float], prefix: str = "dend") -> Dict[str, float]:
    """Divide length-like features by total length and extents by max Euclidean extent."""
    L = feats.get(f"{prefix}_total_length", np.nan)
    E = feats.get(f"{prefix}_max_euclid", np.nan)
    out = dict(feats)
    for k, v in feats.items():
        if not k.startswith(prefix):
            continue
        if "sholl" in k and "peak" not in k:
            out[k] = v / max(feats.get(f"{prefix}_n_tips", 1), 1)
        elif k.endswith(("mean_branch_length", "max_path")):
            out[k] = v / L if L else np.nan
        elif k.endswith(("extent_x", "extent_y", "extent_z", "centroid_y", "sholl_peak_radius")):
            out[k] = v / E if E else np.nan
    out[f"{prefix}_bifs_per_length"] = feats.get(f"{prefix}_n_bifurcations", 0) / L if L else np.nan
    return out


def featurize(n: Neuron, soma_depth_norm: Optional[float] = None, layer_index: Optional[float] = None,
              cortical_thickness_um: Optional[float] = None, include_axon: bool = True,
              normalise: bool = False) -> Dict[str, float]:
    """All representations for one neuron, plus position features when given."""
    f = morphometrics(n, DENDRITES, "dend")
    f.update(density_map(n, DENDRITES, depth_norm=cortical_thickness_um, prefix="dmap_dend"))
    f.update(persistence_summary(n, DENDRITES, prefix="pers_dend"))
    if include_axon and (n.types == AXON).any():
        f.update(morphometrics(n, (AXON,), "axon"))
        f.update(density_map(n, (AXON,), depth_norm=cortical_thickness_um, prefix="dmap_axon"))
    if normalise:
        f = scale_normalise(f, "dend")
        if include_axon and (n.types == AXON).any():
            f = scale_normalise(f, "axon")
    f["has_axon"] = float((n.types == AXON).any())
    if soma_depth_norm is not None:
        f["soma_depth_norm"] = float(soma_depth_norm)
    if layer_index is not None:
        f["layer_index"] = float(layer_index)
    return f


def feature_table(neurons: Iterable[Neuron], ids: Sequence[str], **kw) -> pd.DataFrame:
    rows = {cid: featurize(n, **kw) for cid, n in zip(ids, neurons)}
    return pd.DataFrame.from_dict(rows, orient="index")


def drop_position(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in POSITION_FEATURES if c in df.columns])
