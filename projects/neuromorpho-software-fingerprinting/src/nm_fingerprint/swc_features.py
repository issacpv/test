"""SWC parsing, morphometrics and "sampling fingerprint" features.

An SWC file is a table ``id type x y z radius parent`` (parent = -1 for the root). We keep it as
a dictionary of numpy arrays (``id, type, xyz, radius, parent``) with ids remapped to 0..n-1.

Two feature families are computed:

* :func:`morphometrics` - biology-facing summaries (length, branching, extents, tortuosity ...).
* :func:`sampling_fingerprint` - tracing-pipeline-facing summaries (node spacing distribution,
  coordinate precision, radius uniqueness, z quantisation ...). These are the features a
  reconstruction tool leaves behind irrespective of the neuron.
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4


def read_swc(source: Union[str, Path, io.TextIOBase]) -> Dict[str, np.ndarray]:
    """Parse an SWC file/path/text. Ids are remapped to 0..n-1 preserving order; parent -1 -> -1."""
    if isinstance(source, io.TextIOBase):
        text = source.read()
    else:
        p = Path(str(source))
        text = p.read_text() if p.exists() else str(source)
    rows: List[List[float]] = []
    raw_lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 7:
            continue
        rows.append([float(v) for v in parts[:7]])
        raw_lines.append(s)
    if not rows:
        raise ValueError("no SWC rows found")
    arr = np.asarray(rows, dtype=float)
    ids = arr[:, 0].astype(int)
    remap = {int(i): k for k, i in enumerate(ids)}
    parent = np.array([remap.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=int)
    return {"id": np.arange(len(ids)), "type": arr[:, 1].astype(int), "xyz": arr[:, 2:5], "radius": arr[:, 5],
            "parent": parent, "raw_lines": np.asarray(raw_lines, dtype=object)}


def write_swc(swc: Dict[str, np.ndarray], path: Union[str, Path], header: str = "") -> None:
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    for i in range(len(swc["id"])):
        x, y, z = swc["xyz"][i]
        lines.append(f"{i + 1} {int(swc['type'][i])} {x:.6f} {y:.6f} {z:.6f} {swc['radius'][i]:.6f} "
                     f"{int(swc['parent'][i]) + 1 if swc['parent'][i] >= 0 else -1}")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- helpers
def _segment_lengths(swc: Dict[str, np.ndarray]) -> np.ndarray:
    par = swc["parent"]
    ok = par >= 0
    d = np.zeros(len(par))
    d[ok] = np.linalg.norm(swc["xyz"][ok] - swc["xyz"][par[ok]], axis=1)
    return d


def _children_counts(swc: Dict[str, np.ndarray]) -> np.ndarray:
    par = swc["parent"]
    counts = np.zeros(len(par), dtype=int)
    np.add.at(counts, par[par >= 0], 1)
    return counts


def _branch_orders(swc: Dict[str, np.ndarray]) -> np.ndarray:
    par = swc["parent"]
    nchild = _children_counts(swc)
    order = np.zeros(len(par), dtype=int)
    for i in range(len(par)):  # SWC rows are parent-before-child in valid files
        p = par[i]
        if p < 0:
            continue
        order[i] = order[p] + (1 if (nchild[p] > 1 and swc["type"][p] != SOMA) else 0)
    return order


def _branches(swc: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Node-index paths from each branch start (child of soma/bifurcation) to the next bifurcation/tip."""
    par = swc["parent"]
    nchild = _children_counts(swc)
    children: Dict[int, List[int]] = {}
    for i, p in enumerate(par):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    starts = [i for i, p in enumerate(par) if p >= 0 and (nchild[p] != 1 or swc["type"][p] == SOMA) and swc["type"][i] != SOMA]
    out = []
    for s in starts:
        path = [int(par[s]), s]
        cur = s
        while nchild[cur] == 1:
            cur = children[cur][0]
            path.append(cur)
        out.append(np.asarray(path))
    return out


# --------------------------------------------------------------------------- morphometrics
def morphometrics(swc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Standard summaries (dendrites + axon pooled; per-type lengths reported separately)."""
    seg = _segment_lengths(swc)
    typ = swc["type"]
    nchild = _children_counts(swc)
    neurite = typ != SOMA
    n_bif = int(np.sum((nchild > 1) & neurite))
    n_tips = int(np.sum((nchild == 0) & neurite))
    n_stems = int(np.sum([typ[i] != SOMA and (swc["parent"][i] >= 0) and typ[swc["parent"][i]] == SOMA for i in range(len(typ))]))
    orders = _branch_orders(swc)
    branches = _branches(swc)
    blen, tort = [], []
    for b in branches:
        L = float(np.sum(seg[b[1:]]))
        euc = float(np.linalg.norm(swc["xyz"][b[-1]] - swc["xyz"][b[0]]))
        blen.append(L)
        tort.append(L / euc if euc > 1e-9 else np.nan)
    xyz = swc["xyz"][neurite] if neurite.any() else swc["xyz"]
    ext = xyz.max(axis=0) - xyz.min(axis=0)
    cen = xyz - xyz.mean(axis=0)
    ev = np.linalg.eigvalsh(cen.T @ cen / max(len(cen), 1))
    ev = np.maximum(ev, 0)
    pr = float(ev.sum() ** 2 / np.sum(ev ** 2)) if ev.sum() > 0 else np.nan  # participation ratio (1..3)
    return {
        "n_nodes": int(len(typ)),
        "total_length": float(seg[neurite].sum()),
        "length_axon": float(seg[typ == AXON].sum()),
        "length_basal": float(seg[typ == BASAL].sum()),
        "length_apical": float(seg[typ == APICAL].sum()),
        "n_bifurcations": n_bif,
        "n_tips": n_tips,
        "n_stems": n_stems,
        "max_branch_order": int(orders.max()) if len(orders) else 0,
        "mean_branch_length": float(np.mean(blen)) if blen else np.nan,
        "cv_branch_length": float(np.std(blen) / np.mean(blen)) if blen and np.mean(blen) > 0 else np.nan,
        "mean_tortuosity": float(np.nanmean(tort)) if tort else np.nan,
        "width": float(ext[0]), "height": float(ext[1]), "depth": float(ext[2]),
        "participation_ratio": pr,
        "planarity": float(1 - ev[0] / ev.sum()) if ev.sum() > 0 else np.nan,
        "mean_radius": float(swc["radius"][neurite].mean()) if neurite.any() else np.nan,
        "soma_radius": float(swc["radius"][typ == SOMA].max()) if np.any(typ == SOMA) else np.nan,
    }


# --------------------------------------------------------------------------- fingerprint
def _decimals(values: np.ndarray, max_dec: int = 6) -> np.ndarray:
    """Number of significant decimals for each value (0..max_dec)."""
    out = np.zeros(len(values), dtype=int)
    for k in range(max_dec, 0, -1):
        scaled = values * 10 ** k
        out[np.abs(scaled - np.round(scaled)) > 1e-6 * 10 ** k] = k + 1
    out = np.minimum(out, max_dec)
    # values exactly representable with fewer decimals get the smallest k such that round(v*10^k)==v*10^k
    dec = np.full(len(values), max_dec, dtype=int)
    for k in range(max_dec, -1, -1):
        scaled = values * 10 ** k
        exact = np.abs(scaled - np.round(scaled)) < 1e-9 * max(1.0, float(np.max(np.abs(scaled))))
        dec[exact] = k
    return dec


def sampling_fingerprint(swc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Pipeline-facing features: node spacing, coordinate precision, radius and z quantisation, collinearity."""
    seg = _segment_lengths(swc)
    neurite = swc["type"] != SOMA
    d = seg[neurite & (swc["parent"] >= 0)]
    d = d[d > 0]
    out: Dict[str, float] = {}
    if len(d) >= 5:
        q = np.percentile(d, [10, 50, 90])
        out.update({"spacing_q10": float(q[0]), "spacing_q50": float(q[1]), "spacing_q90": float(q[2]),
                    "spacing_cv": float(d.std() / d.mean()), "spacing_iqr_ratio": float((q[2] - q[0]) / q[1])})
        hist, edges = np.histogram(d, bins=50)
        mode = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        out["spacing_grid_fraction"] = float(np.mean(np.abs(d - mode) <= 0.05 * mode))
        out["spacing_mode"] = float(mode)
    else:
        out.update({k: np.nan for k in ("spacing_q10", "spacing_q50", "spacing_q90", "spacing_cv", "spacing_iqr_ratio",
                                        "spacing_grid_fraction", "spacing_mode")})
    xyz = swc["xyz"]
    for j, name in enumerate("xyz"):
        out[f"decimals_{name}"] = float(np.median(_decimals(xyz[:, j])))
    r = swc["radius"][neurite] if neurite.any() else swc["radius"]
    out["radius_unique_fraction"] = float(len(np.unique(np.round(r, 6))) / max(len(r), 1))
    out["radius_decimals"] = float(np.median(_decimals(r)))
    cnt = Counter(np.round(r, 6).tolist())
    out["radius_mode_fraction"] = float(cnt.most_common(1)[0][1] / max(len(r), 1)) if len(r) else np.nan
    par = swc["parent"]
    ok = (par >= 0) & neurite
    out["radius_equals_parent_fraction"] = float(np.mean(np.isclose(swc["radius"][ok], swc["radius"][par[ok]]))) if ok.any() else np.nan
    z = xyz[:, 2]
    out["z_unique_fraction"] = float(len(np.unique(np.round(z, 6))) / max(len(z), 1))
    dz = np.abs(np.diff(np.unique(np.round(z, 6))))
    out["z_step_min"] = float(dz.min()) if len(dz) else np.nan
    if len(dz) >= 3:
        zc = Counter(np.round(dz, 4).tolist())
        out["z_step_mode_fraction"] = float(zc.most_common(1)[0][1] / len(dz))
    else:
        out["z_step_mode_fraction"] = np.nan
    # collinearity of consecutive segments (exactly straight runs are typical of some resamplers)
    gp = par[par >= 0]
    idx = np.flatnonzero(par >= 0)
    mask = gp >= 0
    i2 = idx[mask]
    p1 = par[i2]
    p0 = par[p1]
    valid = p0 >= 0
    i2, p1, p0 = i2[valid], p1[valid], p0[valid]
    if len(i2):
        v1 = xyz[p1] - xyz[p0]
        v2 = xyz[i2] - xyz[p1]
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        good = (n1 > 1e-9) & (n2 > 1e-9)
        cos = np.sum(v1[good] * v2[good], axis=1) / (n1[good] * n2[good])
        out["collinear_fraction"] = float(np.mean(cos > 0.9999)) if good.any() else np.nan
        out["mean_turning_angle"] = float(np.mean(np.arccos(np.clip(cos, -1, 1)))) if good.any() else np.nan
    else:
        out["collinear_fraction"], out["mean_turning_angle"] = np.nan, np.nan
    out["nodes_per_um"] = float(np.sum(neurite) / max(seg[neurite].sum(), 1e-9))
    return out


def feature_vector(swc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Morphometrics + sampling fingerprint in one flat dict (prefixes ``m_`` and ``s_``)."""
    out = {f"m_{k}": v for k, v in morphometrics(swc).items()}
    out.update({f"s_{k}": v for k, v in sampling_fingerprint(swc).items()})
    return out


# --------------------------------------------------------------------------- resampling
def resample_swc(swc: Dict[str, np.ndarray], spacing: float = 1.0, precision: Optional[int] = 2,
                 drop_radius: bool = False) -> Dict[str, np.ndarray]:
    """Resample every branch to (approximately) uniform node spacing and round coordinates.

    This is the harmonisation step that removes sampling artefacts: bifurcations and tips are
    kept exactly, intermediate nodes are re-interpolated along the original polyline.
    """
    branches = _branches(swc)
    typ, xyz, rad = swc["type"], swc["xyz"], swc["radius"]
    new_xyz: List[np.ndarray] = []
    new_type: List[int] = []
    new_rad: List[float] = []
    new_par: List[int] = []
    index_of: Dict[int, int] = {}

    def add(point: np.ndarray, t: int, r: float, parent: int) -> int:
        new_xyz.append(point)
        new_type.append(t)
        new_rad.append(r)
        new_par.append(parent)
        return len(new_xyz) - 1

    roots = np.flatnonzero(swc["parent"] < 0)
    for rt in roots:
        index_of[int(rt)] = add(xyz[rt], int(typ[rt]), float(rad[rt]), -1)
    # soma nodes that are not roots (multi-point somas) are kept as they are
    for i in np.flatnonzero((typ == SOMA) & (swc["parent"] >= 0)):
        p = int(swc["parent"][i])
        if p in index_of:
            index_of[int(i)] = add(xyz[i], SOMA, float(rad[i]), index_of[p])
    for b in branches:
        start = int(b[0])
        if start not in index_of:
            continue
        pts = xyz[b]
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total = cum[-1]
        n_new = max(1, int(round(total / spacing)))
        targets = np.linspace(0, total, n_new + 1)[1:]
        parent = index_of[start]
        t_branch = int(typ[b[-1]])
        r_branch = float(np.mean(rad[b[1:]]))
        for k, s in enumerate(targets):
            j = int(np.searchsorted(cum, s, side="right") - 1)
            j = min(j, len(seglen) - 1)
            frac = (s - cum[j]) / seglen[j] if seglen[j] > 0 else 0.0
            point = pts[j] + frac * (pts[j + 1] - pts[j])
            r = r_branch if not drop_radius else 1.0
            parent = add(point, t_branch, r, parent)
        index_of[int(b[-1])] = parent
    arr = np.asarray(new_xyz)
    if precision is not None:
        arr = np.round(arr, precision)
    return {"id": np.arange(len(arr)), "type": np.asarray(new_type), "xyz": arr, "radius": np.asarray(new_rad),
            "parent": np.asarray(new_par), "raw_lines": np.asarray([], dtype=object)}


__all__ = ["read_swc", "write_swc", "morphometrics", "sampling_fingerprint", "feature_vector", "resample_swc",
           "SOMA", "AXON", "BASAL", "APICAL"]
