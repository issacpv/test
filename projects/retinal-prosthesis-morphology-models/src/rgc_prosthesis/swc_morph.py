"""SWC parsing, compartmentalisation, RGC morphometrics and synthetic axon / AIS generation.

Coordinates are micrometres. The retinal convention used throughout: the x-y plane is the
retinal surface, z increases from the vitreous (nerve-fibre layer, z ~ 0 at the soma layer)
into the inner plexiform layer (dendrites at positive z for a cell whose soma sits in the
ganglion cell layer). Datasets differ in their z convention; :func:`orient_z` flips a cell so
that its dendrites lie at positive z relative to the soma.

Region codes for the biophysical model: 1 soma, 2 axon (distal), 3 basal dendrite, 4 apical
dendrite (treated as dendrite), 5 hillock, 6 axon initial segment (sodium-channel band).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

SOMA, AXON, BASAL, APICAL, HILLOCK, AIS = 1, 2, 3, 4, 5, 6
DENDRITE_TYPES = (BASAL, APICAL)


def load_swc(path_or_text: str) -> pd.DataFrame:
    """Parse an SWC file path or SWC text; returns columns id, type, x, y, z, r, parent (index = id)."""
    text = path_or_text
    if "\n" not in path_or_text and len(path_or_text) < 4096:
        with open(path_or_text) as fh:
            text = fh.read()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        rows.append([int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), int(p[6])])
    df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
    return df.set_index("id", drop=False)


def soma_center(swc: pd.DataFrame) -> np.ndarray:
    s = swc[swc["type"] == SOMA]
    if s.empty:
        s = swc[swc["parent"] == -1]
    return s[["x", "y", "z"]].mean().to_numpy()


def soma_diameter(swc: pd.DataFrame) -> float:
    s = swc[swc["type"] == SOMA]
    if s.empty:
        return float(2 * swc.loc[swc["parent"] == -1, "r"].max())
    if len(s) == 1:
        return float(2 * s["r"].iloc[0])
    # multi-point soma contour: equivalent diameter of its xy extent
    ext = s[["x", "y"]].max() - s[["x", "y"]].min()
    return float(max(ext.mean(), 2 * s["r"].max()))


def orient_z(swc: pd.DataFrame) -> pd.DataFrame:
    """Flip z so that dendrites are at positive z relative to the soma (retinal convention)."""
    c = soma_center(swc)
    d = swc[swc["type"].isin(DENDRITE_TYPES)]
    if not d.empty and (d["z"].mean() - c[2]) < 0:
        out = swc.copy()
        out["z"] = 2 * c[2] - out["z"]
        return out
    return swc


def morphometrics(swc: pd.DataFrame) -> Dict[str, float]:
    """Soma diameter, dendritic field diameter (convex hull in xy), total dendritic length, branch
    points, stratification depth (mean dendritic z relative to soma) and spread, xy asymmetry."""
    c = soma_center(swc)
    d = swc[swc["type"].isin(DENDRITE_TYPES)]
    out: Dict[str, float] = {"soma_diam_um": soma_diameter(swc), "n_dendrite_nodes": int(len(d))}
    if len(d) < 3:
        out.update({"dend_field_diam_um": 0.0, "dend_length_um": 0.0, "n_branch_points": 0,
                    "strat_depth_um": 0.0, "strat_spread_um": 0.0, "asymmetry_um": 0.0})
        return out
    xy = d[["x", "y"]].to_numpy()
    try:
        area = ConvexHull(xy).volume  # 2-D hull "volume" is the area
    except Exception:  # noqa: BLE001 - degenerate (collinear) arbor
        area = 0.0
    out["dend_field_diam_um"] = float(2 * np.sqrt(area / np.pi))
    coords = swc[["x", "y", "z"]]
    par = d["parent"]
    ok = par.isin(swc.index)
    seg = coords.loc[d.loc[ok, "id"]].to_numpy() - coords.loc[par[ok]].to_numpy()
    out["dend_length_um"] = float(np.linalg.norm(seg, axis=1).sum())
    counts = d["parent"].value_counts()
    out["n_branch_points"] = int(((counts >= 2) & counts.index.isin(d.index)).sum())
    out["strat_depth_um"] = float(d["z"].mean() - c[2])
    out["strat_spread_um"] = float(d["z"].std())
    out["asymmetry_um"] = float(np.linalg.norm(xy.mean(axis=0) - c[:2]))
    return out


def has_axon(swc: pd.DataFrame) -> bool:
    return bool((swc["type"] == AXON).any())


def synthesize_axon(swc: pd.DataFrame, ais_start_um: float = 30.0, ais_len_um: float = 35.0,
                    hillock_diam_um: float = 2.0, ais_diam_um: float = 1.0, axon_diam_um: float = 0.7,
                    axon_len_um: float = 1500.0, direction_xy: Tuple[float, float] = (1.0, 0.0),
                    nfl_z_offset_um: float = -10.0, step_um: float = 5.0, replace: bool = True) -> pd.DataFrame:
    """Append a hillock + AIS + distal axon leaving the soma toward the optic disc.

    The axon runs in the x-y plane at ``soma_z + nfl_z_offset_um`` (nerve-fibre layer is on the
    vitreal side, i.e. negative z in the retinal convention). ``replace=True`` drops any existing
    axon nodes first (so AIS geometry can be varied as a design factor on real reconstructions).
    """
    df = swc[swc["type"] != AXON].copy() if replace else swc.copy()
    df = df[~df["type"].isin((HILLOCK, AIS))]
    c = soma_center(df)
    soma_id = int(df.loc[df["type"] == SOMA, "id"].iloc[0]) if (df["type"] == SOMA).any() else int(df.loc[df["parent"] == -1, "id"].iloc[0])
    r_soma = soma_diameter(df) / 2
    d = np.array(direction_xy, dtype=float)
    d /= np.linalg.norm(d)
    z = c[2] + nfl_z_offset_um
    start = np.array([c[0] + d[0] * r_soma, c[1] + d[1] * r_soma, z])
    total = ais_start_um + ais_len_um + axon_len_um
    n = int(np.ceil(total / step_um))
    next_id = int(df["id"].max()) + 1
    rows = []
    parent = soma_id
    for k in range(1, n + 1):
        s = min(k * step_um, total)
        pos = start + np.array([d[0] * s, d[1] * s, 0.0])
        if s <= ais_start_um:
            t, rad = HILLOCK, hillock_diam_um / 2
        elif s <= ais_start_um + ais_len_um:
            t, rad = AIS, ais_diam_um / 2
        else:
            t, rad = AXON, axon_diam_um / 2
        rows.append([next_id, t, pos[0], pos[1], pos[2], rad, parent])
        parent = next_id
        next_id += 1
    add = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"]).set_index("id", drop=False)
    return pd.concat([df, add])


@dataclass
class Compartments:
    """Cylindrical compartments of a cell: centre (um), length (um), diameter (um), region, parent (-1 root)."""

    xyz: np.ndarray
    length: np.ndarray
    diam: np.ndarray
    region: np.ndarray
    parent: np.ndarray
    swc_id: np.ndarray

    @property
    def n(self) -> int:
        return len(self.length)

    @property
    def area_cm2(self) -> np.ndarray:
        return np.pi * (self.diam * 1e-4) * (self.length * 1e-4)


def compartmentalize(swc: pd.DataFrame, max_len_um: float = 10.0) -> Compartments:
    """Split every SWC edge into cylinders of at most ``max_len_um``; the soma becomes one
    equivalent cylinder (L = d = soma diameter, so area = pi d^2, the sphere's surface)."""
    root_ids = list(swc.loc[swc["parent"] == -1, "id"])
    root = root_ids[0]
    d_soma = soma_diameter(swc)
    c = soma_center(swc)
    xyz: List[np.ndarray] = [c]
    length, diam, region, parent, sid = [d_soma], [d_soma], [SOMA], [-1], [root]
    last_comp: Dict[int, int] = {}
    soma_ids = set(swc.loc[swc["type"] == SOMA, "id"]) | set(root_ids)
    for s in soma_ids:
        last_comp[s] = 0
    coords = swc[["x", "y", "z"]].to_numpy()
    idx = {i: k for k, i in enumerate(swc["id"])}
    order = list(swc["id"])  # SWC guarantees parents precede children in well-formed files
    for nid in order:
        if nid in soma_ids:
            continue
        pid = int(swc.at[nid, "parent"])
        if pid not in last_comp:
            continue  # orphan or parent outside tree
        p0 = coords[idx[pid]]
        p1 = coords[idx[nid]]
        if pid in soma_ids:
            # start at the soma surface along the edge direction
            v = p1 - c
            nv = np.linalg.norm(v)
            p0 = c + v / nv * min(d_soma / 2, nv * 0.5) if nv > 0 else c
        L = float(np.linalg.norm(p1 - p0))
        rad = float(swc.at[nid, "r"])
        t = int(swc.at[nid, "type"])
        t = t if t in (AXON, BASAL, APICAL, HILLOCK, AIS) else BASAL
        n_sub = max(1, int(np.ceil(L / max_len_um)))
        prev = last_comp[pid]
        for k in range(n_sub):
            a = p0 + (p1 - p0) * (k / n_sub)
            b = p0 + (p1 - p0) * ((k + 1) / n_sub)
            xyz.append(0.5 * (a + b))
            length.append(max(L / n_sub, 0.1))
            diam.append(max(2 * rad, 0.2))
            region.append(t)
            parent.append(prev)
            sid.append(nid)
            prev = len(length) - 1
        last_comp[nid] = prev
    return Compartments(np.asarray(xyz, dtype=float), np.asarray(length, dtype=float), np.asarray(diam, dtype=float),
                        np.asarray(region, dtype=int), np.asarray(parent, dtype=int), np.asarray(sid, dtype=int))


def synthetic_rgc(n_dendrites: int = 4, dend_len_um: float = 120.0, soma_diam_um: float = 15.0,
                  strat_z_um: float = 25.0, seed: int = 0, with_axon: bool = True, **axon_kw) -> pd.DataFrame:
    """A small synthetic RGC (soma + radial bifurcating dendrites at IPL depth) for tests and demos."""
    rng = np.random.default_rng(seed)
    rows = [[1, SOMA, 0.0, 0.0, 0.0, soma_diam_um / 2, -1]]
    nid = 2
    for k in range(n_dendrites):
        ang = 2 * np.pi * k / n_dendrites + rng.normal(0, 0.1)
        d = np.array([np.cos(ang), np.sin(ang)])
        # primary dendrite rising to the stratification depth
        p1 = np.array([d[0] * dend_len_um * 0.4, d[1] * dend_len_um * 0.4, strat_z_um])
        rows.append([nid, BASAL, p1[0], p1[1], p1[2], 0.8, 1])
        pid = nid
        nid += 1
        for sgn in (-1, 1):
            perp = np.array([-d[1], d[0]]) * sgn
            p2 = np.array([p1[0] + (d[0] * 0.6 + perp[0] * 0.5) * dend_len_um * 0.6,
                           p1[1] + (d[1] * 0.6 + perp[1] * 0.5) * dend_len_um * 0.6, strat_z_um + rng.normal(0, 2)])
            rows.append([nid, BASAL, p2[0], p2[1], p2[2], 0.5, pid])
            nid += 1
    swc = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"]).set_index("id", drop=False)
    return synthesize_axon(swc, **axon_kw) if with_axon else swc
