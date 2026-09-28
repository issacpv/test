"""Synthetic neuron trees with controllable tracing artefacts.

``make_tree`` grows a random binary tree in 3-D and then applies a "software profile": node
spacing distribution, coordinate precision, radius rounding, z-slice quantisation and
z-compression. Two profiles with identical biology but different artefacts are what a
provenance classifier should separate; identical profiles should be inseparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .swc_features import SOMA, BASAL


@dataclass
class SoftwareProfile:
    """Tracing artefacts characteristic of a reconstruction pipeline."""

    spacing_mean: float = 2.0     # mean inter-node distance (um)
    spacing_cv: float = 0.5       # coefficient of variation of spacing (0 = perfectly regular)
    precision: Optional[int] = 3  # decimals kept for x, y, z (None = full precision)
    radius_quantum: float = 0.0   # radii rounded to multiples of this (0 = none)
    z_quantum: float = 0.0        # z rounded to multiples of this (slice step; 0 = none)
    z_compression: float = 1.0    # multiplicative z shrinkage (1 = none)
    radius_constant: bool = False # a single radius for all neurite nodes


def make_tree(profile: Optional[SoftwareProfile] = None, n_branch_events: int = 25, branch_length_mean: float = 60.0,
              tortuosity: float = 0.15, seed: int = 0, taper: float = 0.85) -> Dict[str, np.ndarray]:
    """Grow a random dendritic tree and sample it according to ``profile``.

    Biology (branch count, branch lengths, tortuosity) is controlled independently of the
    sampling profile so that tests can hold one fixed while varying the other.
    """
    profile = profile or SoftwareProfile()
    rng = np.random.default_rng(seed)
    xyz: List[np.ndarray] = [np.zeros(3)]
    typ: List[int] = [SOMA]
    rad: List[float] = [6.0]
    par: List[int] = [-1]

    # queue of (parent index, direction, radius, depth)
    queue = []
    n_stems = int(rng.integers(3, 6))
    for _ in range(n_stems):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        queue.append((0, d, 1.2, 0))
    events = 0
    while queue:
        p, direction, r, depth = queue.pop(0)
        length = max(5.0, rng.gamma(shape=4.0, scale=branch_length_mean / 4.0))
        # dense underlying polyline with small random turns (the "true" geometry)
        n_dense = max(2, int(length / 0.25))
        pts = [xyz[p]]
        d = direction.copy()
        for _ in range(n_dense):
            d = d + tortuosity * 0.25 * rng.normal(size=3)
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + 0.25 * d)
        pts = np.asarray(pts)
        # resample according to the software profile
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        pos = 0.0
        last = p
        radius_here = r
        while True:
            step = max(0.2, rng.normal(profile.spacing_mean, profile.spacing_cv * profile.spacing_mean)) if profile.spacing_cv > 0 else profile.spacing_mean
            pos += step
            if pos >= cum[-1]:
                break
            j = int(np.searchsorted(cum, pos) - 1)
            frac = (pos - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
            point = pts[j] + frac * (pts[j + 1] - pts[j])
            xyz.append(point)
            typ.append(BASAL)
            rad.append(radius_here)
            par.append(last)
            last = len(xyz) - 1
            radius_here = max(0.15, radius_here * (1 - 0.002 * step))
        # end node (bifurcation or tip)
        xyz.append(pts[-1])
        typ.append(BASAL)
        rad.append(radius_here)
        par.append(last)
        end = len(xyz) - 1
        if events < n_branch_events and depth < 7 and rng.random() < 0.75:
            events += 1
            for _ in range(2):
                nd = d + 0.8 * rng.normal(size=3)
                nd /= np.linalg.norm(nd)
                queue.append((end, nd, radius_here * taper, depth + 1))
    arr = np.asarray(xyz)
    radius = np.asarray(rad)
    types = np.asarray(typ)
    neur = types != SOMA
    arr[:, 2] *= profile.z_compression
    if profile.z_quantum > 0:
        arr[:, 2] = np.round(arr[:, 2] / profile.z_quantum) * profile.z_quantum
    if profile.precision is not None:
        arr = np.round(arr, profile.precision)
    if profile.radius_constant:
        radius[neur] = 0.5
    if profile.radius_quantum > 0:
        radius[neur] = np.maximum(profile.radius_quantum, np.round(radius[neur] / profile.radius_quantum) * profile.radius_quantum)
    return {"id": np.arange(len(arr)), "type": types, "xyz": arr, "radius": radius, "parent": np.asarray(par),
            "raw_lines": np.asarray([], dtype=object)}


__all__ = ["SoftwareProfile", "make_tree"]
