"""Minimal SWC parsing for compartmental modelling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4


@dataclass
class Tree:
    """SWC arrays; ``parent`` holds row indices (-1 for roots)."""

    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.types))

    def children(self) -> List[List[int]]:
        ch: List[List[int]] = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(i)
        return ch

    def seg_length(self) -> np.ndarray:
        out = np.zeros(self.n)
        m = self.parent >= 0
        out[m] = np.linalg.norm(self.xyz[m] - self.xyz[self.parent[m]], axis=1)
        return out

    def soma_radius(self) -> float:
        """Equivalent-sphere radius: max soma-node radius, or sphere with the same area as the soma contour."""
        s = self.types == SOMA
        if not s.any():
            return float(self.radius[self.parent < 0].max())
        return float(self.radius[s].max())


def from_array(arr: np.ndarray) -> Tree:
    ids = arr[:, 0].astype(int)
    idx = {int(i): k for k, i in enumerate(ids)}
    parent = np.array([idx.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=int)
    return Tree(types=arr[:, 1].astype(int), xyz=arr[:, 2:5].astype(float),
                radius=arr[:, 5].astype(float), parent=parent)


def read_swc(path: Path | str) -> Tree:
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                parts = s.split()
                if len(parts) >= 7:
                    rows.append([float(v) for v in parts[:7]])
    if not rows:
        raise ValueError(f"empty SWC: {path}")
    return from_array(np.asarray(rows, float))


def ball_and_stick(r_soma: float = 10.0, diam: float = 2.0, length: float = 400.0, n_seg: int = 40,
                   dend_type: int = BASAL) -> Tree:
    """Synthetic soma + single dendrite, used for analytic checks and as a load-matched control."""
    # soma at the origin; nodes every `step` so every segment (soma->first node included)
    # has length `step` and the cylinder length is exactly `length`
    rows = [[1, SOMA, 0.0, 0.0, 0.0, r_soma, -1]]
    step = length / n_seg
    for k in range(1, n_seg + 1):
        rows.append([k + 1, dend_type, k * step, 0.0, 0.0, diam / 2, k])
    return from_array(np.asarray(rows, float))
