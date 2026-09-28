"""SWC parsing shared by the feature extractors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

SOMA, AXON, BASAL, APICAL = 1, 2, 3, 4
DENDRITES = (BASAL, APICAL)


@dataclass
class Neuron:
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray  # row index, -1 for roots

    @property
    def n(self) -> int:
        return int(len(self.types))

    def children(self) -> List[List[int]]:
        ch: List[List[int]] = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(i)
        return ch

    def soma_center(self) -> np.ndarray:
        s = self.types == SOMA
        return self.xyz[s].mean(axis=0) if s.any() else self.xyz[self.parent < 0].mean(axis=0)

    def seg_length(self) -> np.ndarray:
        out = np.zeros(self.n)
        m = self.parent >= 0
        out[m] = np.linalg.norm(self.xyz[m] - self.xyz[self.parent[m]], axis=1)
        return out


def from_array(arr: np.ndarray) -> Neuron:
    ids = arr[:, 0].astype(int)
    idx = {int(i): k for k, i in enumerate(ids)}
    parent = np.array([idx.get(int(p), -1) if p >= 0 else -1 for p in arr[:, 6]], dtype=int)
    return Neuron(types=arr[:, 1].astype(int), xyz=arr[:, 2:5].astype(float),
                  radius=arr[:, 5].astype(float), parent=parent)


def read_swc(path: Path | str) -> Neuron:
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                p = s.split()
                if len(p) >= 7:
                    rows.append([float(v) for v in p[:7]])
    if not rows:
        raise ValueError(f"empty SWC: {path}")
    return from_array(np.asarray(rows, float))


def synthetic_interneuron(rng: np.random.Generator, n_stems: int = 5, depth_scale: float = 1.0,
                          size_scale: float = 1.0, axon: bool = True, branch_p: float = 0.35) -> Neuron:
    """Random tree used in tests: ``size_scale`` stretches lengths, ``depth_scale`` biases growth along y.

    Rows: [id, type, x, y, z, r, parent]. y is the depth axis (positive = deeper).
    """
    rows = [[1, SOMA, 0.0, 0.0, 0.0, 6.0, -1]]
    next_id = 2

    def grow(parent_id: int, ntype: int, pos: np.ndarray, direction: np.ndarray, depth: int) -> None:
        nonlocal next_id
        n_pts = rng.integers(4, 9)
        p = pos.copy()
        pid = parent_id
        for _ in range(n_pts):
            direction = direction + rng.normal(0, 0.35, 3)
            direction[1] *= depth_scale
            direction /= np.linalg.norm(direction) + 1e-9
            p = p + direction * 12.0 * size_scale
            rows.append([next_id, ntype, *p, 0.5, pid])
            pid = next_id
            next_id += 1
        if depth < 4 and rng.random() < branch_p + 0.3 * (depth == 0):
            for k in range(2):
                d2 = direction + rng.normal(0, 0.8, 3)
                grow(pid, ntype, p, d2, depth + 1)

    for _ in range(n_stems):
        d = rng.normal(0, 1, 3)
        d /= np.linalg.norm(d)
        grow(1, BASAL, np.zeros(3), d, 0)
    if axon:
        d = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.3, 3)
        grow(1, AXON, np.zeros(3), d / np.linalg.norm(d), 0)
    return from_array(np.asarray(rows, float))
