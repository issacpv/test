"""Vertex-level simulator with a ground-truth parcellation.

The 'cortex' is a 1-D array of ``n_vertices`` split into ``n_regions`` contiguous true regions.
Each region has a latent time series; vertices carry their region's series plus noise. The
coupling between regions ``a`` and ``b`` depends on the subject's phenotype, so the FC edge
(a, b) carries the signal. Atlases that respect the a/b boundary recover the phenotype;
atlases that merge a or b with other regions dilute it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VertexDataset:
    ts: np.ndarray             # (n_subjects, T, V)
    y: np.ndarray              # (n_subjects,)
    true_labels: np.ndarray    # (V,) ground-truth parcellation
    signal_regions: Tuple[int, int]
    families: np.ndarray       # (n_subjects,)


def simulate_vertex_dataset(
    n_subjects: int = 120,
    n_vertices: int = 200,
    n_regions: int = 10,
    n_timepoints: int = 150,
    coupling_base: float = 0.2,
    coupling_effect: float = 0.5,
    vertex_noise: float = 0.7,
    seed: int = 0,
) -> VertexDataset:
    rng = np.random.default_rng(seed)
    sizes = np.full(n_regions, n_vertices // n_regions)
    sizes[: n_vertices % n_regions] += 1
    true_labels = np.repeat(np.arange(1, n_regions + 1), sizes)
    a, b = 0, 1
    y = rng.normal(size=n_subjects)
    ts = np.empty((n_subjects, n_timepoints, n_vertices))
    for i in range(n_subjects):
        latent = rng.normal(size=(n_timepoints, n_regions))
        rho = np.clip(coupling_base + coupling_effect * y[i], -0.95, 0.95)
        latent[:, b] = rho * latent[:, a] + np.sqrt(1 - rho ** 2) * latent[:, b]
        ts[i] = latent[:, true_labels - 1] + vertex_noise * rng.normal(size=(n_timepoints, n_vertices))
    return VertexDataset(ts, y, true_labels, (a + 1, b + 1), np.arange(n_subjects) // 2)
