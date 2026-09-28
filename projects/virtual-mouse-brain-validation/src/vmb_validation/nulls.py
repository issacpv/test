"""Null connectomes: degree-preserving rewiring, distance-preserving weight permutation, weight shuffle."""
from __future__ import annotations

from typing import Optional

import numpy as np


def shuffle_weights(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the weights of existing edges among the existing edge positions (topology kept)."""
    out = W.copy()
    idx = np.argwhere(W > 0)
    vals = W[W > 0]
    out[W > 0] = rng.permutation(vals)
    del idx
    return out


def rewire_degree_preserving(W: np.ndarray, rng: np.random.Generator, n_swaps: Optional[int] = None,
                             directed: bool = True) -> np.ndarray:
    """Maslov-Sneppen edge swaps preserving in/out degree (directed) or degree (undirected).

    Weights travel with their edges. Self-loops and duplicate edges are rejected.
    """
    out = W.copy()
    n = len(W)
    edges = [tuple(e) for e in np.argwhere(out > 0) if not directed and e[0] < e[1] or directed]
    if len(edges) < 2:
        return out
    n_swaps = n_swaps or 10 * len(edges)
    attempts = 0
    swaps = 0
    while swaps < n_swaps and attempts < 50 * n_swaps:
        attempts += 1
        (a, b), (c, d) = [edges[k] for k in rng.integers(0, len(edges), 2)]
        if len({a, b, c, d}) < 4:
            continue
        if out[a, d] > 0 or out[c, b] > 0:
            continue
        w1, w2 = out[a, b], out[c, d]
        out[a, b] = out[c, d] = 0.0
        out[a, d], out[c, b] = w1, w2
        if not directed:
            out[b, a] = out[d, c] = 0.0
            out[d, a], out[b, c] = w1, w2
        i1, i2 = edges.index((a, b)), edges.index((c, d))
        edges[i1], edges[i2] = (a, d), (c, b)
        swaps += 1
    return out


def distance_preserving_permutation(W: np.ndarray, coords: np.ndarray, rng: np.random.Generator,
                                    n_bins: int = 10) -> np.ndarray:
    """Permute edge weights only among edges with similar Euclidean length (geometric null).

    Preserves the distance-dependence of weights and the edge count per distance bin,
    destroys region-specific wiring. Zero entries stay zero.
    """
    n = len(W)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    out = W.copy()
    mask = (W > 0) & ~np.eye(n, dtype=bool)
    d_edges = D[mask]
    edges = np.argwhere(mask)
    if len(edges) < 2:
        return out
    bins = np.quantile(d_edges, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(bins, d_edges, side="right") - 1, 0, n_bins - 1)
    vals = W[mask]
    new = vals.copy()
    for b in range(n_bins):
        sel = np.where(which == b)[0]
        new[sel] = vals[rng.permutation(sel)]
    out[mask] = new
    return out


def degree_sequences(W: np.ndarray):
    return (W > 0).sum(axis=1), (W > 0).sum(axis=0)
