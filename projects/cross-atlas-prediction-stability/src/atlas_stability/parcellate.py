"""Parcellation utilities that work on plain arrays.

A parcellation is an integer label per vertex/voxel (0 = unassigned). The same operations are
run in production with ``wb_command -cifti-parcellate`` (surface) or nilearn's
``NiftiLabelsMasker`` (volume); the array versions here make the pipeline testable and are used
to project node- and edge-level findings back into the common vertex space.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def parcellate_timeseries(ts: np.ndarray, labels: np.ndarray, n_parcels: Optional[int] = None) -> np.ndarray:
    """Mean time series per label. ``ts`` is ``(T, V)``; ``labels`` ``(V,)`` with 0 = ignore.

    Returns ``(T, n_parcels)`` with columns ordered by label 1..n_parcels; empty parcels are NaN.
    """
    ts = np.asarray(ts, float)
    labels = np.asarray(labels, int)
    if ts.shape[1] != len(labels):
        raise ValueError("labels must have one entry per column of ts")
    n = int(labels.max()) if n_parcels is None else n_parcels
    counts = np.bincount(labels, minlength=n + 1)[1:]
    sums = np.zeros((ts.shape[0], n))
    for j in range(n):
        idx = labels == j + 1
        if idx.any():
            sums[:, j] = ts[:, idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    out[:, counts == 0] = np.nan
    return out


def fisher_z_fc(ts: np.ndarray) -> np.ndarray:
    """Fisher-z Pearson FC of ``(T, N)``; diagonal zero; NaN columns give NaN rows/columns."""
    X = np.asarray(ts, float)
    X = X - np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = np.nan
    Z = (X / sd)
    R = np.nan_to_num(Z).T @ np.nan_to_num(Z) / X.shape[0]
    bad = ~np.isfinite(sd)
    R[bad, :] = np.nan
    R[:, bad] = np.nan
    R = np.clip(R, -0.999999, 0.999999)
    F = np.arctanh(R)
    np.fill_diagonal(F, 0.0)
    return F


def vectorize_upper(fc: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(fc.shape[0], k=1)
    return np.asarray(fc)[iu]


def edge_weights_to_node_strength(edge_weights: np.ndarray, n_nodes: int, signed: bool = False) -> np.ndarray:
    """Sum of (absolute) edge weights incident to each node, from an upper-triangle vector."""
    iu = np.triu_indices(n_nodes, k=1)
    w = np.asarray(edge_weights, float) if signed else np.abs(edge_weights)
    strength = np.zeros(n_nodes)
    np.add.at(strength, iu[0], w)
    np.add.at(strength, iu[1], w)
    return strength


def node_to_space(node_values: np.ndarray, labels: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Broadcast a per-node quantity to every vertex/voxel of its parcel (common-space map)."""
    labels = np.asarray(labels, int)
    out = np.full(len(labels), fill, float)
    vals = np.asarray(node_values, float)
    assigned = labels > 0
    out[assigned] = vals[labels[assigned] - 1]
    return out


def majority_network_assignment(labels: np.ndarray, network_labels: np.ndarray, n_parcels: Optional[int] = None) -> np.ndarray:
    """Assign each parcel to the network (e.g. Yeo-7) covering most of its vertices; 0 if none."""
    labels = np.asarray(labels, int)
    net = np.asarray(network_labels, int)
    n = int(labels.max()) if n_parcels is None else n_parcels
    out = np.zeros(n, int)
    for j in range(n):
        idx = labels == j + 1
        if idx.any():
            counts = np.bincount(net[idx])
            counts[0] = 0  # ignore unassigned
            out[j] = int(np.argmax(counts)) if counts.sum() > 0 else 0
    return out


def aggregate_nodes_to_networks(node_values: np.ndarray, assignment: np.ndarray, n_networks: int) -> np.ndarray:
    """Mean node value per network (NaN for empty networks)."""
    vals = np.asarray(node_values, float)
    out = np.full(n_networks, np.nan)
    for k in range(1, n_networks + 1):
        idx = assignment == k
        if idx.any():
            out[k - 1] = np.nanmean(vals[idx])
    return out


def random_contiguous_parcellation(n_vertices: int, n_parcels: int, rng: Optional[np.random.Generator] = None, min_size: int = 1) -> np.ndarray:
    """Random contiguous parcellation of a 1-D 'cortex' (labels 1..n_parcels); a null/simulation tool."""
    rng = np.random.default_rng() if rng is None else rng
    if n_parcels * min_size > n_vertices:
        raise ValueError("too many parcels for the number of vertices")
    cuts = np.sort(rng.choice(np.arange(min_size, n_vertices - min_size + 1), size=n_parcels - 1, replace=False)) if n_parcels > 1 else np.array([], int)
    labels = np.zeros(n_vertices, int)
    start = 0
    for j, c in enumerate(list(cuts) + [n_vertices]):
        labels[start:c] = j + 1
        start = c
    return labels
