"""Microstate segmentation and temporal parameters for multichannel M/EEG.

Data convention: ``data`` is an array of shape ``(n_channels, n_times)``, already filtered and
(for EEG) average-referenced. For MEG magnetometers the "polarity" of a topography is
physically meaningful, so ``polarity_invariant`` is an explicit option everywhere (EEG
microstate practice ignores polarity; MEG studies differ).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- GFP
def gfp(data: np.ndarray) -> np.ndarray:
    """Global field power: spatial standard deviation at each time point."""
    data = np.asarray(data, dtype=float)
    return data.std(axis=0)


def gfp_peaks(g: np.ndarray, min_distance: int = 1) -> np.ndarray:
    """Indices of local maxima of the GFP curve separated by at least ``min_distance`` samples."""
    g = np.asarray(g, dtype=float)
    if len(g) < 3:
        return np.zeros(0, dtype=int)
    cand = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:])) + 1
    if min_distance <= 1 or len(cand) == 0:
        return cand
    keep = [cand[0]]
    for c in cand[1:]:
        if c - keep[-1] >= min_distance:
            keep.append(c)
        elif g[c] > g[keep[-1]]:
            keep[-1] = c
    return np.asarray(keep, dtype=int)


# --------------------------------------------------------------------------- clustering
def _normalise_maps(maps: np.ndarray) -> np.ndarray:
    maps = maps - maps.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(maps, axis=1, keepdims=True)
    return maps / np.maximum(norms, 1e-12)


def _activation(maps: np.ndarray, templates: np.ndarray, polarity_invariant: bool) -> np.ndarray:
    """Spatial correlation between each map (rows) and each template (rows)."""
    a = _normalise_maps(maps) @ _normalise_maps(templates).T
    return np.abs(a) if polarity_invariant else a


def modified_kmeans(maps: np.ndarray, n_states: int = 4, n_inits: int = 10, max_iter: int = 300, tol: float = 1e-6,
                    polarity_invariant: bool = True, seed: int = 0) -> Dict[str, np.ndarray]:
    """Modified k-means (Pascual-Marqui, Michel & Lehmann, 1995) on GFP-peak maps.

    ``maps`` has shape ``(n_samples, n_channels)``. Returns ``templates (n_states, n_channels)``
    (unit norm), ``labels`` and ``gev``. With ``polarity_invariant`` the template update uses the
    first principal component of assigned maps (sign-free); otherwise the (normalised) mean map.
    """
    rng = np.random.default_rng(seed)
    X = _normalise_maps(np.asarray(maps, dtype=float))
    n, c = X.shape
    if n < n_states:
        raise ValueError("need at least n_states maps")
    best: Optional[Dict[str, np.ndarray]] = None
    best_gev = -np.inf
    for _ in range(n_inits):
        T = X[rng.choice(n, size=n_states, replace=False)].copy()
        prev = -np.inf
        for _it in range(max_iter):
            act = _activation(X, T, polarity_invariant)
            labels = np.argmax(act, axis=1)
            for k in range(n_states):
                members = X[labels == k]
                if len(members) == 0:
                    T[k] = X[rng.integers(n)]
                    continue
                if polarity_invariant:
                    # first eigenvector of the members' covariance (sign-free template)
                    _, vecs = np.linalg.eigh(members.T @ members)
                    T[k] = vecs[:, -1]
                else:
                    T[k] = members.mean(axis=0)
            T = _normalise_maps(T)
            fit = float(np.sum(np.max(_activation(X, T, polarity_invariant), axis=1) ** 2))
            if fit - prev < tol:
                break
            prev = fit
        act = _activation(X, T, polarity_invariant)
        labels = np.argmax(act, axis=1)
        g = float(np.mean(np.max(act, axis=1) ** 2))  # GEV with unit-norm maps = mean squared correlation
        if g > best_gev:
            best_gev = g
            best = {"templates": T.copy(), "labels": labels.copy(), "gev": np.array(g)}
    assert best is not None
    return best


def backfit(data: np.ndarray, templates: np.ndarray, polarity_invariant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Assign every time point to the template with the highest spatial correlation.

    Returns ``(labels (n_times,), correlation (n_times,))``.
    """
    act = _activation(np.asarray(data, dtype=float).T, templates, polarity_invariant)
    labels = np.argmax(act, axis=1)
    return labels, act[np.arange(len(labels)), labels]


def smooth_labels(labels: np.ndarray, min_samples: int) -> np.ndarray:
    """Reassign segments shorter than ``min_samples`` to the neighbouring segment with the longer run.

    Iterates until no short segment remains (or the sequence is a single segment).
    """
    labels = np.asarray(labels).copy()
    if min_samples <= 1 or len(labels) == 0:
        return labels
    changed = True
    while changed:
        changed = False
        bounds = np.flatnonzero(np.diff(labels)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(labels)]])
        lengths = ends - starts
        if len(starts) == 1:
            break
        for i in np.argsort(lengths):
            if lengths[i] >= min_samples:
                break
            left_len = lengths[i - 1] if i > 0 else -1
            right_len = lengths[i + 1] if i < len(starts) - 1 else -1
            if left_len < 0 and right_len < 0:
                continue
            src = starts[i - 1] if left_len >= right_len else starts[i + 1]
            labels[starts[i]:ends[i]] = labels[src]
            changed = True
            break
    return labels


def global_explained_variance(data: np.ndarray, templates: np.ndarray, labels: np.ndarray,
                              polarity_invariant: bool = True) -> float:
    """GEV = sum_t gfp_t^2 corr_t^2 / sum_t gfp_t^2 over all time points."""
    data = np.asarray(data, dtype=float)
    g = gfp(data)
    act = _activation(data.T, templates, polarity_invariant)
    corr = act[np.arange(len(labels)), labels]
    return float(np.sum(g ** 2 * corr ** 2) / np.sum(g ** 2))


# --------------------------------------------------------------------------- parameters
def microstate_parameters(labels: np.ndarray, sfreq: float, n_states: int) -> Dict[str, np.ndarray]:
    """Temporal parameters of a label sequence.

    Returns mean duration (s) per state, occurrence (segments per second) per state, coverage
    (fraction of time) per state, the row-normalised transition matrix between *different*
    states (diagonal zero), the entropy rate of that Markov chain (bits) and the DFA exponent.
    """
    labels = np.asarray(labels)
    n = len(labels)
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    seg_labels = labels[starts]
    seg_len = (ends - starts) / sfreq
    duration = np.array([seg_len[seg_labels == k].mean() if np.any(seg_labels == k) else np.nan for k in range(n_states)])
    occurrence = np.array([np.sum(seg_labels == k) for k in range(n_states)]) / (n / sfreq)
    coverage = np.array([np.mean(labels == k) for k in range(n_states)])
    trans = np.zeros((n_states, n_states))
    for a, b in zip(seg_labels[:-1], seg_labels[1:]):
        trans[a, b] += 1
    row = trans.sum(axis=1, keepdims=True)
    trans_p = np.divide(trans, row, out=np.zeros_like(trans), where=row > 0)
    stationary = row[:, 0] / max(row.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_rows = -np.nansum(np.where(trans_p > 0, trans_p * np.log2(trans_p), 0.0), axis=1)
    entropy_rate = float(np.sum(stationary * h_rows))
    return {"duration": duration, "occurrence": occurrence, "coverage": coverage, "transition": trans_p,
            "transition_counts": trans, "entropy_rate": np.array(entropy_rate),
            "dfa": np.array(dfa_exponent(labels_to_random_walk(labels, n_states)))}


def labels_to_random_walk(labels: np.ndarray, n_states: int, n_partitions: int = 8, seed: int = 0) -> np.ndarray:
    """Embed a categorical sequence as a random walk (Van de Ville et al., 2010): average over random
    +/-1 assignments of states of the cumulative sum of the resulting +/-1 series."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    walks = []
    for _ in range(n_partitions):
        signs = rng.choice([-1.0, 1.0], size=n_states)
        if np.all(signs == signs[0]):
            signs[0] *= -1
        walks.append(np.cumsum(signs[labels]))
    return np.mean(walks, axis=0)


def dfa_exponent(x: np.ndarray, min_win: int = 16, max_win: Optional[int] = None, n_scales: int = 12) -> float:
    """Detrended fluctuation analysis exponent (linear detrending) of a series ``x``.

    The input is treated as the profile if it is already a cumulative sum; here we always
    integrate the mean-removed series, as in the standard DFA-1 definition.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 * min_win:
        return float("nan")
    y = np.cumsum(x - x.mean())
    max_win = max_win or n // 4
    scales = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int))
    flucts = []
    for s in scales:
        n_seg = n // s
        segs = y[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        coef, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)
        resid = segs - (A @ coef).T
        flucts.append(np.sqrt(np.mean(resid ** 2)))
    slope = np.polyfit(np.log(scales), np.log(np.asarray(flucts) + 1e-12), 1)[0]
    return float(slope)


def segment(data: np.ndarray, sfreq: float, n_states: int = 4, min_duration_ms: float = 20.0, peak_min_distance_ms: float = 8.0,
            polarity_invariant: bool = True, seed: int = 0, templates: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Full single-recording pipeline: GFP peaks -> (cluster) -> back-fit -> smooth -> parameters."""
    data = np.asarray(data, dtype=float)
    g = gfp(data)
    peaks = gfp_peaks(g, min_distance=max(1, int(round(peak_min_distance_ms / 1000 * sfreq))))
    if templates is None:
        km = modified_kmeans(data[:, peaks].T, n_states=n_states, polarity_invariant=polarity_invariant, seed=seed)
        templates = km["templates"]
    labels, corr = backfit(data, templates, polarity_invariant)
    labels = smooth_labels(labels, int(round(min_duration_ms / 1000 * sfreq)))
    params = microstate_parameters(labels, sfreq, templates.shape[0])
    return {"templates": templates, "labels": labels, "corr": corr, "peaks": peaks, "params": params,
            "gev": global_explained_variance(data, templates, labels, polarity_invariant)}


def match_templates(a: np.ndarray, b: np.ndarray, polarity_invariant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of template sets by spatial correlation; returns (perm, corrs)."""
    act = _activation(a, b, polarity_invariant)
    perm = -np.ones(a.shape[0], dtype=int)
    corrs = np.zeros(a.shape[0])
    used = set()
    for _ in range(a.shape[0]):
        masked = act.copy()
        masked[perm >= 0, :] = -np.inf
        masked[:, list(used)] = -np.inf
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        perm[i], corrs[i] = j, act[i, j]
        used.add(j)
    return perm, corrs


__all__ = ["gfp", "gfp_peaks", "modified_kmeans", "backfit", "smooth_labels", "global_explained_variance",
           "microstate_parameters", "labels_to_random_walk", "dfa_exponent", "segment", "match_templates"]
