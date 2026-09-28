"""Spatial autocorrelation and surrogate maps for regional (3-D) brain data.

Spin tests need a sphere; for regions in a volume we use Moran spectral randomisation
(MSR; Wagner & Dray, 2015, Methods Ecol Evol): surrogates are built from the Moran
eigenvector basis of a spatial weight matrix so that they preserve the observed
Moran's I (and the map's spectral profile) while destroying region-specific values.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def distance_weights(coords: np.ndarray, kind: str = "inverse", k: Optional[int] = None, power: float = 1.0
                     ) -> np.ndarray:
    """Symmetric spatial weight matrix from region centroids (inverse distance, optionally kNN-sparsified)."""
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        W = 1.0 / np.power(D, power)
    np.fill_diagonal(W, 0.0)
    W[~np.isfinite(W)] = 0.0
    if k is not None:
        n = len(W)
        keep = np.zeros_like(W, dtype=bool)
        order = np.argsort(D, axis=1)[:, 1:k + 1]
        for i in range(n):
            keep[i, order[i]] = True
        keep = keep | keep.T
        W = W * keep
    return W


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    x = np.asarray(x, float)
    z = x - x.mean()
    S0 = W.sum()
    return float(len(x) / S0 * (z @ W @ z) / (z @ z)) if S0 > 0 and (z @ z) > 0 else float("nan")


def moran_eigenvectors(W: np.ndarray) -> np.ndarray:
    """Eigenvectors of the doubly-centred weight matrix (Moran's eigenvector maps), sorted by eigenvalue desc."""
    n = len(W)
    H = np.eye(n) - np.ones((n, n)) / n
    M = H @ ((W + W.T) / 2) @ H
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    return V[:, order]


def msr_surrogates(x: np.ndarray, W: np.ndarray, n_surr: int = 1000, seed: int = 0,
                   eigvecs: Optional[np.ndarray] = None) -> np.ndarray:
    """Moran spectral randomisation (singleton procedure): (n_surr, n) surrogates matching mean, var and Moran's I.

    x is projected on the Moran eigenvector basis; the signs of the loadings are randomised
    and the loadings are re-scaled, following the "singleton" MSR variant.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    V = moran_eigenvectors(W) if eigvecs is None else eigvecs
    z = (x - x.mean()) / x.std()
    a = V.T @ z                       # loadings on the eigenbasis
    out = np.empty((n_surr, n))
    for s in range(n_surr):
        signs = rng.choice([-1.0, 1.0], size=n)
        # random rotation within the loadings' magnitude profile keeps the Moran spectrum
        b = signs * a
        y = V @ b
        y = (y - y.mean()) / y.std()
        out[s] = y * x.std() + x.mean()
    return out


def spatial_pvalue(stat_fn: Callable[[np.ndarray], float], x: np.ndarray, W: np.ndarray, n_surr: int = 1000,
                   seed: int = 0, two_sided: bool = True) -> Dict[str, float]:
    """Permutation p-value of ``stat_fn(x)`` against MSR surrogates of the target map ``x``."""
    obs = float(stat_fn(x))
    S = msr_surrogates(x, W, n_surr, seed)
    null = np.array([stat_fn(s) for s in S])
    null = null[np.isfinite(null)]
    if two_sided:
        p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
    else:
        p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return {"observed": obs, "p_spatial": p, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "moran_i_obs": morans_i(x, W), "moran_i_null_mean": float(np.mean([morans_i(s, W) for s in S[:100]]))}


def naive_pvalue(stat_fn: Callable[[np.ndarray], float], x: np.ndarray, n_perm: int = 1000, seed: int = 0) -> float:
    """Plain permutation p-value (for showing inflation relative to the spatial null)."""
    rng = np.random.default_rng(seed)
    obs = abs(float(stat_fn(x)))
    null = np.array([abs(stat_fn(rng.permutation(x))) for _ in range(n_perm)])
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def variogram(x: np.ndarray, coords: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Empirical semivariogram of a regional map (for checking that surrogates match spatial structure)."""
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu = np.triu_indices(len(x), 1)
    d = D[iu]
    g = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
    return {"lag": np.array([d[which == b].mean() for b in range(n_bins)]),
            "gamma": np.array([g[which == b].mean() for b in range(n_bins)])}
