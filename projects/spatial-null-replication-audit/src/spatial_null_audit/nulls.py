"""Spatial null models for parcellated cortical maps, implemented from scratch in numpy/scipy.

All functions work on a 1-D map ``x`` of length ``n`` (one value per parcel) together with either
unit-sphere coordinates ``coords`` (n x 3) for the spin test or a distance matrix ``dist`` (n x n)
for the spectral and variogram nulls.

Nulls
-----
naive_permutation     plain shuffling (anti-conservative reference)
spin_surrogates       random rotations of parcel centroids on the sphere, hemisphere-mirrored
                      (Alexander-Bloch et al., 2018); ``assignment='nearest'`` allows duplicates,
                      ``assignment='unique'`` uses the one-to-one greedy matching of Vasa et al. (2018);
                      ``sa_tolerance`` discards rotations whose surrogate autocorrelation deviates
                      from the original's (projection-corrected spin, Imaging Neuroscience 2025).
moran_surrogates      Moran spectral randomisation, singleton scheme (Wagner & Dray, 2015)
variogram_surrogates  variogram-matched surrogates in the style of BrainSMASH (Burt et al., 2020)

Diagnostics
-----------
morans_i, variogram, autocorr_length (exponential-variogram length scale lambda)

Simulation helpers
------------------
fibonacci_sphere, project_to_sphere, gaussian_random_field
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

__all__ = [
    "fibonacci_sphere",
    "project_to_sphere",
    "sphere_distance",
    "gaussian_random_field",
    "naive_permutation",
    "random_rotation",
    "spin_surrogates",
    "moran_eigenvectors",
    "moran_surrogates",
    "variogram",
    "variogram_surrogates",
    "morans_i",
    "autocorr_length",
    "rank_resample",
]


# ------------------------------------------------------------ geometry
def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` approximately uniformly spaced points on the unit sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def project_to_sphere(coords: np.ndarray, hemi: Optional[np.ndarray] = None) -> np.ndarray:
    """Project (e.g. MNI) centroids to the unit sphere around each hemisphere's centre of mass.

    This is an approximation for volumetric atlases; surface-based claims should use the true
    spherical vertex coordinates (fsaverage/fsLR ``sphere`` surfaces).
    """
    coords = np.asarray(coords, dtype=float)
    out = np.empty_like(coords)
    groups = [np.arange(len(coords))] if hemi is None else [np.where(np.asarray(hemi) == h)[0] for h in np.unique(hemi)]
    for idx in groups:
        c = coords[idx] - coords[idx].mean(axis=0)
        out[idx] = c / np.linalg.norm(c, axis=1, keepdims=True)
    return out


def sphere_distance(coords: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix (radians) between unit-sphere points."""
    c = np.asarray(coords, dtype=float)
    dots = np.clip(c @ c.T, -1.0, 1.0)
    d = np.arccos(dots)
    np.fill_diagonal(d, 0.0)  # self-distance is exactly zero (arccos(1 - eps) is not)
    return d


def gaussian_random_field(dist: np.ndarray, length_scale: float, rng: Optional[np.random.Generator] = None, n_maps: int = 1, jitter: float = 1e-8) -> np.ndarray:
    """Draw ``n_maps`` smooth Gaussian random fields with covariance ``exp(-d / length_scale)``.

    Returns an array of shape ``(n_maps, n)`` (or ``(n,)`` when ``n_maps == 1``).
    """
    rng = rng or np.random.default_rng(0)
    dist = np.asarray(dist, dtype=float)
    cov = np.exp(-dist / float(length_scale)) + jitter * np.eye(len(dist))
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((len(dist), n_maps))
    fields = (L @ z).T
    return fields[0] if n_maps == 1 else fields


# ------------------------------------------------------- basic nulls
def naive_permutation(x: np.ndarray, n_perm: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Plain permutations of ``x`` (shape ``(n_perm, n)``)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    return np.stack([x[rng.permutation(x.size)] for _ in range(n_perm)])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3x3 rotation (Haar) via QR of a Gaussian matrix with sign correction."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _assign_unique(orig: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    """One-to-one greedy matching (Vasa et al., 2018): repeatedly pair the closest remaining points."""
    d = sphere_distance_between(orig, rotated)
    n = d.shape[0]
    perm = np.full(n, -1, dtype=int)
    d = d.copy()
    for _ in range(n):
        i, j = np.unravel_index(np.argmin(d), d.shape)
        perm[i] = j
        d[i, :] = np.inf
        d[:, j] = np.inf
    return perm


def sphere_distance_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distances between two point sets on the unit sphere."""
    return np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def spin_surrogates(
    x: np.ndarray,
    coords: np.ndarray,
    n_perm: int,
    hemi: Optional[np.ndarray] = None,
    assignment: str = "unique",
    rng: Optional[np.random.Generator] = None,
    sa_tolerance: Optional[float] = None,
    dist: Optional[np.ndarray] = None,
    max_tries_factor: int = 20,
    return_perms: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Spin-test surrogates of a parcellated map.

    Parameters
    ----------
    x : map values (n,).
    coords : unit-sphere coordinates of the parcels (n, 3).
    hemi : optional labels (e.g. 'L'/'R'); the rotation applied to the second hemisphere is the
        mirror image (x -> -x) of the first, as in Alexander-Bloch et al. (2018).
    assignment : 'nearest' (each original parcel takes the value of the nearest rotated parcel,
        duplicates possible) or 'unique' (one-to-one greedy matching, Vasa et al. 2018).
    sa_tolerance : if given, a rotation is accepted only when the surrogate's Moran's I is within
        ``sa_tolerance`` (absolute) of the original's; ``dist`` is then required. This is the
        projection-corrected spin test.
    return_perms : also return the (n_perm, n) index array.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n = x.size
    if sa_tolerance is not None and dist is None:
        raise ValueError("sa_tolerance requires dist")
    hemi_arr = np.zeros(n, dtype=int) if hemi is None else (np.asarray(hemi) == np.asarray(hemi)[0]).astype(int)
    groups = [np.where(hemi_arr == g)[0] for g in np.unique(hemi_arr)]
    mirror = np.diag([-1.0, 1.0, 1.0])
    i_orig = morans_i(x, dist) if sa_tolerance is not None else None

    perms = np.empty((n_perm, n), dtype=int)
    surr = np.empty((n_perm, n), dtype=float)
    accepted = 0
    tries = 0
    max_tries = max_tries_factor * n_perm
    while accepted < n_perm and tries < max_tries:
        tries += 1
        R = random_rotation(rng)
        perm = np.empty(n, dtype=int)
        for g_idx, idx in enumerate(groups):
            Rg = R if g_idx == 0 else mirror @ R @ mirror
            rotated = coords[idx] @ Rg.T
            if assignment == "nearest":
                d = sphere_distance_between(coords[idx], rotated)
                local = np.argmin(d, axis=1)
            elif assignment == "unique":
                local = _assign_unique(coords[idx], rotated)
            else:
                raise ValueError("assignment must be 'nearest' or 'unique'")
            perm[idx] = idx[local]
        cand = x[perm]
        if sa_tolerance is not None and abs(morans_i(cand, dist) - i_orig) > sa_tolerance:
            continue
        perms[accepted] = perm
        surr[accepted] = cand
        accepted += 1
    if accepted < n_perm:
        raise RuntimeError(f"only {accepted}/{n_perm} rotations accepted within sa_tolerance={sa_tolerance}")
    return (surr, perms) if return_perms else surr


# ------------------------------------------------- Moran spectral null
def _weights(dist: np.ndarray, kernel: str = "inverse", bandwidth: Optional[float] = None) -> np.ndarray:
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    with np.errstate(divide="ignore"):
        if kernel == "inverse":
            w = 1.0 / dist
        elif kernel == "gaussian":
            bw = bandwidth or np.median(dist[dist > 0])
            w = np.exp(-((dist / bw) ** 2))
        elif kernel == "exp":
            bw = bandwidth or np.median(dist[dist > 0])
            w = np.exp(-dist / bw)
        else:
            raise ValueError("kernel must be inverse|gaussian|exp")
    w[~np.isfinite(w)] = 0.0
    w[np.arange(n), np.arange(n)] = 0.0
    return 0.5 * (w + w.T)


def moran_eigenvectors(dist: np.ndarray, kernel: str = "inverse", bandwidth: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Moran eigenvector maps: eigen-decomposition of the doubly-centred weight matrix ``H W H``."""
    W = _weights(dist, kernel, bandwidth)
    n = W.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    vals, vecs = np.linalg.eigh(H @ W @ H)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def moran_surrogates(
    x: np.ndarray,
    dist: np.ndarray,
    n_perm: int,
    rng: Optional[np.random.Generator] = None,
    kernel: str = "inverse",
    bandwidth: Optional[float] = None,
    mems: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Moran spectral randomisation, singleton scheme (Wagner & Dray, 2015).

    The map is expanded on the Moran eigenvector basis; each surrogate keeps the magnitude of every
    coefficient (so the Moran spectrum, hence spatial autocorrelation, is preserved) and randomises
    the signs. Surrogates are rescaled to the original mean and standard deviation.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    if mems is None:
        _, mems = moran_eigenvectors(dist, kernel, bandwidth)
    mems = mems[:, :-1]  # drop the constant (zero-eigenvalue) direction
    xc = x - x.mean()
    coef = mems.T @ xc
    signs = rng.choice([-1.0, 1.0], size=(n_perm, coef.size))
    surr = (signs * coef) @ mems.T
    surr = surr - surr.mean(axis=1, keepdims=True)
    surr *= x.std() / np.maximum(surr.std(axis=1, keepdims=True), 1e-12)
    return surr + x.mean()


# ------------------------------------------------- variogram surrogates
def variogram(x: np.ndarray, dist: np.ndarray, n_bins: int = 25, max_dist_frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Binned empirical semivariogram ``gamma(h) = mean 0.5 (x_i - x_j)^2`` over pairs with d in bin.

    Bins are equally spaced from 0 to ``max_dist_frac * max(dist)`` (BrainSMASH's default of the
    25th percentile behaves similarly). Returns ``(bin_centres, gamma)``.
    """
    x = np.asarray(x, dtype=float)
    dist = np.asarray(dist, dtype=float)
    iu = np.triu_indices_from(dist, k=1)
    d = dist[iu]
    g = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    edges = np.linspace(0, max_dist_frac * d.max(), n_bins + 1)
    which = np.digitize(d, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.array([g[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)])
    ok = np.isfinite(gamma)
    return centres[ok], gamma[ok]


def rank_resample(surr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Replace each surrogate's values by the original values with the same ranks."""
    x_sorted = np.sort(np.asarray(x, dtype=float))
    ranks = np.argsort(np.argsort(surr, axis=-1), axis=-1)
    return x_sorted[ranks]


def variogram_surrogates(
    x: np.ndarray,
    dist: np.ndarray,
    n_perm: int,
    rng: Optional[np.random.Generator] = None,
    n_bandwidths: int = 20,
    n_bins: int = 25,
    max_dist_frac: float = 0.25,
    resample: bool = True,
) -> np.ndarray:
    """Variogram-matched surrogates in the spirit of BrainSMASH (Burt et al., 2020).

    For each surrogate: permute ``x``; for a grid of exponential-kernel bandwidths, smooth the permuted
    map; regress the target variogram on the smoothed map's variogram (``gamma_x ~ alpha + beta
    gamma_s``, ``alpha, beta >= 0``); keep the best bandwidth; return
    ``sqrt(beta) * smoothed + sqrt(alpha) * noise`` (optionally rank-resampled to the original values).
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    dist = np.asarray(dist, dtype=float)
    n = x.size
    dmax = dist.max()
    bandwidths = np.geomspace(0.02 * dmax, 1.0 * dmax, n_bandwidths)
    kernels = [np.exp(-dist / b) for b in bandwidths]
    for K in kernels:
        K /= K.sum(axis=1, keepdims=True)
    h, gamma_x = variogram(x, dist, n_bins, max_dist_frac)
    out = np.empty((n_perm, n))
    for k in range(n_perm):
        xp = x[rng.permutation(n)]
        best = (np.inf, None, 0.0, 0.0)
        for K in kernels:
            s = K @ xp
            s = (s - s.mean()) / max(s.std(), 1e-12)
            _, gamma_s = variogram(s, dist, n_bins, max_dist_frac)
            m = min(gamma_s.size, gamma_x.size)
            A = np.column_stack([np.ones(m), gamma_s[:m]])
            coef, _ = optimize.nnls(A, gamma_x[:m])
            sse = float(np.sum((A @ coef - gamma_x[:m]) ** 2))
            if sse < best[0]:
                best = (sse, s, coef[0], coef[1])
        _, s, alpha, beta = best
        surr = np.sqrt(beta) * s + np.sqrt(alpha) * rng.standard_normal(n)
        out[k] = surr
    if resample:
        out = rank_resample(out, x)
    else:
        out = (out - out.mean(axis=1, keepdims=True)) / np.maximum(out.std(axis=1, keepdims=True), 1e-12)
        out = out * x.std() + x.mean()
    return out


# ------------------------------------------------------------ diagnostics
def morans_i(x: np.ndarray, dist: np.ndarray, kernel: str = "inverse", bandwidth: Optional[float] = None) -> float:
    """Moran's I with inverse-distance (default) weights."""
    x = np.asarray(x, dtype=float)
    W = _weights(dist, kernel, bandwidth)
    z = x - x.mean()
    denom = float(z @ z)
    if denom == 0:
        return 0.0
    return float(x.size / W.sum() * (z @ W @ z) / denom)


def autocorr_length(x: np.ndarray, dist: np.ndarray, n_bins: int = 25, max_dist_frac: float = 0.5) -> Dict[str, float]:
    """Fit ``gamma(h) = nugget + sill (1 - exp(-h / lambda))`` to the binned variogram.

    Returns ``lambda`` (autocorrelation length in the units of ``dist``), sill and nugget. Larger
    lambda means smoother maps and fewer effective degrees of freedom.
    """
    h, g = variogram(x, dist, n_bins, max_dist_frac)
    if h.size < 4:
        return {"lambda": float("nan"), "sill": float("nan"), "nugget": float("nan")}

    def model(hh: np.ndarray, nugget: float, sill: float, lam: float) -> np.ndarray:
        return nugget + sill * (1 - np.exp(-hh / lam))

    p0 = (0.0, float(np.nanmax(g)), float(h.max() / 3))
    try:
        popt, _ = optimize.curve_fit(model, h, g, p0=p0, bounds=([0, 0, 1e-6], [np.inf, np.inf, 10 * h.max()]), maxfev=5000)
    except (RuntimeError, ValueError):
        return {"lambda": float("nan"), "sill": float("nan"), "nugget": float("nan")}
    return {"lambda": float(popt[2]), "sill": float(popt[1]), "nugget": float(popt[0])}
