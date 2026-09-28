"""Spatial null models for parcellated cortical maps.

Implemented (pure numpy/scipy so they can be audited and unit-tested):

``spin_permutations``
    Parcel-level spin test (Alexander-Bloch et al., 2018, *NeuroImage*):
    random rotations of parcel centroids on the sphere with a mirrored
    rotation for the right hemisphere.  Two assignment schemes:
    ``"nearest"`` (each rotated parcel takes the value of its nearest
    original parcel; duplicates possible) and ``"vasa"`` (Váša et al., 2018,
    *Cereb Cortex*: greedy unique assignment so the result is a true
    permutation).
``variogram_surrogates``
    BrainSMASH-style surrogates (Burt et al., 2020, *NeuroImage*): permute,
    smooth with a distance kernel at several neighbourhood scales, pick the
    scale whose variogram best matches the target, regress and rank-match.
``moran_spectral_randomization``
    Moran spectral randomization (Wagner & Dray, 2015, *Methods Ecol Evol*;
    used for brain maps by Markello & Misic, 2021): random sign flips
    ("singleton") or paired rotations of Moran eigenvector coefficients,
    preserving Moran's I of the map for a given weight matrix.
``naive_permutations``
    Non-spatial shuffle, as the (wrong) reference.

Wrappers ``neuromaps_nulls`` / ``brainsmash_surrogates`` call the reference
implementations when the packages are installed, so that the pure-numpy
versions can be validated against them (a stated deliverable of the audit).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.stats import special_ortho_group


# ----------------------------------------------------------------------- spins
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    return special_ortho_group.rvs(3, random_state=rng)


def spin_permutations(
    coords_lh: np.ndarray, coords_rh: Optional[np.ndarray] = None, n_perm: int = 1000,
    method: str = "vasa", seed: int = 0,
) -> np.ndarray:
    """Spin-test permutation indices for parcel centroids.

    Parameters
    ----------
    coords_lh, coords_rh
        (n_lh, 3) and (n_rh, 3) unit-sphere centroids.  Right-hemisphere
        rotations are the left rotation reflected across the sagittal plane
        (``R_rh = M R M`` with ``M = diag(-1, 1, 1)``).
    method
        ``"nearest"`` (with replacement) or ``"vasa"`` (unique assignment).

    Returns
    -------
    (n_perm, n_lh + n_rh) integer array: row ``k`` gives, for each original
    parcel position, the index of the parcel whose value it takes.
    """
    rng = np.random.default_rng(seed)
    M = np.diag([-1.0, 1.0, 1.0])
    hemis = [coords_lh] + ([coords_rh] if coords_rh is not None else [])
    offsets = np.cumsum([0] + [h.shape[0] for h in hemis])
    perms = np.empty((n_perm, offsets[-1]), dtype=int)
    for k in range(n_perm):
        R = _random_rotation(rng)
        for h, coords in enumerate(hemis):
            Rh = R if h == 0 else M @ R @ M
            c = coords / np.linalg.norm(coords, axis=1, keepdims=True)
            rotated = c @ Rh.T
            if method == "nearest":
                _, idx = cKDTree(c).query(rotated, k=1)
            elif method == "vasa":
                idx = _vasa_assignment(c, rotated)
            else:
                raise ValueError("method must be 'nearest' or 'vasa'")
            perms[k, offsets[h]:offsets[h + 1]] = idx + offsets[h]
    return perms


def _vasa_assignment(original: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    """Greedy unique matching: process rotated parcels in order of their
    minimum distance to any unassigned original parcel (Váša et al., 2018)."""
    n = original.shape[0]
    D = np.arccos(np.clip(rotated @ original.T, -1, 1))  # angular distance rotated x original
    assigned = np.full(n, -1)
    free_orig = np.ones(n, dtype=bool)
    free_rot = np.ones(n, dtype=bool)
    Dw = D.copy()
    for _ in range(n):
        Dw[~free_rot, :] = np.inf
        Dw[:, ~free_orig] = np.inf
        # the rotated parcel with the smallest available distance is matched first
        i, j = np.unravel_index(np.argmin(Dw), Dw.shape)
        assigned[i] = j
        free_rot[i] = False
        free_orig[j] = False
    return assigned


def naive_permutations(n: int, n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(n_perm)])


# ------------------------------------------------------------------ variogram
def _variogram(x: np.ndarray, D: np.ndarray, bins: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    g = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    which = np.digitize(d, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = which == b
        if m.any():
            out[b] = g[m].mean()
    return out


def _smooth(x: np.ndarray, D: np.ndarray, k: int, kernel: str = "exp") -> np.ndarray:
    n = x.size
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    d = np.take_along_axis(D, idx, axis=1)
    dk = d[:, -1:]
    if kernel == "exp":
        w = np.exp(-d / np.maximum(dk, 1e-12))
    elif kernel == "gaussian":
        w = np.exp(-1.25 * (d / np.maximum(dk, 1e-12)) ** 2)
    elif kernel == "uniform":
        w = np.ones_like(d)
    else:
        raise ValueError(kernel)
    return (w * x[idx]).sum(axis=1) / w.sum(axis=1)


def variogram_surrogates(
    x: np.ndarray, D: np.ndarray, n_surr: int = 1000, deltas: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
    n_bins: int = 25, pv: float = 25.0, kernel: str = "exp", resample: bool = True, seed: int = 0,
) -> np.ndarray:
    """Variogram-matched surrogate maps (BrainSMASH-style).

    Parameters
    ----------
    x : (n,) target map
    D : (n, n) distance matrix (geodesic / great-circle)
    deltas : neighbourhood fractions to try for smoothing
    pv : percentile of pairwise distances up to which the variogram is fitted
    resample : rank-match surrogate values to the original distribution

    Returns
    -------
    (n_surr, n) array of surrogates.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = x.size
    ok = ~np.isnan(x)
    xs = (x[ok] - x[ok].mean()) / x[ok].std()
    Ds = D[np.ix_(ok, ok)]
    dmax = np.percentile(Ds[np.triu_indices_from(Ds, k=1)], pv)
    bins = np.linspace(0, dmax, n_bins + 1)
    v_target = _variogram(xs, Ds, bins)
    good = ~np.isnan(v_target)
    sorted_x = np.sort(x[ok])
    out = np.full((n_surr, n), np.nan)
    m = ok.sum()
    for s in range(n_surr):
        perm = xs[rng.permutation(m)]
        best, best_err = None, np.inf
        for delta in deltas:
            k = max(int(delta * m), 1)
            sm = _smooth(perm, Ds, k, kernel)
            v_sm = _variogram(sm, Ds, bins)
            A = np.column_stack([v_sm[good], np.ones(good.sum())])
            coef, *_ = np.linalg.lstsq(A, v_target[good], rcond=None)
            alpha, beta = coef
            err = np.sum((A @ coef - v_target[good]) ** 2)
            if err < best_err:
                best_err, best = err, (sm, alpha, beta)
        sm, alpha, beta = best
        surr = np.sqrt(np.abs(alpha)) * sm + np.sqrt(np.abs(beta)) * rng.normal(size=m)
        if resample:
            surr = sorted_x[np.argsort(np.argsort(surr))]
        out[s, ok] = surr
    return out


# ---------------------------------------------------------------------- Moran
def knn_weights(D: np.ndarray, k: int = 6, symmetric: bool = True) -> np.ndarray:
    """Binary k-nearest-neighbour spatial weight matrix from a distance matrix."""
    n = D.shape[0]
    W = np.zeros((n, n))
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    W[np.arange(n)[:, None], idx] = 1.0
    if symmetric:
        W = np.maximum(W, W.T)
    return W


def inverse_distance_weights(D: np.ndarray, power: float = 1.0) -> np.ndarray:
    with np.errstate(divide="ignore"):
        W = 1.0 / D ** power
    W[~np.isfinite(W)] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def moran_i(x: np.ndarray, W: np.ndarray) -> float:
    """Moran's I spatial autocorrelation statistic."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    num = z @ W @ z
    den = z @ z
    return float(x.size / W.sum() * num / den)


def moran_eigenvectors(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Moran eigenvector maps of the doubly-centred symmetric weight matrix."""
    n = W.shape[0]
    Ws = 0.5 * (W + W.T)
    H = np.eye(n) - np.ones((n, n)) / n
    vals, vecs = np.linalg.eigh(H @ Ws @ H)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = np.abs(vals) > 1e-10 * np.max(np.abs(vals))
    return vals[keep], vecs[:, keep]


def moran_spectral_randomization(
    x: np.ndarray, W: np.ndarray, n_surr: int = 1000, procedure: str = "singleton", seed: int = 0,
) -> np.ndarray:
    """Surrogates preserving Moran's I via spectral randomization.

    ``procedure="singleton"`` flips the sign of each eigenvector coefficient
    (Moran's I preserved exactly); ``"pair"`` rotates coefficients of
    consecutive eigenvector pairs by a random angle (I preserved approximately,
    many more distinct surrogates).  Surrogates have the same mean and
    variance as ``x``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    _, E = moran_eigenvectors(W)
    z = x - x.mean()
    coef = E.T @ z
    out = np.empty((n_surr, x.size))
    for s in range(n_surr):
        c = coef.copy()
        if procedure == "singleton":
            c *= rng.choice([-1.0, 1.0], size=c.size)
        elif procedure == "pair":
            for i in range(0, c.size - 1, 2):
                theta = rng.uniform(0, 2 * np.pi)
                a, b = c[i], c[i + 1]
                c[i] = a * np.cos(theta) - b * np.sin(theta)
                c[i + 1] = a * np.sin(theta) + b * np.cos(theta)
            if c.size % 2 == 1:
                c[-1] *= rng.choice([-1.0, 1.0])
        else:
            raise ValueError(procedure)
        out[s] = E @ c + x.mean()
    return out


# ------------------------------------------------------------------- inference
def null_pvalue(observed: float, null: np.ndarray, two_sided: bool = True) -> float:
    """Permutation p-value with the +1 correction (Phipson & Smyth, 2010)."""
    null = np.asarray(null, dtype=float)
    null = null[~np.isnan(null)]
    if two_sided:
        k = np.sum(np.abs(null) >= abs(observed))
    else:
        k = np.sum(null >= observed)
    return float((k + 1) / (null.size + 1))


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if method == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def compare_nulls(
    x: np.ndarray, y: np.ndarray, coords_lh: np.ndarray, coords_rh: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None, W: Optional[np.ndarray] = None, n_perm: int = 500,
    corr: str = "pearson", seed: int = 0, methods: Sequence[str] = ("naive", "spin_nearest", "spin_vasa", "variogram", "moran"),
) -> Dict[str, Dict[str, float]]:
    """Correlation between maps ``x`` and ``y`` with p-values under each null.

    ``x`` is the map that gets permuted/surrogated (the "target" in
    imaging-transcriptomics parlance).  Returns ``{method: {"r": r, "p": p}}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r_obs = _corr(x, y, corr)
    out: Dict[str, Dict[str, float]] = {}
    for m in methods:
        if m == "naive":
            perms = naive_permutations(x.size, n_perm, seed)
            null = np.array([_corr(x[p], y, corr) for p in perms])
        elif m.startswith("spin"):
            perms = spin_permutations(coords_lh, coords_rh, n_perm, method=m.split("_")[1], seed=seed)
            null = np.array([_corr(x[p], y, corr) for p in perms])
        elif m == "variogram":
            if D is None:
                raise ValueError("variogram null needs a distance matrix D")
            surr = variogram_surrogates(x, D, n_perm, seed=seed)
            null = np.array([_corr(s, y, corr) for s in surr])
        elif m == "moran":
            if W is None:
                if D is None:
                    raise ValueError("moran null needs W or D")
                W = knn_weights(D, k=6)
            surr = moran_spectral_randomization(x, W, n_perm, seed=seed)
            null = np.array([_corr(s, y, corr) for s in surr])
        else:
            raise ValueError(m)
        out[m] = {"r": r_obs, "p": null_pvalue(r_obs, null), "null_sd": float(np.nanstd(null))}
    return out


# --------------------------------------------------------------------- wrappers
def neuromaps_nulls(data: np.ndarray, atlas: str = "fsLR", density: str = "32k", parcellation=None,
                    method: str = "vasa", n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    """Reference implementation via ``neuromaps.nulls`` (Markello et al., 2022, *Nat Methods*)."""
    try:
        from neuromaps import nulls  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install neuromaps") from e
    fn: Callable = getattr(nulls, method)
    return fn(data, atlas=atlas, density=density, parcellation=parcellation, n_perm=n_perm, seed=seed)


def brainsmash_surrogates(x: np.ndarray, D: np.ndarray, n_surr: int = 1000, seed: int = 0, **kwargs) -> np.ndarray:
    """Reference implementation via ``brainsmash.mapgen.base.Base`` (Burt et al., 2020)."""
    try:
        from brainsmash.mapgen.base import Base  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install brainsmash") from e
    gen = Base(x=np.asarray(x, dtype=float), D=np.asarray(D, dtype=float), seed=seed, **kwargs)
    return np.asarray(gen(n=n_surr))
