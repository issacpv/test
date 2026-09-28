"""Spatially constrained null models for comparing brain maps.

Correlating two brain maps and reporting a parametric p-value is invalid: both
maps are spatially autocorrelated, so the effective number of independent
observations is a small fraction of the parcel count, and naive p-values are
anticonservative by orders of magnitude. This matters acutely here because an
E-field map is *extremely* smooth (it is the solution of an elliptic PDE) and a
seed-based connectivity map is smooth too, so a spurious correlation between them
is close to guaranteed.

Two standard null families are implemented:

* **Spin test** (Alexander-Bloch et al. 2018, *NeuroImage*): rotate parcel
  centroids on the sphere by a random rotation and reassign values by nearest
  neighbour. Exactly preserves the empirical spatial autocorrelation because it
  moves the real map rigidly; costs a small amount of distortion from the
  nearest-neighbour matching.
* **Variogram matching** (Burt et al. 2020, *NeuroImage*, the BrainSMASH
  algorithm): permute the map, smooth it with a distance kernel, and rescale so
  the surrogate's variogram matches the target's. Works for any geometry
  including volumetric and subcortical data, where a sphere has no meaning.

Markello & Misic (2021, *NeuroImage*) show these null families are not
interchangeable and can disagree substantially, so the honest practice -- adopted
here -- is to report both and treat a result that survives only one as
provisional. If ``neuromaps`` or ``brainsmash`` is installed, prefer their
reference implementations for a published analysis; these are dependency-free
equivalents suitable for development and for environments where those packages
cannot be installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

__all__ = [
    "NullResult",
    "random_rotation",
    "spin_surrogates",
    "variogram_surrogates",
    "variogram",
    "spatial_correlation_test",
    "effective_dof",
]

LOG = logging.getLogger(__name__)

CorrKind = Literal["pearson", "spearman"]


@dataclass
class NullResult:
    """Outcome of a spatially constrained map comparison.

    Attributes:
        statistic: Observed correlation between the two maps.
        p_value: Two-sided p-value against the surrogate distribution.
        n_surrogates: Number of valid surrogates used.
        null_mean: Mean of the surrogate statistics.
        null_sd: Standard deviation of the surrogate statistics.
        method: ``"spin"`` or ``"variogram"``.
        corr_kind: Which correlation was used.
        z_score: ``(statistic - null_mean) / null_sd``, for reporting effect size
            on a scale comparable across analyses.
        parametric_p: Naive parametric p-value, reported **only** so the inflation
            relative to ``p_value`` can be quantified in the paper.
    """

    statistic: float
    p_value: float
    n_surrogates: int
    null_mean: float
    null_sd: float
    method: str
    corr_kind: CorrKind
    z_score: float
    parametric_p: float

    def inflation_factor(self) -> float:
        """Ratio of the spatial p-value to the naive parametric one."""
        if self.parametric_p <= 0:
            return float("inf")
        return self.p_value / self.parametric_p


def random_rotation(rng: np.random.Generator, *, reflect: bool = True) -> np.ndarray:
    """Draw a uniformly random 3x3 rotation matrix.

    Uses a QR decomposition of a Gaussian matrix, with the sign correction that
    makes the distribution Haar-uniform (without it, the result is biased).

    Args:
        rng: Random generator.
        reflect: Allow improper rotations (determinant -1). The original spin test
            permits reflections; they double the effective number of distinct
            surrogates.

    Returns:
        (3, 3) orthogonal matrix.
    """
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if not reflect and np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def spin_surrogates(
    values: np.ndarray,
    coords: np.ndarray,
    *,
    n_surrogates: int = 1000,
    hemisphere: np.ndarray | None = None,
    seed: int = 0,
    reflect: bool = True,
) -> np.ndarray:
    """Generate spin-test surrogates for a parcellated map.

    Parcel centroids are projected onto the unit sphere, rotated, and each rotated
    position is matched to its nearest original centroid; the map value at the
    matched parcel becomes the surrogate value.

    When ``hemisphere`` is given, the rotation is applied **within** each
    hemisphere with the mirrored rotation on the right, which is the standard
    treatment: a rotation that swaps hemispheres would destroy the strong
    left-right symmetry that both connectivity and E-field maps possess, making
    the null too easy to beat.

    Args:
        values: (p,) map values. ``nan`` entries propagate to surrogates.
        coords: (p, 3) parcel centroid coordinates in mm.
        n_surrogates: Number of surrogates.
        hemisphere: (p,) labels containing ``"L"``/``"R"``; other values are
            rotated as a single extra group.
        seed: RNG seed.
        reflect: Passed to :func:`random_rotation`.

    Returns:
        (n_surrogates, p) array of surrogate maps.

    Raises:
        ValueError: If shapes are inconsistent or ``n_surrogates < 1``.
    """
    values = np.asarray(values, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    p = values.size
    if coords.shape != (p, 3):
        raise ValueError(f"coords must be ({p}, 3), got {coords.shape}")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1")

    if hemisphere is None:
        groups = [np.arange(p)]
    else:
        hemi = np.asarray(hemisphere).astype(str)
        groups = [np.flatnonzero(hemi == "L"), np.flatnonzero(hemi == "R")]
        other = np.flatnonzero(~np.isin(hemi, ("L", "R")))
        if other.size:
            groups.append(other)
        groups = [g for g in groups if g.size]

    unit: list[np.ndarray] = []
    for g in groups:
        centred = coords[g] - coords[g].mean(axis=0)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        unit.append(centred / np.where(norms == 0, 1.0, norms))

    rng = np.random.default_rng(seed)
    out = np.empty((n_surrogates, p), dtype=float)
    for s in range(n_surrogates):
        rot = random_rotation(rng, reflect=reflect)
        for g, sphere in zip(groups, unit):
            # Mirror the rotation across the midline for the right hemisphere so
            # both hemispheres move together rather than independently.
            r = rot.copy()
            if hemisphere is not None and str(np.asarray(hemisphere).astype(str)[g[0]]) == "R":
                flip = np.diag([-1.0, 1.0, 1.0])
                r = flip @ rot @ flip
            rotated = sphere @ r.T
            # Nearest original centroid for each rotated position.
            nearest = np.argmax(rotated @ sphere.T, axis=1)
            out[s, g] = values[g[nearest]]
    return out


def variogram(
    values: np.ndarray, distances: np.ndarray, *, n_bins: int = 25, max_distance: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical semivariogram of a map.

    Args:
        values: (p,) map values; ``nan`` pairs are excluded.
        distances: (p, p) pairwise distances.
        n_bins: Number of distance bins.
        max_distance: Largest distance considered; defaults to the 75th percentile
            of finite distances, since the far tail is dominated by few pairs and
            is unstable.

    Returns:
        ``(bin_centres, semivariance)``, both (n_bins,), with ``nan`` in empty bins.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    values = np.asarray(values, dtype=float).ravel()
    distances = np.asarray(distances, dtype=float)
    p = values.size
    if distances.shape != (p, p):
        raise ValueError(f"distances must be ({p}, {p}), got {distances.shape}")

    iu = np.triu_indices(p, k=1)
    d = distances[iu]
    gamma = 0.5 * (values[iu[0]] - values[iu[1]]) ** 2
    ok = np.isfinite(d) & np.isfinite(gamma)
    d, gamma = d[ok], gamma[ok]
    if d.size == 0:
        return np.full(n_bins, np.nan), np.full(n_bins, np.nan)

    hi = float(np.percentile(d, 75)) if max_distance is None else float(max_distance)
    hi = max(hi, float(np.min(d)) + 1e-9)
    edges = np.linspace(float(d.min()), hi, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    inside = d <= hi

    centres = 0.5 * (edges[:-1] + edges[1:])
    semi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = inside & (idx == b)
        if sel.any():
            semi[b] = float(gamma[sel].mean())
    return centres, semi


def variogram_surrogates(
    values: np.ndarray,
    distances: np.ndarray,
    *,
    n_surrogates: int = 1000,
    n_bins: int = 25,
    kernel_widths: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0),
    seed: int = 0,
) -> np.ndarray:
    """Generate variogram-matched surrogates (BrainSMASH-style).

    For each surrogate: permute the map, smooth the permutation with an
    exponential distance kernel at several bandwidths, and keep the bandwidth
    whose variogram best matches the target's. The surrogate is then rescaled to
    the target's mean and variance so only its *spatial* structure, not its
    marginal distribution, is what the null contributes.

    Unlike the spin test this needs no sphere, so it works for volumetric and
    subcortical maps -- which is why it is the primary null for the E-field
    comparisons here, E-field maps being volumetric by nature.

    Args:
        values: (p,) map values.
        distances: (p, p) pairwise distances.
        n_surrogates: Number of surrogates.
        n_bins: Variogram bins used for matching.
        kernel_widths: Bandwidths as multiples of the median pairwise distance.
        seed: RNG seed.

    Returns:
        (n_surrogates, p) surrogate maps.

    Raises:
        ValueError: If shapes are inconsistent or fewer than 3 finite values exist.
    """
    values = np.asarray(values, dtype=float).ravel()
    distances = np.asarray(distances, dtype=float)
    p = values.size
    if distances.shape != (p, p):
        raise ValueError(f"distances must be ({p}, {p}), got {distances.shape}")
    finite = np.isfinite(values)
    if finite.sum() < 3:
        raise ValueError("need at least 3 finite values")

    target_centres, target_semi = variogram(values, distances, n_bins=n_bins)
    valid_bins = np.isfinite(target_semi)
    median_d = float(np.median(distances[np.triu_indices(p, k=1)]))
    median_d = median_d if median_d > 0 else 1.0

    kernels = []
    for width in kernel_widths:
        scale = max(width * median_d, 1e-9)
        k = np.exp(-distances / scale)
        np.fill_diagonal(k, 1.0)
        kernels.append(k / k.sum(axis=1, keepdims=True))

    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    rng = np.random.default_rng(seed)
    out = np.empty((n_surrogates, p), dtype=float)

    for s in range(n_surrogates):
        base = values.copy()
        base[finite] = rng.permutation(values[finite])
        base = np.where(np.isfinite(base), base, mu)

        best: np.ndarray | None = None
        best_cost = np.inf
        for k in kernels:
            smoothed = k @ base
            sd = smoothed.std()
            candidate = mu + sigma * (smoothed - smoothed.mean()) / (sd if sd > 0 else 1.0)
            _, semi = variogram(candidate, distances, n_bins=n_bins)
            both = valid_bins & np.isfinite(semi)
            if not both.any():
                continue
            cost = float(np.mean((semi[both] - target_semi[both]) ** 2))
            if cost < best_cost:
                best_cost, best = cost, candidate
        out[s] = base if best is None else best
        out[s][~finite] = np.nan
    return out


def _correlate(x: np.ndarray, y: np.ndarray, kind: CorrKind) -> float:
    """Correlation over the pairwise-complete entries of two maps."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a, b = x[ok], y[ok]
    if kind == "spearman":
        a = np.argsort(np.argsort(a)).astype(float)
        b = np.argsort(np.argsort(b)).astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spatial_correlation_test(
    map_a: np.ndarray,
    map_b: np.ndarray,
    *,
    surrogates: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    method: Literal["spin", "variogram"] = "variogram",
    n_surrogates: int = 1000,
    corr_kind: CorrKind = "spearman",
    hemisphere: np.ndarray | None = None,
    seed: int = 0,
) -> NullResult:
    """Test the correlation between two brain maps against a spatial null.

    ``map_a`` is the one that gets randomised; make it the *predictor* (e.g. the
    connectivity map), keeping the observed effect map fixed.

    Args:
        map_a: (p,) first map; surrogates are generated from this one.
        map_b: (p,) second map.
        surrogates: (n, p) precomputed surrogates. Supplying these once and
            reusing them across many tests is both faster and fairer, since every
            test then faces the same null realisations.
        coords: (p, 3) centroids, required for ``method="spin"``.
        distances: (p, p) distances, required for ``method="variogram"``.
        method: Null family.
        n_surrogates: Number of surrogates when generating them here.
        corr_kind: ``"pearson"`` or ``"spearman"``. Spearman is the safer default
            for E-field maps, whose magnitude distribution is heavily skewed.
        hemisphere: (p,) hemisphere labels for the spin test.
        seed: RNG seed.

    Returns:
        A :class:`NullResult`.

    Raises:
        ValueError: If the maps differ in length, or the geometry required by the
            chosen method is missing.
    """
    map_a = np.asarray(map_a, dtype=float).ravel()
    map_b = np.asarray(map_b, dtype=float).ravel()
    if map_a.size != map_b.size:
        raise ValueError(f"maps must be the same length, got {map_a.size} and {map_b.size}")

    observed = _correlate(map_a, map_b, corr_kind)

    if surrogates is None:
        if method == "spin":
            if coords is None:
                raise ValueError("method='spin' requires coords")
            surrogates = spin_surrogates(
                map_a, coords, n_surrogates=n_surrogates, hemisphere=hemisphere, seed=seed
            )
        elif method == "variogram":
            if distances is None:
                raise ValueError("method='variogram' requires distances")
            surrogates = variogram_surrogates(
                map_a, distances, n_surrogates=n_surrogates, seed=seed
            )
        else:
            raise ValueError(f"unknown method {method!r}")
    surrogates = np.atleast_2d(np.asarray(surrogates, dtype=float))

    null = np.array([_correlate(s, map_b, corr_kind) for s in surrogates])
    null = null[np.isfinite(null)]
    n_valid = int(null.size)
    if n_valid == 0 or not np.isfinite(observed):
        return NullResult(observed, float("nan"), n_valid, float("nan"), float("nan"),
                          method, corr_kind, float("nan"), float("nan"))

    # Add-one smoothing: a permutation p-value should never be exactly zero.
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_valid + 1))
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1)) if n_valid > 1 else float("nan")

    ok = np.isfinite(map_a) & np.isfinite(map_b)
    n_eff = int(ok.sum())
    parametric_p = _parametric_p(observed, n_eff)

    return NullResult(
        statistic=observed,
        p_value=p_value,
        n_surrogates=n_valid,
        null_mean=null_mean,
        null_sd=null_sd,
        method=method,
        corr_kind=corr_kind,
        z_score=float((observed - null_mean) / null_sd) if null_sd and np.isfinite(null_sd) and null_sd > 0 else float("nan"),
        parametric_p=parametric_p,
    )


def _parametric_p(r: float, n: int) -> float:
    """Two-sided parametric p-value for a correlation, for inflation reporting only."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    try:
        from scipy import stats

        t = r * np.sqrt((n - 2) / max(1.0 - r**2, 1e-300))
        return float(2.0 * stats.t.sf(abs(t), df=n - 2))
    except ImportError:  # pragma: no cover
        return float("nan")


def effective_dof(map_a: np.ndarray, surrogates: np.ndarray, map_b: np.ndarray,
                  corr_kind: CorrKind = "spearman") -> float:
    """Effective degrees of freedom implied by a surrogate distribution.

    Inverts the standard-error formula ``sd(r) ~ 1/sqrt(n_eff - 3)`` against the
    observed spread of the null correlations. A useful sanity figure for a paper:
    it typically reveals that a 360-parcel map carries only a few dozen
    independent observations, which is the quantitative reason naive p-values fail.

    Args:
        map_a: (p,) predictor map (used only for its length).
        surrogates: (n, p) surrogates of ``map_a``.
        map_b: (p,) target map.
        corr_kind: Correlation used.

    Returns:
        Estimated effective number of independent observations.
    """
    null = np.array([_correlate(s, map_b, corr_kind) for s in np.atleast_2d(surrogates)])
    null = null[np.isfinite(null)]
    if null.size < 2:
        return float("nan")
    sd = float(null.std(ddof=1))
    if sd <= 0:
        return float("inf")
    return float(1.0 / sd**2 + 3.0)
