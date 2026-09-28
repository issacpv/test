"""E-field dose metrics and an analytical toy head model.

The real pipeline reads SimNIBS ``*_scalar.msh`` files (via ``simnibs.read_msh``)
and evaluates the same metrics on tetrahedral grey-matter elements. Here the
metrics operate on plain arrays (``|E|`` per node, node coordinates, ROI
mask) so they can be tested without SimNIBS, and a homogeneous-sphere model
with two surface point electrodes provides a physically sensible synthetic
field (potential given by a Legendre series; E = -grad V by finite
differences).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import eval_legendre


@dataclass(frozen=True)
class DoseMetrics:
    """Summary of an E-field for one subject and montage (all in V/m unless noted)."""

    roi_mean: float
    roi_normal_mean: float
    p99: float
    gm_mean: float
    focality_fraction: float  # fraction of GM volume with |E| >= 0.5 * p99
    current_mA: float

    @property
    def gain(self) -> float:
        """Global gain: grey-matter mean |E| (used to separate gain from shape)."""
        return self.gm_mean


def sphere_potential(
    points: np.ndarray,
    source: np.ndarray,
    sink: np.ndarray,
    current_a: float = 1e-3,
    sigma: float = 0.33,
    radius: float = 0.09,
    n_terms: int = 60,
) -> np.ndarray:
    """Potential (V) inside a homogeneous conducting sphere with two surface point electrodes.

    Uses the Legendre-series solution for a point current injected at the
    surface of a sphere with insulating boundary elsewhere; the anode/cathode
    pair is obtained by superposition.

    Parameters
    ----------
    points : (N, 3) array of positions in metres (|r| <= radius).
    source, sink : (3,) unit vectors pointing to the anode and cathode.
    current_a : injected current in amperes.
    sigma : conductivity in S/m.
    radius : sphere radius in metres.
    n_terms : number of Legendre terms.
    """
    pts = np.asarray(points, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    rr = np.clip(r / radius, 0.0, 1.0 - 1e-9)
    unit = np.divide(pts, np.maximum(r, 1e-12)[:, None])
    cos_s = unit @ (np.asarray(source) / np.linalg.norm(source))
    cos_k = unit @ (np.asarray(sink) / np.linalg.norm(sink))
    n = np.arange(1, n_terms + 1)
    coef = (2 * n + 1) / n  # (n_terms,)
    radial = rr[:, None] ** n[None, :]  # (N, n_terms)
    ps = np.stack([eval_legendre(k, cos_s) for k in n], axis=1)
    pk = np.stack([eval_legendre(k, cos_k) for k in n], axis=1)
    series = (coef[None, :] * radial * (ps - pk)).sum(axis=1)
    return current_a / (4 * np.pi * sigma * radius) * series


def sphere_efield(
    points: np.ndarray,
    source: np.ndarray,
    sink: np.ndarray,
    current_a: float = 1e-3,
    sigma: float = 0.33,
    radius: float = 0.09,
    h: float = 5e-4,
    n_terms: int = 60,
) -> np.ndarray:
    """E-field vectors (V/m) at ``points`` from :func:`sphere_potential` via central differences."""
    pts = np.asarray(points, dtype=float)
    e = np.zeros_like(pts)
    for ax in range(3):
        d = np.zeros(3)
        d[ax] = h
        vp = sphere_potential(pts + d, source, sink, current_a, sigma, radius, n_terms)
        vm = sphere_potential(pts - d, source, sink, current_a, sigma, radius, n_terms)
        e[:, ax] = -(vp - vm) / (2 * h)
    return e


def shell_points(radius_inner: float, radius_outer: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random points in a spherical shell (a stand-in for grey matter)."""
    u = rng.random(n)
    r = (radius_inner**3 + u * (radius_outer**3 - radius_inner**3)) ** (1 / 3)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r[:, None]


def scale_to_current(e: np.ndarray, from_ma: float, to_ma: float) -> np.ndarray:
    """E-fields are linear in current: rescale a field simulated at ``from_ma`` to ``to_ma``."""
    if from_ma <= 0:
        raise ValueError("from_ma must be positive")
    return np.asarray(e) * (to_ma / from_ma)


def current_for_target(e_per_ma: float, target_v_per_m: float, max_ma: float = 4.0) -> float:
    """Current (mA) needed to reach ``target_v_per_m`` given the field at 1 mA, capped at ``max_ma``."""
    if e_per_ma <= 0:
        raise ValueError("e_per_ma must be positive")
    return float(min(target_v_per_m / e_per_ma, max_ma))


def dose_metrics(
    e_vec: np.ndarray,
    gm_mask: np.ndarray,
    roi_mask: np.ndarray,
    roi_normals: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    current_ma: float = 1.0,
) -> DoseMetrics:
    """Compute ROI and global dose metrics from E-field vectors.

    Parameters
    ----------
    e_vec : (N, 3) E-field vectors.
    gm_mask : (N,) bool, grey-matter elements.
    roi_mask : (N,) bool, target ROI elements (subset of GM).
    roi_normals : (N, 3) optional unit surface normals for the normal component.
    volumes : (N,) optional element volumes for volume-weighted focality.
    current_ma : current the field was simulated at (stored for provenance).
    """
    e_vec = np.asarray(e_vec, dtype=float)
    mag = np.linalg.norm(e_vec, axis=1)
    gm = np.asarray(gm_mask, bool)
    roi = np.asarray(roi_mask, bool) & gm
    if roi.sum() == 0 or gm.sum() == 0:
        raise ValueError("empty ROI or GM mask")
    vol = np.ones(len(mag)) if volumes is None else np.asarray(volumes, float)
    p99 = float(np.percentile(mag[gm], 99))
    focal = float(vol[gm & (mag >= 0.5 * p99)].sum() / vol[gm].sum())
    if roi_normals is not None:
        normal = float(np.mean(np.einsum("ij,ij->i", e_vec[roi], np.asarray(roi_normals)[roi])))
    else:
        normal = float("nan")
    return DoseMetrics(
        roi_mean=float(np.average(mag[roi], weights=vol[roi])),
        roi_normal_mean=normal,
        p99=p99,
        gm_mean=float(np.average(mag[gm], weights=vol[gm])),
        focality_fraction=focal,
        current_mA=current_ma,
    )
