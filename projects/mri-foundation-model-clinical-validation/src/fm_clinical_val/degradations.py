"""Controlled image degradations on 3D numpy volumes (float32, arbitrary intensity scale).

All functions are deterministic given ``rng`` and preserve shape. Severity
parameters are chosen so that ``level`` in [0, 1] spans "none" to "severe" and
can be mapped onto MR-ART motion levels by calibration.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def make_phantom(size: int = 48, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic head-like phantom: background 0, 'brain' sphere (label 1) with a bright 'lesion' blob (label 2)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:size, :size, :size]
    c = size / 2
    r_brain = size * 0.38 * rng.uniform(0.95, 1.05)
    brain = ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= r_brain ** 2
    lc = c + rng.uniform(-0.15, 0.15, 3) * size
    r_les = size * rng.uniform(0.06, 0.1)
    lesion = ((z - lc[0]) ** 2 + (y - lc[1]) ** 2 + (x - lc[2]) ** 2) <= r_les ** 2
    lab = np.zeros((size,) * 3, np.int16)
    lab[brain] = 1
    lab[brain & lesion] = 2
    img = np.zeros((size,) * 3, np.float32)
    img[lab == 1] = 100.0
    img[lab == 2] = 160.0
    img = ndimage.gaussian_filter(img, 0.8) + rng.normal(0, 2.0, img.shape).astype(np.float32)
    return img.astype(np.float32), lab


def rician_noise(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Rician noise with sigma = level × 30% of the 99th-percentile intensity."""
    sigma = level * 0.3 * np.percentile(img, 99)
    if sigma <= 0:
        return img.copy()
    re = img + rng.normal(0, sigma, img.shape)
    im = rng.normal(0, sigma, img.shape)
    return np.sqrt(re ** 2 + im ** 2).astype(np.float32)


def kspace_motion(img: np.ndarray, level: float, rng: np.random.Generator, axis: int = 0) -> np.ndarray:
    """Motion artefact by perturbing the phase of a random subset of k-space lines along ``axis``.

    Simulates inter-shot rigid motion: a fraction ``level``×40% of phase-encoding lines
    receive a random linear phase ramp (translation) of up to ``level``×4 voxels.
    """
    if level <= 0:
        return img.copy()
    k = np.fft.fftshift(np.fft.fftn(img))
    n = img.shape[axis]
    n_lines = int(level * 0.4 * n)
    lines = rng.choice(n, size=max(1, n_lines), replace=False)
    grid = [np.fft.fftshift(np.fft.fftfreq(s)) for s in img.shape]
    mesh = np.meshgrid(*grid, indexing="ij")
    for ln in lines:
        shift = rng.uniform(-4, 4, 3) * level
        phase = np.exp(-2j * np.pi * sum(m * s for m, s in zip(mesh, shift)))
        sl = [slice(None)] * 3
        sl[axis] = ln
        k[tuple(sl)] = k[tuple(sl)] * phase[tuple(sl)]
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(k)))
    return out.astype(np.float32)


def bias_field(img: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative bias field with amplitude ±level×50% built from a random low-order polynomial."""
    if level <= 0:
        return img.copy()
    coords = [np.linspace(-1, 1, s) for s in img.shape]
    z, y, x = np.meshgrid(*coords, indexing="ij")
    coef = rng.normal(0, 1, 9)
    field = (coef[0] * z + coef[1] * y + coef[2] * x + coef[3] * z * y + coef[4] * z * x + coef[5] * y * x
             + coef[6] * z ** 2 + coef[7] * y ** 2 + coef[8] * x ** 2)
    field = field / (np.abs(field).max() + 1e-9)
    return (img * (1 + 0.5 * level * field)).astype(np.float32)


def downsample_throughplane(img: np.ndarray, level: float, axis: int = 0) -> np.ndarray:
    """Simulate thick slices: block-average along ``axis`` by factor 1 + round(level×4), then resample back."""
    f = 1 + int(round(level * 4))
    if f <= 1:
        return img.copy()
    zoom = [1.0] * 3
    zoom[axis] = 1.0 / f
    low = ndimage.zoom(img, zoom, order=1)
    back = ndimage.zoom(low, [img.shape[i] / low.shape[i] for i in range(3)], order=1)
    return back.astype(np.float32)


def ghosting(img: np.ndarray, level: float, axis: int = 1, n_ghosts: int = 2) -> np.ndarray:
    """Nyquist-like ghosts: add shifted copies along ``axis`` with amplitude level×30%."""
    if level <= 0:
        return img.copy()
    out = img.copy().astype(np.float32)
    n = img.shape[axis]
    for g in range(1, n_ghosts + 1):
        out += 0.3 * level / g * np.roll(img, n * g // (n_ghosts + 1), axis=axis)
    return out


DEGRADATIONS = {
    "noise": lambda img, lvl, rng: rician_noise(img, lvl, rng),
    "motion": lambda img, lvl, rng: kspace_motion(img, lvl, rng),
    "bias": lambda img, lvl, rng: bias_field(img, lvl, rng),
    "downsample": lambda img, lvl, rng: downsample_throughplane(img, lvl),
    "ghosting": lambda img, lvl, rng: ghosting(img, lvl),
}


def apply(img: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    """Apply a named degradation at ``level`` in [0, 1]."""
    if kind not in DEGRADATIONS:
        raise KeyError(f"unknown degradation {kind!r}; choose from {sorted(DEGRADATIONS)}")
    return DEGRADATIONS[kind](np.asarray(img, np.float32), float(level), np.random.default_rng(seed))


def dose_levels(n: int = 5) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)
