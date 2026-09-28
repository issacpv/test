"""Controlled vendor/acquisition counterfactuals on echo frames.

These transforms let us render a source frame as if it had been acquired with a
different vendor's intensity mapping, sector geometry or frame rate, so the
cross-vendor view-classification drop can be decomposed into acquisition
components. They also serve as training-time augmentations.

Frames are float arrays in [0, 1], shape (H, W). No heavy image library is
required for the core transforms; scikit-image is optional.
"""
from __future__ import annotations

import numpy as np


def histogram_match(source: np.ndarray, reference: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Match the intensity CDF of ``source`` to that of ``reference`` (1-channel).

    A self-contained CDF-matching implementation (no scikit-image dependency).
    """
    src = np.asarray(source, float).ravel()
    ref = np.asarray(reference, float).ravel()
    s_vals, s_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
    r_vals, r_counts = np.unique(ref, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(float) / src.size
    r_cdf = np.cumsum(r_counts).astype(float) / ref.size
    interp = np.interp(s_cdf, r_cdf, r_vals)
    return interp[s_idx].reshape(np.asarray(source).shape)


def gamma_adjust(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Apply a gamma/contrast change (vendor post-processing surrogate)."""
    f = np.clip(np.asarray(frame, float), 0, 1)
    return f ** gamma


def sector_mask(frame: np.ndarray, apex=(0.5, 0.0), half_angle_deg: float = 37.0,
                r_inner: float = 0.05, r_outer: float = 0.98) -> np.ndarray:
    """Apply a fan/sector mask (approximate scan-conversion geometry).

    Coordinates are normalised to [0,1]; apex at top-center by default. Pixels
    outside the sector are set to zero, mimicking a different sector width.
    """
    f = np.asarray(frame, float)
    h, w = f.shape
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    ax, ay = apex
    dx = xs - ax
    dy = ys - ay
    r = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.abs(np.degrees(np.arctan2(dx, dy + 1e-9)))  # angle from vertical
    mask = (ang <= half_angle_deg) & (r >= r_inner) & (r <= r_outer)
    return f * mask


def resample_frame_rate(clip: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Temporally resample a clip (T, H, W) from src_fps to dst_fps by index interp."""
    clip = np.asarray(clip, float)
    T = clip.shape[0]
    dur = T / src_fps
    new_T = max(1, int(round(dur * dst_fps)))
    idx = np.linspace(0, T - 1, new_T)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (idx - lo)[:, None, None]
    return (1 - frac) * clip[lo] + frac * clip[hi]


# Registry so the decomposition can toggle operations by name.
def make_transform(name: str, **kwargs):
    """Return a frame->frame transform by name for the Shapley decomposition."""
    if name == "intensity":
        ref = kwargs["reference"]
        return lambda f: histogram_match(f, ref)
    if name == "gamma":
        g = kwargs.get("gamma", 1.5)
        return lambda f: gamma_adjust(f, g)
    if name == "geometry":
        ha = kwargs.get("half_angle_deg", 30.0)
        return lambda f: sector_mask(f, half_angle_deg=ha)
    raise KeyError(f"unknown transform '{name}'")
