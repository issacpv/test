"""Current-source density, L4 sink anchoring and depth -> layer assignment for laminar probes.

Depths are in micrometres below the cortical surface (positive = deeper). Layer boundaries are expressed
relative to the centre of the flash-evoked L4 current sink so that probe-angle and template errors do not
propagate; mouse V1 thickness priors are defaults and can be replaced per area.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

# approximate mouse V1 layer thicknesses (um); L4 is centred on the CSD sink
MOUSE_V1_THICKNESS_UM: Dict[str, float] = {"L1": 100.0, "L2/3": 200.0, "L4": 150.0, "L5": 250.0, "L6": 300.0}
LAYER_ORDER: Sequence[str] = ("L1", "L2/3", "L4", "L5", "L6")


def compute_csd(lfp: np.ndarray, spacing_um: float, sigma_s_per_m: float = 0.3, smooth_channels: float = 0.0
                ) -> np.ndarray:
    """Standard (second spatial derivative) CSD along the channel axis.

    ``lfp`` is (n_time, n_channels) ordered from superficial to deep with uniform ``spacing_um``. Returns
    (n_time, n_channels - 2) in A/m^3 units up to the conductivity constant (sinks negative). Optional Gaussian
    smoothing across channels (in units of channels) before differentiation reduces high-spatial-frequency noise.
    """
    X = np.asarray(lfp, dtype=float)
    if smooth_channels > 0:
        X = gaussian_filter1d(X, smooth_channels, axis=1, mode="nearest")
    h = spacing_um * 1e-6
    d2 = (X[:, :-2] - 2 * X[:, 1:-1] + X[:, 2:]) / (h * h)
    return -sigma_s_per_m * d2


def evoked_average(x: np.ndarray, fs: float, onsets_s: np.ndarray, window: Tuple[float, float] = (-0.05, 0.25),
                   baseline: Optional[Tuple[float, float]] = (-0.05, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Trial-averaged (n_win, n_channels) response around onsets, baseline-subtracted per channel."""
    i0 = int(round(window[0] * fs))
    i1 = int(round(window[1] * fs))
    t = np.arange(i0, i1) / fs
    segs = []
    for on in onsets_s:
        c = int(round(on * fs))
        if c + i0 < 0 or c + i1 > len(x):
            continue
        segs.append(x[c + i0:c + i1])
    if not segs:
        raise ValueError("no complete trials in window")
    M = np.mean(segs, axis=0)
    if baseline is not None:
        b = (t >= baseline[0]) & (t < baseline[1])
        M = M - M[b].mean(axis=0, keepdims=True)
    return M, t


def find_l4_sink(csd: np.ndarray, t: np.ndarray, depths_um: np.ndarray, window: Tuple[float, float] = (0.02, 0.08),
                 min_depth_um: float = 100.0, max_depth_um: float = 700.0) -> Dict[str, float]:
    """Locate the earliest strong current sink after stimulus onset: the granular (L4) landmark.

    Within ``window`` and a plausible depth range, the sink is the most negative CSD value; its depth is
    refined as the CSD-weighted centroid of contiguous negative channels around it at the time of the minimum.
    """
    tsel = (t >= window[0]) & (t < window[1])
    dsel = (depths_um >= min_depth_um) & (depths_um <= max_depth_um)
    if not tsel.any() or not dsel.any():
        return {"depth_um": np.nan, "latency_s": np.nan, "magnitude": np.nan}
    sub = csd[np.ix_(tsel, dsel)]
    it, ic = np.unravel_index(np.argmin(sub), sub.shape)
    t_idx = np.flatnonzero(tsel)[it]
    d_idx = np.flatnonzero(dsel)[ic]
    profile = csd[t_idx]
    # extend across contiguous negative channels
    lo = d_idx
    while lo - 1 >= 0 and profile[lo - 1] < 0:
        lo -= 1
    hi = d_idx
    while hi + 1 < len(profile) and profile[hi + 1] < 0:
        hi += 1
    w = -profile[lo:hi + 1]
    centroid = float(np.sum(w * depths_um[lo:hi + 1]) / np.sum(w))
    return {"depth_um": centroid, "latency_s": float(t[t_idx]), "magnitude": float(profile[d_idx]),
            "sink_extent_um": float(depths_um[hi] - depths_um[lo])}


def layer_boundaries(l4_center_um: float, thickness: Dict[str, float] = MOUSE_V1_THICKNESS_UM) -> Dict[str, Tuple[float, float]]:
    """Layer depth intervals (um below surface) anchored at the L4 centre."""
    l4_top = l4_center_um - thickness["L4"] / 2
    l4_bot = l4_center_um + thickness["L4"] / 2
    l23_top = l4_top - thickness["L2/3"]
    l1_top = l23_top - thickness["L1"]
    l5_bot = l4_bot + thickness["L5"]
    l6_bot = l5_bot + thickness["L6"]
    return {"L1": (l1_top, l23_top), "L2/3": (l23_top, l4_top), "L4": (l4_top, l4_bot),
            "L5": (l4_bot, l5_bot), "L6": (l5_bot, l6_bot)}


def assign_layers(depths_um: np.ndarray, l4_center_um: float, thickness: Dict[str, float] = MOUSE_V1_THICKNESS_UM,
                  jitter_um: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Map unit depths to layer labels ('L1', 'L2/3', 'L4', 'L5', 'L6', or 'outside').

    ``jitter_um`` shifts every boundary by an independent N(0, jitter) draw (sensitivity analysis H6)."""
    b = layer_boundaries(l4_center_um, thickness)
    if jitter_um > 0:
        rng = np.random.default_rng(0) if rng is None else rng
        edges = [b["L1"][0], b["L2/3"][0], b["L4"][0], b["L5"][0], b["L6"][0], b["L6"][1]]
        edges = np.sort(np.asarray(edges) + rng.normal(0, jitter_um, len(edges)))
        names = list(LAYER_ORDER)
        b = {n: (edges[i], edges[i + 1]) for i, n in enumerate(names)}
    d = np.asarray(depths_um, dtype=float)
    out = np.full(len(d), "outside", dtype=object)
    for name in LAYER_ORDER:
        lo, hi = b[name]
        out[(d >= lo) & (d < hi)] = name
    return out


def layer_counts(labels: np.ndarray) -> Dict[str, int]:
    return {name: int(np.sum(np.asarray(labels) == name)) for name in LAYER_ORDER}


def synthetic_laminar_lfp(fs: float, n_channels: int, spacing_um: float, onsets_s: np.ndarray, duration_s: float,
                          sink_depth_um: float, rng: np.random.Generator, sink_latency_s: float = 0.04,
                          noise: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """LFP with a flash-evoked dipole whose sink sits at ``sink_depth_um`` (for tests).

    Channel 0 is the surface (depth 0). The evoked potential is a Gaussian in depth (sink) flanked by sources,
    integrated twice in depth so that the second derivative reproduces the sink; here we directly build the
    potential as a smooth profile whose curvature is maximal at the sink depth."""
    depths = np.arange(n_channels) * spacing_um
    n = int(duration_s * fs)
    x = rng.normal(0, noise, size=(n, n_channels))
    t = np.arange(n) / fs
    # potential profile: negative Gaussian (its second derivative is negative at the centre -> sink)
    prof = -np.exp(-((depths - sink_depth_um) ** 2) / (2 * (1.5 * spacing_um) ** 2))
    for on in onsets_s:
        kern = 30.0 * np.exp(-((t - on - sink_latency_s) ** 2) / (2 * 0.008 ** 2))
        x += np.outer(kern, prof)
    return x, depths
