"""SO-spindle coupling: event-locked phases, circular statistics, surrogates and Tort's modulation index.

Phase convention: Hilbert phase of the SO-band signal, so that a positive peak has phase 0 and a trough has
phase +/- pi. With the 'negative_first' SO polarity used by the detectors, the up-state (surface-positive /
depth-negative) corresponds to phase ~ 0.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .detect import bandpass


def so_phase(x: np.ndarray, fs: float, band=(0.3, 1.5), invert: bool = False) -> np.ndarray:
    """Instantaneous SO phase (radians, -pi..pi) from the band-passed signal."""
    y = bandpass(x, fs, *band)
    if invert:
        y = -y
    return np.angle(hilbert(y))


def phases_at_times(times_s: np.ndarray, phase: np.ndarray, fs: float) -> np.ndarray:
    idx = np.clip(np.round(np.asarray(times_s, dtype=float) * fs).astype(int), 0, len(phase) - 1)
    return phase[idx]


def couple_spindles_to_so(spindles: pd.DataFrame, so: pd.DataFrame, phase: np.ndarray, fs: float,
                          window_cycles: float = 1.0) -> pd.DataFrame:
    """Pair each spindle peak with the nearest SO trough within ``window_cycles`` SO periods.

    Returns one row per coupled spindle: spindle index, SO index, phase at spindle peak, offset from the SO
    trough in seconds and in SO cycles (offset / SO duration).
    """
    if len(spindles) == 0 or len(so) == 0:
        return pd.DataFrame(columns=["spindle", "so", "phase", "offset_s", "offset_cycles"])
    troughs = so["trough_s"].to_numpy()
    durs = so["duration_s"].to_numpy()
    order = np.argsort(troughs)
    troughs_sorted = troughs[order]
    rows = []
    for i, t in enumerate(spindles["peak_s"].to_numpy()):
        j = np.searchsorted(troughs_sorted, t)
        cand = [k for k in (j - 1, j) if 0 <= k < len(troughs_sorted)]
        if not cand:
            continue
        k = min(cand, key=lambda kk: abs(troughs_sorted[kk] - t))
        so_idx = int(order[k])
        dt = t - troughs[so_idx]
        if abs(dt) <= window_cycles * durs[so_idx]:
            rows.append({"spindle": i, "so": so_idx, "phase": float(phases_at_times([t], phase, fs)[0]),
                         "offset_s": float(dt), "offset_cycles": float(dt / durs[so_idx])})
    return pd.DataFrame(rows, columns=["spindle", "so", "phase", "offset_s", "offset_cycles"])


def circular_stats(phases: np.ndarray) -> Dict[str, float]:
    """Mean direction, mean resultant length (MVL), Rayleigh z and p (Zar's approximation)."""
    ph = np.asarray(phases, dtype=float)
    n = len(ph)
    if n == 0:
        return {"n": 0, "mean_phase": np.nan, "mvl": np.nan, "rayleigh_z": np.nan, "rayleigh_p": np.nan}
    c = np.exp(1j * ph).mean()
    mvl = float(np.abs(c))
    Rn = mvl * n
    z = Rn ** 2 / n
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n)))
    return {"n": int(n), "mean_phase": float(np.angle(c)), "mvl": mvl, "rayleigh_z": float(z),
            "rayleigh_p": min(max(p, 0.0), 1.0)}


def circular_distance(a: float, b: float) -> float:
    """Signed angular difference a - b wrapped to (-pi, pi]."""
    return float(np.angle(np.exp(1j * (a - b))))


def surrogate_mvl(phase_pool: np.ndarray, n_events: int, n_perm: int = 500,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Null MVL distribution from ``n_events`` phases drawn at random from ``phase_pool`` (e.g., all NREM
    samples). Matching the event count removes the small-n upward bias of the MVL."""
    rng = np.random.default_rng(0) if rng is None else rng
    pool = np.asarray(phase_pool, dtype=float)
    out = np.empty(n_perm)
    for i in range(n_perm):
        pick = rng.choice(pool, size=n_events, replace=True)
        out[i] = np.abs(np.exp(1j * pick).mean())
    return out


def shifted_surrogate_phases(spindle_peaks_s: np.ndarray, phase: np.ndarray, fs: float, n_perm: int = 200,
                             min_shift_s: float = 5.0, max_shift_s: float = 20.0,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Null MVLs from circularly time-shifted spindle trains (preserves the spindle autocorrelation)."""
    rng = np.random.default_rng(0) if rng is None else rng
    T = len(phase) / fs
    out = np.empty(n_perm)
    for i in range(n_perm):
        shift = rng.uniform(min_shift_s, max_shift_s) * rng.choice([-1, 1])
        t = np.mod(np.asarray(spindle_peaks_s) + shift, T)
        out[i] = np.abs(np.exp(1j * phases_at_times(t, phase, fs)).mean())
    return out


def mvl_zscore(mvl: float, null: np.ndarray) -> float:
    return float((mvl - null.mean()) / (null.std(ddof=1) + 1e-12))


def tort_modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
    """Tort et al. (2010) modulation index: KL divergence of the phase-binned mean amplitude from uniform."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    mean_amp = np.array([amplitude[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(n_bins)])
    p = mean_amp / (mean_amp.sum() + 1e-12)
    p = np.where(p > 0, p, 1e-12)
    return float((np.log(n_bins) + np.sum(p * np.log(p))) / np.log(n_bins))


def coupling_summary(coupled: pd.DataFrame, phase: np.ndarray, fs: float, n_perm: int = 500,
                     rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Circular statistics plus event-count-matched surrogate z-score and scale-free offsets."""
    stats = circular_stats(coupled["phase"].to_numpy()) if len(coupled) else circular_stats(np.array([]))
    if stats["n"] >= 5:
        null = surrogate_mvl(phase, stats["n"], n_perm=n_perm, rng=rng)
        stats["mvl_z"] = mvl_zscore(stats["mvl"], null)
        stats["mvl_null_mean"] = float(null.mean())
    else:
        stats["mvl_z"] = np.nan
        stats["mvl_null_mean"] = np.nan
    stats["offset_cycles_mean"] = float(coupled["offset_cycles"].mean()) if len(coupled) else np.nan
    stats["offset_s_mean"] = float(coupled["offset_s"].mean()) if len(coupled) else np.nan
    return stats
