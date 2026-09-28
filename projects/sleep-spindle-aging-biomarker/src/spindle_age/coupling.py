"""SO-spindle coupling metrics: event-based phase coupling, PAC and event-locked power.

Phase convention: the SO phase is the Hilbert phase of the 0.3-1.5 Hz signal,
so 0 rad = SO positive peak (up-state), +/-pi = SO trough (down-state).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import norm

from .detect import _bandpass
from .io_edf import N2, N3


# --------------------------------------------------------------------------- #
# circular statistics
# --------------------------------------------------------------------------- #
def mean_resultant_length(phases: np.ndarray) -> float:
    p = np.asarray(phases, float)
    return float(np.abs(np.mean(np.exp(1j * p)))) if p.size else np.nan


def circular_mean(phases: np.ndarray) -> float:
    p = np.asarray(phases, float)
    return float(np.angle(np.mean(np.exp(1j * p)))) if p.size else np.nan


def rayleigh_test(phases: np.ndarray) -> Tuple[float, float]:
    """Rayleigh z and p-value for non-uniformity (Zar's approximation)."""
    p = np.asarray(phases, float)
    n = p.size
    if n == 0:
        return np.nan, np.nan
    R = mean_resultant_length(p) * n
    z = R ** 2 / n
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return float(z), float(min(max(pval, 0.0), 1.0))


def modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
    """Tort et al. (2010) modulation index between a phase series and an amplitude series."""
    phase = np.asarray(phase, float); amplitude = np.asarray(amplitude, float)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    mean_amp = np.array([amplitude[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(n_bins)])
    if mean_amp.sum() <= 0:
        return np.nan
    p = mean_amp / mean_amp.sum()
    p = np.clip(p, 1e-12, None)
    H = -np.sum(p * np.log(p))
    return float((np.log(n_bins) - H) / np.log(n_bins))


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def so_phase_signal(x: np.ndarray, fs: float, band: Tuple[float, float] = (0.3, 1.5)) -> np.ndarray:
    return np.angle(hilbert(_bandpass(np.asarray(x, float), fs, *band)))


def sigma_envelope(x: np.ndarray, fs: float, band: Tuple[float, float] = (11.0, 16.0)) -> np.ndarray:
    return np.abs(hilbert(_bandpass(np.asarray(x, float), fs, *band)))


# --------------------------------------------------------------------------- #
# event-based coupling
# --------------------------------------------------------------------------- #
def coupled_spindles(spindles: pd.DataFrame, sos: pd.DataFrame, so_phase: np.ndarray, fs: float,
                     window: Tuple[float, float] = (-1.2, 1.2), ref: str = "NegPeak") -> pd.DataFrame:
    """Spindles whose peak lies within ``window`` (s) of an SO reference point.

    Adds columns ``so_index``, ``lag`` (spindle peak - SO reference, s) and ``phase``
    (SO phase at spindle peak, rad). Each spindle is matched to the nearest SO.
    """
    if len(spindles) == 0 or len(sos) == 0:
        return spindles.iloc[0:0].assign(so_index=pd.Series(dtype=int), lag=pd.Series(dtype=float), phase=pd.Series(dtype=float))
    so_ref = np.sort(sos[ref].to_numpy(float))
    order = np.argsort(sos[ref].to_numpy(float))
    peaks = spindles["Peak"].to_numpy(float)
    j = np.searchsorted(so_ref, peaks)
    j0 = np.clip(j - 1, 0, len(so_ref) - 1); j1 = np.clip(j, 0, len(so_ref) - 1)
    d0 = peaks - so_ref[j0]; d1 = peaks - so_ref[j1]
    use1 = np.abs(d1) < np.abs(d0)
    lag = np.where(use1, d1, d0)
    near = np.where(use1, order[j1], order[j0])
    keep = (lag >= window[0]) & (lag <= window[1])
    idx = np.clip((peaks[keep] * fs).round().astype(int), 0, len(so_phase) - 1)
    out = spindles.loc[keep].copy()
    out["so_index"] = near[keep]
    out["lag"] = lag[keep]
    out["phase"] = so_phase[idx]
    return out.reset_index(drop=True)


def so_windows_mask(sos: pd.DataFrame, n: int, fs: float, window: Tuple[float, float] = (-1.2, 1.2),
                    ref: str = "NegPeak") -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for t in sos[ref].to_numpy(float):
        a = max(int((t + window[0]) * fs), 0); b = min(int((t + window[1]) * fs), n)
        m[a:b] = True
    return m


def surrogate_mrl(so_phase: np.ndarray, candidate_mask: np.ndarray, n_events: int, n_surr: int = 200,
                  seed: int = 0) -> np.ndarray:
    """Null MRL distribution: ``n_events`` phases sampled at random positions inside SO windows."""
    rng = np.random.default_rng(seed)
    cand = np.where(candidate_mask)[0]
    if cand.size == 0 or n_events == 0:
        return np.full(n_surr, np.nan)
    out = np.empty(n_surr)
    for i in range(n_surr):
        out[i] = mean_resultant_length(so_phase[rng.choice(cand, n_events, replace=True)])
    return out


def event_locked_sigma_power(x: np.ndarray, fs: float, event_times: Sequence[float],
                             window: Tuple[float, float] = (-2.0, 2.0), band: Tuple[float, float] = (11.0, 16.0),
                             baseline: Optional[Tuple[float, float]] = (-2.0, -1.5)) -> Tuple[np.ndarray, np.ndarray]:
    """Mean sigma envelope (optionally baseline-normalized, %) locked to events; returns ``(t, curve)``."""
    env = sigma_envelope(x, fs, band)
    a, b = int(window[0] * fs), int(window[1] * fs)
    t = np.arange(a, b) / fs
    segs = []
    for et in event_times:
        c = int(round(et * fs))
        if c + a >= 0 and c + b <= len(env):
            segs.append(env[c + a: c + b])
    if not segs:
        return t, np.full(len(t), np.nan)
    curve = np.mean(segs, axis=0)
    if baseline is not None:
        bm = (t >= baseline[0]) & (t < baseline[1])
        base = curve[bm].mean() if bm.any() else np.nan
        curve = 100.0 * (curve - base) / base
    return t, curve


def coupling_metrics(x: np.ndarray, fs: float, hypno_samples: np.ndarray, spindles: pd.DataFrame, sos: pd.DataFrame,
                     window: Tuple[float, float] = (-1.2, 1.2), n_surr: int = 200, seed: int = 0,
                     nrem_stages: Sequence[int] = (N2, N3)) -> Dict[str, float]:
    """Per-night coupling summary.

    Keys: ``n_spindles, n_sos, n_coupled, coupled_fraction, so_rate_per_min, preferred_phase,
    mrl, rayleigh_z, rayleigh_p, mrl_null_mean, mrl_z, mi_pac, sigma_upstate_pct``.
    """
    x = np.asarray(x, float)
    n = len(x)
    ph = so_phase_signal(x, fs)
    nrem = np.isin(np.asarray(hypno_samples)[:n], list(nrem_stages))
    cs = coupled_spindles(spindles, sos, ph, fs, window)
    phases = cs["phase"].to_numpy(float) if len(cs) else np.empty(0)
    mrl = mean_resultant_length(phases)
    z, p = rayleigh_test(phases)
    cand = so_windows_mask(sos, n, fs, window) & nrem
    null = surrogate_mrl(ph, cand, len(phases), n_surr, seed)
    mrl_z = (mrl - np.nanmean(null)) / np.nanstd(null) if np.isfinite(null).any() and np.nanstd(null) > 0 else np.nan
    env = sigma_envelope(x, fs)
    mi = modulation_index(ph[nrem], env[nrem]) if nrem.sum() > fs * 60 else np.nan
    t, curve = event_locked_sigma_power(x, fs, sos["NegPeak"].to_numpy(float) if len(sos) else [])
    up = curve[(t > 0.2) & (t < 0.8)].mean() if np.isfinite(curve).any() else np.nan  # up-state follows the trough
    nrem_min = nrem.sum() / fs / 60.0
    return {
        "n_spindles": int(len(spindles)), "n_sos": int(len(sos)), "n_coupled": int(len(cs)),
        "coupled_fraction": float(len(cs) / len(spindles)) if len(spindles) else np.nan,
        "so_rate_per_min": float(len(sos) / nrem_min) if nrem_min > 0 else np.nan,
        "preferred_phase": circular_mean(phases), "mrl": mrl, "rayleigh_z": z, "rayleigh_p": p,
        "mrl_null_mean": float(np.nanmean(null)) if np.isfinite(null).any() else np.nan, "mrl_z": float(mrl_z),
        "mi_pac": mi, "sigma_upstate_pct": float(up),
    }


def phase_to_zscore(mrl: float, n: int) -> float:
    """Approximate z for an MRL given n events under uniformity (Rayleigh normal approximation)."""
    if not np.isfinite(mrl) or n <= 0:
        return np.nan
    _, p = rayleigh_test(np.zeros(0)) if n == 0 else (None, None)
    z = mrl * np.sqrt(n)
    return float(norm.isf(np.exp(-z ** 2)) if z > 0 else 0.0)
