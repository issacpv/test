"""Spikes -> calcium fluorescence forward model and AR(1) deconvolution.

Kernel time constants are approximate single-action-potential values for GCaMP6 (Chen et al., 2013, Nature:
GCaMP6f half-decay ~140 ms, rise ~45 ms; GCaMP6s half-decay ~550 ms, rise ~180 ms). Supralinear burst
amplification and saturation are modelled phenomenologically: F = A * (c ** gamma) / (1 + c ** gamma / c_sat).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.signal import fftconvolve

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Indicator:
    name: str
    tau_rise_s: float
    tau_decay_s: float          # e-fold decay (= half-decay / ln 2)
    amplitude: float = 0.1      # dF/F per spike in the linear regime
    gamma: float = 1.0          # supralinearity exponent (>1 amplifies bursts)
    c_sat: float = np.inf       # saturation level of the calcium proxy (inf = none)


INDICATORS: Dict[str, Indicator] = {
    "GCaMP6f": Indicator("GCaMP6f", tau_rise_s=0.045, tau_decay_s=0.142 / LN2, amplitude=0.08),
    "GCaMP6s": Indicator("GCaMP6s", tau_rise_s=0.18, tau_decay_s=0.55 / LN2, amplitude=0.2),
}


def calcium_kernel(fs: float, tau_rise_s: float, tau_decay_s: float, duration_s: Optional[float] = None) -> np.ndarray:
    """Difference-of-exponentials kernel normalised to unit peak."""
    duration_s = 6.0 * tau_decay_s if duration_s is None else duration_s
    t = np.arange(0, duration_s, 1.0 / fs)
    k = np.exp(-t / tau_decay_s) - np.exp(-t / tau_rise_s)
    return k / (k.max() + 1e-12)


def bin_spikes(spike_times: np.ndarray, fs: float, duration_s: float) -> np.ndarray:
    n = int(np.ceil(duration_s * fs))
    idx = np.floor(np.asarray(spike_times, dtype=float) * fs).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    return np.bincount(idx, minlength=n).astype(float)


def spikes_to_fluorescence(spike_times: np.ndarray, fs: float, duration_s: float, indicator: Indicator = INDICATORS["GCaMP6f"],
                           noise_sd: float = 0.02, rng: Optional[np.random.Generator] = None,
                           gamma: Optional[float] = None, c_sat: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """dF/F trace (and time axis) from spike times through kernel, nonlinearity and additive noise."""
    rng = np.random.default_rng(0) if rng is None else rng
    g = indicator.gamma if gamma is None else gamma
    cs = indicator.c_sat if c_sat is None else c_sat
    s = bin_spikes(spike_times, fs, duration_s)
    k = calcium_kernel(fs, indicator.tau_rise_s, indicator.tau_decay_s)
    c = fftconvolve(s, k, mode="full")[: len(s)]
    c = np.clip(c, 0, None)
    f = indicator.amplitude * (c ** g) / (1.0 + (c ** g) / cs)
    f = f + noise_sd * rng.normal(size=len(f))
    t = np.arange(len(f)) / fs
    return f, t


def event_responses(trace: np.ndarray, fs: float, onsets_s: np.ndarray, window: Tuple[float, float] = (0.0, 0.5),
                    baseline: Optional[Tuple[float, float]] = (-0.25, 0.0), stat: str = "mean") -> np.ndarray:
    """Per-trial response from a continuous trace: mean (or max) in window minus mean baseline."""
    out = np.empty(len(onsets_s))
    for i, on in enumerate(onsets_s):
        a, b = int(round((on + window[0]) * fs)), int(round((on + window[1]) * fs))
        seg = trace[max(a, 0):max(b, 0)]
        val = seg.mean() if stat == "mean" else seg.max()
        if baseline is not None:
            a0, b0 = int(round((on + baseline[0]) * fs)), int(round((on + baseline[1]) * fs))
            val -= trace[max(a0, 0):max(b0, 0)].mean() if b0 > a0 and b0 > 0 else 0.0
        out[i] = val
    return out


def ar1_gamma(fs: float, tau_decay_s: float) -> float:
    """AR(1) coefficient for a given decay time constant and sampling rate."""
    return float(np.exp(-1.0 / (fs * tau_decay_s)))


def deconvolve_ar1(trace: np.ndarray, gamma: float, lam: float = 0.0, chunk: int = 2000) -> np.ndarray:
    """Non-negative AR(1) deconvolution by NNLS on a lower-triangular AR kernel (a FOOPSI-style baseline).

    Solves min ||y - K s||^2 + lam * sum(s), s >= 0 with K[i, j] = gamma^(i-j) for i >= j. The trace is
    processed in overlapping chunks to keep the dense kernel small; for production use OASIS (Friedrich, Zhou &
    Paninski, 2017), which solves the same problem online."""
    y = np.asarray(trace, dtype=float)
    n = len(y)
    s = np.zeros(n)
    overlap = min(chunk // 4, n)
    start = 0
    while start < n:
        end = min(start + chunk, n)
        m = end - start
        i, j = np.indices((m, m))
        K = np.where(i >= j, gamma ** np.clip(i - j, 0, None), 0.0)
        if lam > 0:
            K = np.vstack([K, np.sqrt(lam) * np.ones((1, m))])
            yy = np.append(y[start:end], 0.0)
        else:
            yy = y[start:end]
        sol, _ = nnls(K, yy)
        keep_from = 0 if start == 0 else overlap // 2
        s[start + keep_from:end] = sol[keep_from:]
        if end == n:
            break
        start = end - overlap
    return s


def fit_forward_parameters(metric_fn, spike_trains, fs: float, duration_s: float, target_value: float,
                           indicator: Indicator, gammas=(1.0, 1.25, 1.5, 2.0), c_sats=(np.inf, 5.0, 2.0),
                           rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Grid search of (gamma, c_sat) so that ``metric_fn`` of the forward-modelled population matches a target
    value (e.g., the 2P median lifetime sparseness). ``metric_fn(list_of_traces, fs) -> float``."""
    rng = np.random.default_rng(0) if rng is None else rng
    best = None
    for g in gammas:
        for cs in c_sats:
            traces = [spikes_to_fluorescence(st, fs, duration_s, indicator, rng=rng, gamma=g, c_sat=cs)[0]
                      for st in spike_trains]
            val = float(metric_fn(traces, fs))
            err = abs(val - target_value)
            if best is None or err < best["error"]:
                best = {"gamma": g, "c_sat": cs, "value": val, "error": err}
    return best
