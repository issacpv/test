"""Resting-state HRF estimation by point-process deconvolution.

Follows the logic of Wu et al. (2013, Medical Image Analysis) and the rsHRF toolbox
(Wu et al., 2021, NeuroImage): BOLD "pseudo-events" are detected as supra-threshold
excursions; an FIR response is fitted for a grid of onset delays; the delay with the best
fit gives the HRF, from which height, time-to-peak and FWHM are extracted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class HRFParams:
    height: float          # peak amplitude (arbitrary units of the input series)
    time_to_peak: float    # seconds
    fwhm: float            # seconds
    undershoot: float      # minimum after the peak (negative = undershoot)
    onset_delay: float     # best onset delay (s) between pseudo-event and response start
    r2: float              # variance explained by the event model


def canonical_hrf(tr: float, duration: float = 32.0, peak_delay: float = 6.0, undershoot_delay: float = 16.0,
                  dispersion: float = 1.0, undershoot_dispersion: float = 1.0, ratio: float = 6.0) -> np.ndarray:
    """SPM-style double-gamma HRF sampled at ``tr`` and scaled to unit peak."""
    t = np.arange(0, duration, tr)
    g1 = sps.gamma.pdf(t, peak_delay / dispersion, scale=dispersion)
    g2 = sps.gamma.pdf(t, undershoot_delay / undershoot_dispersion, scale=undershoot_dispersion)
    h = g1 - g2 / ratio
    return h / h.max()


def detect_pseudo_events(ts: np.ndarray, threshold: float = 1.0, min_separation: int = 2) -> np.ndarray:
    """Indices where the z-scored series crosses ``threshold`` upward (local maxima above it).

    ``min_separation`` (samples) suppresses events closer than that to a preceding event.
    """
    z = (ts - ts.mean()) / (ts.std() + 1e-12)
    cand = np.flatnonzero((z[1:-1] > threshold) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])) + 1
    events: list[int] = []
    for c in cand:
        if not events or c - events[-1] >= min_separation:
            events.append(int(c))
    return np.asarray(events, dtype=int)


def _fir_design(events: np.ndarray, n: int, length: int, delay: int) -> np.ndarray:
    X = np.zeros((n, length))
    for e in events:
        start = e - delay
        for k in range(length):
            t = start + k
            if 0 <= t < n:
                X[t, k] += 1.0
    return X


def estimate_hrf(ts: np.ndarray, tr: float, hrf_length: float = 24.0, max_onset_delay: float = 8.0,
                 threshold: float = 1.0, min_events: int = 5) -> tuple[np.ndarray, HRFParams]:
    """Point-process FIR estimate of the HRF for one series.

    Returns ``(hrf, params)``; ``hrf`` is sampled at ``tr`` starting at the event onset.
    For each candidate onset delay ``d`` (0..max_onset_delay s) the event train is shifted
    back by ``d`` and an FIR of ``hrf_length`` s is fitted by least squares; the delay with
    the highest R² wins, which makes the estimate robust to the fact that the BOLD peak, not
    the neural onset, triggered the pseudo-event.
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts - ts.mean()
    n = ts.size
    events = detect_pseudo_events(ts, threshold=threshold)
    if events.size < min_events:
        raise ValueError(f"only {events.size} pseudo-events detected; need >= {min_events}")
    L = int(round(hrf_length / tr))
    best = (-np.inf, None, 0)
    for d in range(0, int(round(max_onset_delay / tr)) + 1):
        X = _fir_design(events, n, L, d)
        beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
        resid = ts - X @ beta
        r2 = 1 - resid.var() / ts.var()
        if r2 > best[0]:
            best = (r2, beta, d)
    r2, hrf, d = best
    params = hrf_parameters(hrf, tr)
    params.onset_delay = d * tr
    params.r2 = float(r2)
    return hrf, params


def hrf_parameters(hrf: np.ndarray, tr: float) -> HRFParams:
    """Height, time-to-peak, FWHM (linear interpolation at half height) and undershoot."""
    hrf = np.asarray(hrf, dtype=float)
    k = int(np.argmax(hrf))
    height = float(hrf[k])
    ttp = k * tr
    half = height / 2.0
    # left crossing
    left = k
    while left > 0 and hrf[left] > half:
        left -= 1
    if hrf[left] <= half and left < k:
        t_left = left + (half - hrf[left]) / (hrf[left + 1] - hrf[left])
    else:
        t_left = float(left)
    right = k
    while right < hrf.size - 1 and hrf[right] > half:
        right += 1
    if hrf[right] <= half and right > k:
        t_right = right - (half - hrf[right]) / (hrf[right - 1] - hrf[right])
    else:
        t_right = float(right)
    fwhm = (t_right - t_left) * tr
    undershoot = float(hrf[k:].min()) if k < hrf.size else 0.0
    return HRFParams(height=height, time_to_peak=float(ttp), fwhm=float(fwhm), undershoot=undershoot,
                     onset_delay=0.0, r2=float("nan"))


def wiener_deconvolve(ts: np.ndarray, hrf: np.ndarray, noise_ratio: float = 0.05) -> np.ndarray:
    """Frequency-domain Wiener deconvolution of ``ts`` by ``hrf`` (same sampling)."""
    ts = np.asarray(ts, dtype=float)
    n = ts.size
    H = np.fft.rfft(hrf, n=n)
    Y = np.fft.rfft(ts - ts.mean(), n=n)
    G = np.conj(H) / (np.abs(H) ** 2 + noise_ratio * np.max(np.abs(H) ** 2))
    return np.fft.irfft(Y * G, n=n)


def simulate_bold(n: int, tr: float, hrf: np.ndarray, event_rate: float = 0.08, noise_sd: float = 0.3,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic BOLD = sparse neural events ⊗ ``hrf`` + white noise. Returns (bold, events)."""
    rng = np.random.default_rng() if rng is None else rng
    events = (rng.random(n) < event_rate).astype(float)
    bold = np.convolve(events, hrf)[:n] + noise_sd * rng.standard_normal(n)
    return bold, events
