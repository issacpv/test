"""ECG R-peak detection, HRV and cardiac-phase coupling of microstate transitions."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


def detect_r_peaks(ecg: np.ndarray, sfreq: float, refractory_s: float = 0.25, band: Tuple[float, float] = (5.0, 30.0)
                   ) -> np.ndarray:
    """Pan-Tompkins-style R-peak detector: band-pass, derivative, squaring, moving average, adaptive threshold.

    Returns sample indices of R peaks, refined to the local maximum of the band-passed signal.
    Polarity is handled by flipping the signal if the largest deflections are negative.
    """
    ecg = np.asarray(ecg, dtype=float)
    nyq = sfreq / 2
    b, a = signal.butter(3, [band[0] / nyq, min(band[1], nyq * 0.95) / nyq], btype="band")
    x = signal.filtfilt(b, a, ecg - ecg.mean())
    if np.abs(x.min()) > np.abs(x.max()):
        x = -x
    d = np.gradient(x)
    e = d ** 2
    win = max(1, int(round(0.12 * sfreq)))
    env = np.convolve(e, np.ones(win) / win, mode="same")
    thr = 0.3 * np.percentile(env, 99)
    refr = int(round(refractory_s * sfreq))
    cand, _ = signal.find_peaks(env, height=thr, distance=refr)
    # refine to the maximum of the band-passed ECG within +/- 50 ms
    half = int(round(0.05 * sfreq))
    peaks = []
    for c in cand:
        lo, hi = max(0, c - half), min(len(x), c + half + 1)
        peaks.append(lo + int(np.argmax(x[lo:hi])))
    peaks = np.unique(np.asarray(peaks, dtype=int))
    if len(peaks) > 1:
        keep = np.concatenate([[True], np.diff(peaks) >= refr])
        peaks = peaks[keep]
    return peaks


def rr_intervals(peaks: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """RR intervals in seconds and their time stamps (at the second beat of each pair)."""
    peaks = np.asarray(peaks, dtype=float)
    rr = np.diff(peaks) / sfreq
    t = peaks[1:] / sfreq
    return rr, t


def clean_rr(rr: np.ndarray, t: np.ndarray, max_rel_change: float = 0.2, bounds: Tuple[float, float] = (0.3, 2.0)
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Drop physiologically implausible and ectopic intervals (relative jump from a running median).

    Returns cleaned rr, times, and the fraction of intervals removed.
    """
    rr = np.asarray(rr, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = (rr > bounds[0]) & (rr < bounds[1])
    if len(rr) >= 5:
        med = signal.medfilt(rr, kernel_size=5)
        ok &= np.abs(rr - med) / med <= max_rel_change
    return rr[ok], t[ok], float(1 - ok.mean()) if len(rr) else 0.0


def time_domain_hrv(rr: np.ndarray) -> Dict[str, float]:
    """Mean HR (bpm), SDNN, RMSSD (ms), pNN50 (%)."""
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 3:
        return {"hr_bpm": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan, "pnn50": np.nan, "n_beats": len(rr)}
    d = np.diff(rr)
    return {"hr_bpm": float(60.0 / rr.mean()), "sdnn_ms": float(rr.std(ddof=1) * 1000), "rmssd_ms": float(np.sqrt(np.mean(d ** 2)) * 1000),
            "pnn50": float(np.mean(np.abs(d) > 0.05) * 100), "n_beats": int(len(rr))}


def frequency_domain_hrv(rr: np.ndarray, t: np.ndarray, fs_interp: float = 4.0) -> Dict[str, float]:
    """LF (0.04-0.15 Hz), HF (0.15-0.40 Hz) power (s^2) and LF/HF from a Welch spectrum of the interpolated RR series."""
    rr = np.asarray(rr, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(rr) < 10 or t[-1] - t[0] < 60:
        return {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan}
    grid = np.arange(t[0], t[-1], 1.0 / fs_interp)
    x = np.interp(grid, t, rr)
    x = x - x.mean()
    nper = min(len(x), int(fs_interp * 120))
    f, p = signal.welch(x, fs=fs_interp, nperseg=nper)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    lf = float(trapz(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
    hf = float(trapz(p[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)]))
    return {"lf": lf, "hf": hf, "lf_hf": lf / hf if hf > 0 else np.nan}


def cardiac_phase(n_samples: int, peaks: np.ndarray) -> np.ndarray:
    """Cardiac phase in [0, 1) for every sample (0 at each R peak); nan outside the first/last peak."""
    phase = np.full(n_samples, np.nan)
    for a, b in zip(peaks[:-1], peaks[1:]):
        phase[a:b] = (np.arange(a, b) - a) / (b - a)
    return phase


def cardiac_phase_histogram(labels: np.ndarray, peaks: np.ndarray, n_states: int, n_bins: int = 8,
                            n_perm: int = 500, seed: int = 0) -> Dict[str, np.ndarray]:
    """Distribution of microstate *transition onsets* over the cardiac cycle with a circular-shift null.

    Returns the observed (n_states, n_bins) onset counts, the expected counts under uniform phase,
    a chi-square-type statistic per state and permutation p-values obtained by circularly shifting
    the R-peak train (which preserves the RR structure and the label sequence).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    onsets = np.flatnonzero(np.diff(labels)) + 1
    onset_states = labels[onsets]
    n = len(labels)

    def _hist(pk: np.ndarray) -> np.ndarray:
        ph = cardiac_phase(n, pk)
        h = np.zeros((n_states, n_bins))
        for o, s in zip(onsets, onset_states):
            if np.isfinite(ph[o]):
                h[s, min(int(ph[o] * n_bins), n_bins - 1)] += 1
        return h

    def _stat(h: np.ndarray) -> np.ndarray:
        exp = h.sum(axis=1, keepdims=True) / n_bins
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum(np.where(exp > 0, (h - exp) ** 2 / exp, 0.0), axis=1)

    obs = _hist(peaks)
    stat = _stat(obs)
    null = np.empty((n_perm, n_states))
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        null[i] = _stat(_hist(np.sort((peaks + shift) % n)))
    p = (1 + np.sum(null >= stat, axis=0)) / (n_perm + 1)
    return {"observed": obs, "expected": obs.sum(axis=1, keepdims=True) / n_bins * np.ones((1, n_bins)),
            "statistic": stat, "p_perm": p}


__all__ = ["detect_r_peaks", "rr_intervals", "clean_rr", "time_domain_hrv", "frequency_domain_hrv", "cardiac_phase",
           "cardiac_phase_histogram"]
