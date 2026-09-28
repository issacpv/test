"""Scale-free slow-oscillation and spindle detectors.

Thresholds are percentiles of the recording's own candidate distribution (no microvolt constants), spindle
band edges come from the individualised sigma peak, and duration limits can be given in *cycles* so that the
same rule applies to fast rodent spindles and slower human spindles.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def _zero_crossings(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of positive-to-negative and negative-to-positive zero crossings."""
    s = np.sign(y)
    s[s == 0] = 1
    d = np.diff(s)
    return np.flatnonzero(d < 0) + 1, np.flatnonzero(d > 0) + 1


def detect_slow_oscillations(x: np.ndarray, fs: float, band: Tuple[float, float] = (0.3, 1.5),
                             min_dur: float = 0.5, max_dur: float = 2.5, amp_percentile: float = 75.0,
                             polarity: str = "negative_first") -> pd.DataFrame:
    """Detect SO cycles by zero crossings in the SO band with a percentile peak-to-peak criterion.

    A cycle starts at a positive-to-negative zero crossing, contains a trough (negative half-wave), a
    negative-to-positive crossing, a peak (positive half-wave), and ends at the next positive-to-negative
    crossing. With ``polarity='positive_first'`` the signal is inverted first (use for depth recordings whose
    down-state is positive). Candidates within the duration limits are kept if their peak-to-peak amplitude is
    at or above ``amp_percentile`` of all candidates.

    Returns a DataFrame with columns start_s, trough_s, zero_s, peak_s, end_s, duration_s, ptp, neg_amp, pos_amp.
    """
    y = bandpass(x, fs, *band)
    if polarity == "positive_first":
        y = -y
    p2n, n2p = _zero_crossings(y)
    rows = []
    for i in range(len(p2n) - 1):
        start, end = p2n[i], p2n[i + 1]
        mid = n2p[(n2p > start) & (n2p < end)]
        if len(mid) != 1:
            continue
        mid = int(mid[0])
        dur = (end - start) / fs
        if not (min_dur <= dur <= max_dur):
            continue
        trough = start + int(np.argmin(y[start:mid]))
        peak = mid + int(np.argmax(y[mid:end]))
        rows.append({"start_s": start / fs, "trough_s": trough / fs, "zero_s": mid / fs, "peak_s": peak / fs,
                     "end_s": end / fs, "duration_s": dur, "neg_amp": float(y[trough]), "pos_amp": float(y[peak]),
                     "ptp": float(y[peak] - y[trough])})
    df = pd.DataFrame(rows, columns=["start_s", "trough_s", "zero_s", "peak_s", "end_s", "duration_s",
                                     "neg_amp", "pos_amp", "ptp"])
    if len(df):
        thr = np.percentile(df["ptp"], amp_percentile)
        df = df[df["ptp"] >= thr].reset_index(drop=True)
    df.attrs["band"] = band
    return df


def spindle_envelope(x: np.ndarray, fs: float, center_freq: float, half_bandwidth: float = 2.0,
                     smooth_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Hilbert amplitude envelope (smoothed) and instantaneous frequency of the spindle-band signal."""
    y = bandpass(x, fs, center_freq - half_bandwidth, center_freq + half_bandwidth)
    an = hilbert(y)
    env = np.abs(an)
    k = max(int(round(smooth_s * fs)), 1)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    inst_f = np.diff(np.unwrap(np.angle(an))) * fs / (2 * np.pi)
    inst_f = np.append(inst_f, inst_f[-1])
    return env, inst_f


def _segments(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = np.diff(mask.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    return starts, ends


def detect_spindles(x: np.ndarray, fs: float, center_freq: float, half_bandwidth: float = 2.0,
                    thresh_percentile: float = 95.0, boundary_percentile: float = 80.0,
                    min_cycles: Optional[float] = 5.0, max_cycles: Optional[float] = 30.0,
                    min_dur: Optional[float] = None, max_dur: Optional[float] = None, merge_gap_s: float = 0.1,
                    envelope: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Detect spindles with a two-threshold rule on the envelope of the individualised band.

    A candidate must exceed the *detection* threshold (``thresh_percentile`` of the envelope over the NREM
    signal); its extent is the surrounding segment above the lower *boundary* threshold (``boundary_percentile``),
    as in classic two-threshold spindle detectors. Duration limits are in cycles of ``center_freq`` (scale-free)
    unless ``min_dur``/``max_dur`` in seconds are passed (conventional rule). Pass a pre-computed ``envelope``
    to reuse it across specifications.

    Returns columns start_s, end_s, peak_s, duration_s, peak_amp, freq_hz, n_cycles.
    """
    if envelope is None:
        env, inst_f = spindle_envelope(x, fs, center_freq, half_bandwidth)
    else:
        env = envelope
        _, inst_f = spindle_envelope(x, fs, center_freq, half_bandwidth)
    lo_dur = min_dur if min_dur is not None else (min_cycles / center_freq)
    hi_dur = max_dur if max_dur is not None else (max_cycles / center_freq)
    thr = np.percentile(env, thresh_percentile)
    thr_b = np.percentile(env, min(boundary_percentile, thresh_percentile))
    b_starts, b_ends = _segments(env >= thr_b)
    # keep boundary segments that contain at least one detection-threshold sample
    kept = [(int(s), int(e)) for s, e in zip(b_starts, b_ends) if np.any(env[s:e] >= thr)]
    # merge segments separated by short gaps
    merged = []
    for s, e in kept:
        if merged and (s - merged[-1][1]) / fs <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    rows = []
    for s, e in merged:
        dur = (e - s) / fs
        if not (lo_dur <= dur <= hi_dur):
            continue
        pk = s + int(np.argmax(env[s:e]))
        freq = float(np.median(inst_f[s:e]))
        rows.append({"start_s": s / fs, "end_s": e / fs, "peak_s": pk / fs, "duration_s": dur,
                     "peak_amp": float(env[pk]), "freq_hz": freq, "n_cycles": dur * freq})
    df = pd.DataFrame(rows, columns=["start_s", "end_s", "peak_s", "duration_s", "peak_amp", "freq_hz", "n_cycles"])
    df.attrs["threshold"] = float(thr)
    df.attrs["band"] = (center_freq - half_bandwidth, center_freq + half_bandwidth)
    return df


def event_density(events: pd.DataFrame, duration_s: float) -> float:
    """Events per minute."""
    return 60.0 * len(events) / duration_s if duration_s > 0 else np.nan
