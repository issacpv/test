"""Recompute a neurofeedback signal from raw EEG under a :class:`FeedbackSpec`.

The pipeline mirrors what an online system does, step by step:

1. spatial reference (none / CAR / Laplacian / linked mastoids),
2. target extraction (mean of target channels),
3. optional artifact handling (freeze on amplitude threshold, or regress the
   frontal EOG proxy),
4. band definition (fixed or individualised on the baseline),
5. band-power estimation in sliding windows,
6. normalisation to the baseline,
7. smoothing,
8. reward decision (threshold rule).

Everything is numpy/scipy; loading EDF/BrainVision files is done by the caller
(``mne`` is the recommended reader, see data/README.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

from .specs import FeedbackSpec

DEFAULT_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
    "Pz": ("P3", "P4", "Cz", "O1", "O2"),
    "P3": ("P7", "Pz", "C3", "O1"),
    "P4": ("P8", "Pz", "C4", "O2"),
    "Cz": ("Fz", "C3", "C4", "Pz"),
    "C3": ("F3", "P3", "Cz", "T7"),
    "C4": ("F4", "P4", "Cz", "T8"),
    "Fz": ("Fp1", "Fp2", "F3", "F4", "Cz"),
    "O1": ("P3", "Pz", "P7", "O2"),
    "O2": ("P4", "Pz", "P8", "O1"),
}


@dataclass
class FeedbackResult:
    times: np.ndarray  # window centres (s)
    values: np.ndarray  # feedback values (NaN where frozen)
    raw_power: np.ndarray  # un-normalised band power per window
    band: Tuple[float, float]
    spec: FeedbackSpec
    bad_fraction: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.times, "feedback": self.values, "power": self.raw_power})


# --------------------------------------------------------------------------- #
# spatial filtering and target extraction
# --------------------------------------------------------------------------- #
def _idx(ch_names: Sequence[str], names: Iterable[str]) -> List[int]:
    lookup = {c.upper(): i for i, c in enumerate(ch_names)}
    out = []
    for n in names:
        if n.upper() in lookup:
            out.append(lookup[n.upper()])
    return out


def extract_target(eeg: np.ndarray, ch_names: Sequence[str], reference: str, target_channels: Sequence[str], neighbors: Optional[Dict[str, Tuple[str, ...]]] = None) -> np.ndarray:
    """Return the 1-D target signal after spatial referencing."""
    eeg = np.asarray(eeg, dtype=np.float64)
    tgt = _idx(ch_names, target_channels)
    if not tgt:
        raise ValueError(f"none of {target_channels} found in {ch_names}")
    if reference == "none":
        return eeg[tgt].mean(0)
    if reference == "car":
        scalp = [i for i, c in enumerate(ch_names) if c.upper() not in ("M1", "M2", "A1", "A2", "TP9", "TP10")]
        return (eeg[tgt] - eeg[scalp].mean(0, keepdims=True)).mean(0)
    if reference == "mastoid":
        m = _idx(ch_names, ("M1", "M2", "A1", "A2", "TP9", "TP10"))
        if not m:
            return (eeg[tgt] - eeg.mean(0, keepdims=True)).mean(0)  # fall back to CAR
        return (eeg[tgt] - eeg[m].mean(0, keepdims=True)).mean(0)
    if reference == "laplacian":
        nb = neighbors or DEFAULT_NEIGHBORS
        out = []
        for name, i in zip([c for c in target_channels if _idx(ch_names, [c])], tgt):
            ni = _idx(ch_names, nb.get(name, ()))
            out.append(eeg[i] - eeg[ni].mean(0) if ni else eeg[i] - eeg.mean(0))
        return np.mean(out, axis=0)
    raise ValueError(f"unknown reference {reference}")


def eog_proxy(eeg: np.ndarray, ch_names: Sequence[str], fs: float, channels: Sequence[str] = ("Fp1", "Fp2"), cutoff_hz: float = 15.0) -> np.ndarray:
    """Low-passed mean of frontal-polar channels as a blink/eye-movement regressor.

    The cut-off is a genuine trade-off: too low and the step-like leakage of
    saccades into the alpha band is not removed; too high and frontal alpha is
    regressed out with the artifact.  It is therefore part of the multiverse.
    """
    idx = _idx(ch_names, channels)
    if not idx:
        return np.zeros(np.asarray(eeg).shape[1])
    b, a = sps.butter(2, cutoff_hz / (fs / 2), btype="low")
    return sps.filtfilt(b, a, np.asarray(eeg, float)[idx].mean(0))


def regress_out(x: np.ndarray, reg: np.ndarray) -> np.ndarray:
    reg = reg - reg.mean()
    denom = float(reg @ reg)
    if denom <= 0:
        return x
    beta = float((x - x.mean()) @ reg) / denom
    return x - beta * reg


# --------------------------------------------------------------------------- #
# band and power
# --------------------------------------------------------------------------- #
def individual_alpha_frequency(x: np.ndarray, fs: float, search: Tuple[float, float] = (7.0, 13.0)) -> float:
    """Peak frequency in ``search`` after removing the 1/f trend from the Welch PSD."""
    f, p = sps.welch(np.asarray(x, float), fs=fs, nperseg=int(min(len(x), 4 * fs)))
    m = (f >= 2) & (f <= 40) & (p > 0)
    coef = np.polyfit(np.log(f[m]), np.log(p[m]), 1)
    resid = np.log(p[m]) - np.polyval(coef, np.log(f[m]))
    fm = f[m]
    sel = (fm >= search[0]) & (fm <= search[1])
    if not sel.any():
        return float(np.mean(search))
    return float(fm[sel][np.argmax(resid[sel])])


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    hi = min(hi, fs / 2 - 1e-3)
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


def band_power_windows(x: np.ndarray, fs: float, band: Tuple[float, float], window_s: float, step_s: float, estimator: str) -> Tuple[np.ndarray, np.ndarray]:
    """Band power per sliding window. Returns ``(centre_times, power)``."""
    x = np.asarray(x, float)
    n = len(x)
    w = int(round(window_s * fs))
    s = max(1, int(round(step_s * fs)))
    starts = np.arange(0, n - w + 1, s)
    times = (starts + w / 2) / fs
    lo, hi = band
    if estimator in ("hilbert", "bandpass_rms"):
        xb = _bandpass(x, fs, lo, hi)
        inst = np.abs(sps.hilbert(xb)) ** 2 if estimator == "hilbert" else xb**2
        csum = np.concatenate([[0.0], np.cumsum(inst)])
        power = (csum[starts + w] - csum[starts]) / w
        return times, power
    power = np.empty(len(starts))
    for k, st in enumerate(starts):
        seg = x[st : st + w]
        if estimator == "welch":
            f, p = sps.welch(seg, fs=fs, nperseg=min(w, int(fs)))
        elif estimator == "fft":
            f, p = sps.periodogram(seg, fs=fs, window="hann")
        else:
            raise ValueError(f"unknown estimator {estimator}")
        m = (f >= lo) & (f <= hi)
        power[k] = float(np.mean(p[m])) if m.any() else np.nan
    return times, power


def normalize(power: np.ndarray, baseline_power: np.ndarray, method: str) -> np.ndarray:
    bl = baseline_power[np.isfinite(baseline_power)]
    mu, sd = float(np.mean(bl)), float(np.std(bl) + 1e-12)
    if method == "none":
        return power
    if method == "baseline_z":
        return (power - mu) / sd
    if method == "baseline_ratio":
        return power / mu
    if method == "log_ratio":
        return np.log(np.clip(power, 1e-12, None) / mu)
    raise ValueError(f"unknown normalization {method}")


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - k + 1) : i + 1]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def hold_last(x: np.ndarray) -> np.ndarray:
    """Replace NaN by the previous finite value (what online systems do when feedback is frozen)."""
    out = np.array(x, dtype=float)
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def compute_feedback(
    eeg: np.ndarray,
    ch_names: Sequence[str],
    fs: float,
    spec: FeedbackSpec,
    target_channels: Sequence[str] = ("Pz",),
    baseline: Tuple[float, float] = (0.0, 30.0),
    neighbors: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> FeedbackResult:
    """Feedback time series for one recording under ``spec``.

    ``baseline`` is the (start, end) in seconds of the eyes-open rest period
    used for normalisation and for the individual alpha frequency.
    """
    eeg = np.asarray(eeg, float)
    x = extract_target(eeg, ch_names, spec.reference, target_channels, neighbors)
    if spec.artifact == "regress_eog":
        x = regress_out(x, eog_proxy(eeg, ch_names, fs))
    b0, b1 = int(baseline[0] * fs), int(baseline[1] * fs)
    if spec.band == "individual":
        iaf = individual_alpha_frequency(x[b0:b1], fs)
        half = (spec.band_high - spec.band_low) / 2
        band = (iaf - half, iaf + half)
    else:
        band = (spec.band_low, spec.band_high)
    times, power = band_power_windows(x, fs, band, spec.window_s, spec.step_s, spec.estimator)
    bad = np.zeros(len(times), bool)
    if spec.artifact == "threshold":
        w = int(round(spec.window_s * fs))
        starts = np.arange(0, len(x) - w + 1, max(1, int(round(spec.step_s * fs))))
        # peak-to-peak on the raw (recording-reference) target channels, as online systems do
        raw = eeg[[i for i, c in enumerate(ch_names) if c.upper() in {t.upper() for t in target_channels}]].mean(0)
        for k, st in enumerate(starts[: len(times)]):
            seg = raw[st : st + w]
            bad[k] = (seg.max() - seg.min()) > spec.artifact_threshold_uv
        power = np.where(bad, np.nan, power)
    in_bl = (times >= baseline[0]) & (times < baseline[1])
    values = normalize(power, power[in_bl] if in_bl.any() else power, spec.normalization)
    values = moving_average(hold_last(values), spec.smoothing_windows)
    return FeedbackResult(times, values, power, band, spec, float(bad.mean()))


# --------------------------------------------------------------------------- #
# reward decisions and agreement
# --------------------------------------------------------------------------- #
def reward_decisions(values: np.ndarray, rule: str = "baseline_threshold", level: float = 0.0, adaptive_window: int = 40, direction: str = "up") -> np.ndarray:
    """Binary reward per window.

    ``'baseline_threshold'``: value > ``level`` (for z / log-ratio units 0 means
    "above baseline mean"); ``'percentile'``: above the ``level`` quantile of the
    whole session (offline convenience); ``'adaptive_median'``: above the running
    median of the previous ``adaptive_window`` windows (common in commercial systems).
    """
    v = np.asarray(values, float)
    if rule == "baseline_threshold":
        dec = v > level
    elif rule == "percentile":
        dec = v > np.nanquantile(v, level)
    elif rule == "adaptive_median":
        dec = np.zeros(len(v), bool)
        for i in range(len(v)):
            ref = v[max(0, i - adaptive_window) : i]
            ref = ref[np.isfinite(ref)]
            dec[i] = v[i] > np.median(ref) if len(ref) >= 5 else False
    else:
        raise ValueError(rule)
    dec = np.where(np.isfinite(v), dec, False)
    return ~dec if direction == "down" else dec


def decision_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of windows on which two reward sequences agree (aligned by index)."""
    n = min(len(a), len(b))
    return float(np.mean(np.asarray(a[:n]) == np.asarray(b[:n])))


def align_to_times(times_ref: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resample ``values`` onto ``times_ref`` (for comparing different step sizes)."""
    idx = np.clip(np.searchsorted(times, times_ref), 0, len(times) - 1)
    left = np.clip(idx - 1, 0, len(times) - 1)
    pick = np.where(np.abs(times[left] - times_ref) < np.abs(times[idx] - times_ref), left, idx)
    return np.asarray(values)[pick]


def agreement_matrix(decisions: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Pairwise agreement between named ``(times, decisions)`` sequences, resampled onto the first."""
    names = list(decisions)
    ref_t = decisions[names[0]][0]
    aligned = {n: align_to_times(ref_t, t, d) for n, (t, d) in decisions.items()}
    M = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                M[i, j] = M[j, i] = decision_agreement(aligned[a], aligned[b])
    return pd.DataFrame(M, index=names, columns=names)
