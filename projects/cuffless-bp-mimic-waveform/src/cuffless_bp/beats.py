"""Beat detection and pulse-arrival / pulse-transit-time features.

* :func:`detect_r_peaks`   simplified Pan-Tompkins (band-pass, derivative, squaring, integration,
                           adaptive threshold, refractory period) refined to the local maximum.
* :func:`ppg_fiducials`    systolic peak, preceding foot (onset) and maximum-slope point per beat.
* :func:`abp_beats`        per-beat SBP (peak), DBP (foot), MAP (mean over the beat).
* :func:`pat_features`     per-window features: PAT to foot / peak / max-slope, PPG-ABP PTT,
                           HR, PPG morphology, and the reference SBP/DBP/MAP.

Conventions: sample indices are ints; times in seconds; pressures in mmHg.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(hi, 0.99 * nyq)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def detect_r_peaks(ecg: np.ndarray, fs: float, refractory_s: float = 0.25, search_s: float = 0.05) -> np.ndarray:
    """R-peak sample indices (Pan & Tompkins, 1985, simplified; adaptive threshold per window)."""
    x = np.asarray(ecg, dtype=float)
    if len(x) < int(2 * fs) or np.std(x) < 1e-12:
        return np.array([], dtype=int)
    xf = bandpass(x, fs, 5.0, 20.0)
    d = np.gradient(xf)
    sq = d ** 2
    w = max(int(0.15 * fs), 1)
    integ = np.convolve(sq, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(integ, 98)
    cand, _ = find_peaks(integ, height=thr, distance=int(refractory_s * fs))
    # refine to the maximum |bandpassed| within +-search_s (R-peaks may be negative in some leads)
    r = int(search_s * fs)
    sign = 1.0 if np.abs(np.max(xf)) >= np.abs(np.min(xf)) else -1.0
    peaks = []
    for c in cand:
        a, b = max(c - r, 0), min(c + r + 1, len(xf))
        peaks.append(a + int(np.argmax(sign * xf[a:b])))
    peaks = np.unique(np.array(peaks, dtype=int))
    return peaks


def ppg_fiducials(ppg: np.ndarray, fs: float, min_ibi_s: float = 0.3, foot_search_s: float = 0.5) -> dict[str, np.ndarray]:
    """Systolic peaks, feet (minimum in the preceding ``foot_search_s``) and max-slope points."""
    x = np.asarray(ppg, dtype=float)
    empty = {"peak": np.array([], int), "foot": np.array([], int), "max_slope": np.array([], int)}
    if len(x) < int(2 * fs) or np.std(x) < 1e-12:
        return empty
    xf = bandpass(x, fs, 0.5, 8.0)
    prom = 0.3 * np.std(xf)
    peaks, _ = find_peaks(xf, distance=int(min_ibi_s * fs), prominence=prom)
    if len(peaks) == 0:
        return empty
    dx = np.gradient(xf)
    feet, slopes, keep = [], [], []
    r = int(foot_search_s * fs)
    for p in peaks:
        a = max(p - r, 0)
        if p - a < 3:
            continue
        foot = a + int(np.argmin(xf[a:p]))
        ms = foot + int(np.argmax(dx[foot:p + 1]))
        feet.append(foot)
        slopes.append(ms)
        keep.append(p)
    return {"peak": np.array(keep, int), "foot": np.array(feet, int), "max_slope": np.array(slopes, int)}


def abp_beats(abp: np.ndarray, fs: float, min_ibi_s: float = 0.3, min_prominence: float = 8.0) -> dict[str, np.ndarray]:
    """Per-beat SBP/DBP/MAP from the raw arterial pressure waveform."""
    x = np.asarray(abp, dtype=float)
    empty = {k: np.array([]) for k in ("peak", "foot", "sbp", "dbp", "map")}
    if len(x) < int(2 * fs) or np.std(x) < 1e-12:
        return empty
    peaks, _ = find_peaks(x, distance=int(min_ibi_s * fs), prominence=min_prominence)
    if len(peaks) < 2:
        return empty
    r = int(0.5 * fs)
    feet = []
    for p in peaks:
        a = max(p - r, 0)
        feet.append(a + int(np.argmin(x[a:p + 1])))
    feet = np.array(feet, int)
    sbp = x[peaks]
    dbp = x[feet]
    maps = np.full(len(peaks), np.nan)
    for i in range(len(feet) - 1):
        maps[i] = float(np.mean(x[feet[i]:feet[i + 1]]))
    return {"peak": peaks, "foot": feet, "sbp": sbp, "dbp": dbp, "map": maps}


def pair_events(ref: np.ndarray, target: np.ndarray, fs: float, min_s: float = 0.08, max_s: float = 0.6) -> np.ndarray:
    """For each reference event, the first target event within (min_s, max_s] seconds later.

    Returns an array of delays in seconds (NaN when no match), aligned with ``ref``.
    """
    ref = np.asarray(ref, int)
    target = np.asarray(target, int)
    out = np.full(len(ref), np.nan)
    if len(target) == 0:
        return out
    j = 0
    for i, r in enumerate(ref):
        while j < len(target) and target[j] <= r + min_s * fs:
            j += 1
        if j < len(target) and target[j] - r <= max_s * fs:
            out[i] = (target[j] - r) / fs
    return out


FEATURE_COLUMNS = ("pat_foot", "pat_peak", "pat_slope", "ptt_abp_ppg", "hr", "ppg_amp", "ppg_rise_t", "ppg_width50",
                   "pat_foot_sd", "hr_sd", "n_beats")
TARGET_COLUMNS = ("sbp", "dbp", "map")


def pat_features(ecg: np.ndarray, ppg: np.ndarray, abp: np.ndarray, fs: float) -> dict[str, float]:
    """Window-level feature dict (medians over matched beats) + reference SBP/DBP/MAP.

    PAT_x = time from R-peak to PPG fiducial x. PTT_abp_ppg = time from the ABP foot (radial
    catheter) to the PPG foot (finger) - a true transit time without the pre-ejection period.
    """
    r = detect_r_peaks(ecg, fs)
    f = ppg_fiducials(ppg, fs)
    b = abp_beats(abp, fs)
    out: dict[str, float] = {k: np.nan for k in FEATURE_COLUMNS + TARGET_COLUMNS}
    out["n_beats"] = float(min(len(r), len(f["peak"])))
    if len(r) >= 3 and len(f["peak"]) >= 3:
        pat_foot = pair_events(r, f["foot"], fs)
        pat_peak = pair_events(r, f["peak"], fs, max_s=0.8)
        pat_slope = pair_events(r, f["max_slope"], fs)
        out["pat_foot"] = float(np.nanmedian(pat_foot))
        out["pat_peak"] = float(np.nanmedian(pat_peak))
        out["pat_slope"] = float(np.nanmedian(pat_slope))
        out["pat_foot_sd"] = float(np.nanstd(pat_foot))
        rr = np.diff(r) / fs
        out["hr"] = float(60.0 / np.median(rr))
        out["hr_sd"] = float(np.std(60.0 / rr))
        xf = bandpass(ppg, fs, 0.5, 8.0)
        out["ppg_amp"] = float(np.median(xf[f["peak"]] - xf[f["foot"]]))
        out["ppg_rise_t"] = float(np.median((f["peak"] - f["foot"]) / fs))
        # width at 50% of the upstroke amplitude
        widths = []
        for pk, ft in zip(f["peak"], f["foot"]):
            half = xf[ft] + 0.5 * (xf[pk] - xf[ft])
            j = pk
            while j < len(xf) and xf[j] > half:
                j += 1
            i = pk
            while i > ft and xf[i] > half:
                i -= 1
            widths.append((j - i) / fs)
        out["ppg_width50"] = float(np.median(widths)) if widths else np.nan
    if len(b["sbp"]) >= 3:
        out["sbp"] = float(np.median(b["sbp"]))
        out["dbp"] = float(np.median(b["dbp"]))
        out["map"] = float(np.nanmedian(b["map"]))
        if len(f["foot"]) >= 3:
            out["ptt_abp_ppg"] = float(np.nanmedian(pair_events(b["foot"], f["foot"], fs, min_s=0.0, max_s=0.4)))
    return out


def feature_table(windows: Sequence, fs: float | None = None) -> pd.DataFrame:
    """Apply :func:`pat_features` to an iterable of ``Window`` objects -> tidy DataFrame."""
    rows = []
    for w in windows:
        feats = pat_features(w.ecg, w.ppg, w.abp, fs or w.fs)
        rows.append({"source": w.source, "record": w.record, "subject_id": w.subject_id, "start_s": w.start_s,
                     "timestamp": getattr(w, "timestamp", None), **feats})
    return pd.DataFrame(rows)
