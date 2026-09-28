"""ABP beat detection, per-beat features, and a Windkessel generator for synthetic waveforms.

* ``synthetic_abp`` drives a two-element Windkessel (Frank, 1899) with a half-sine
  ejection flow whose area is the stroke volume, so the *true* SV of every beat
  is known. It is used by the tests and for sanity checks of the estimators.
* ``detect_onsets`` is a compact slope-sum-function detector in the spirit of
  the open-source WABP algorithm (Zong et al., Computers in Cardiology 2003).
* ``beat_features`` returns the quantities every pulse-contour estimator needs:
  SBP, DBP, MAP, pulse pressure, systolic area, dicrotic-notch time, dP/dt max
  and the diastolic decay time constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PHYSIOLOGIC = {"sbp": (40.0, 300.0), "dbp": (20.0, 200.0), "pp": (10.0, 200.0), "hr": (25.0, 220.0)}


@dataclass
class SyntheticABP:
    t: np.ndarray
    abp: np.ndarray
    onsets: np.ndarray
    sv_true: np.ndarray
    fs: float


def synthetic_abp(fs: float = 125.0, n_beats: int = 40, hr: float = 75.0, sv_ml: float = 70.0,
                  sv_variation: float = 0.15, resp_rate: float = 15.0, R: float = 1.0, C: float = 1.5,
                  noise_mmhg: float = 0.5, seed: int = 0, p0: float = 80.0) -> SyntheticABP:
    """Two-element Windkessel ABP with respiratory SV modulation.

    Units: pressure mmHg, flow mL/s, ``R`` mmHg*s/mL, ``C`` mL/mmHg. Ejection is a half-sine of
    duration ~0.3 s (scaled with sqrt(RR)); its area equals the beat's stroke volume.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    rr = 60.0 / hr
    t_ej = 0.30 * np.sqrt(rr / 0.8)
    onset_times = np.cumsum(np.r_[0.5, rr * (1 + 0.02 * rng.normal(size=n_beats - 1))])
    total = onset_times[-1] + rr + 0.5
    n = int(np.ceil(total * fs))
    t = np.arange(n) / fs
    sv_true = sv_ml * (1 + sv_variation * np.sin(2 * np.pi * resp_rate / 60 * onset_times) + 0.03 * rng.normal(size=n_beats))
    q = np.zeros(n)
    for t0, sv in zip(onset_times, sv_true):
        i0, i1 = int(round(t0 * fs)), int(round((t0 + t_ej) * fs))
        tau = (np.arange(i0, min(i1, n)) / fs - t0) / t_ej
        q[i0:min(i1, n)] = (np.pi * sv / (2 * t_ej)) * np.sin(np.pi * tau)
    p = np.empty(n)
    p[0] = p0
    for i in range(1, n):
        p[i] = p[i - 1] + dt * (q[i - 1] / C - p[i - 1] / (R * C))
    p += noise_mmhg * rng.normal(size=n)
    onsets = np.round(onset_times * fs).astype(int)
    return SyntheticABP(t, p, onsets, sv_true, fs)


def slope_sum(abp: np.ndarray, fs: float, window_s: float = 0.128) -> np.ndarray:
    """Slope sum function: windowed sum of positive first differences (Zong et al., 2003)."""
    d = np.diff(abp, prepend=abp[0])
    d[d < 0] = 0.0
    w = max(1, int(round(window_s * fs)))
    return np.convolve(d, np.ones(w), mode="full")[: len(abp)]


def detect_onsets(abp: np.ndarray, fs: float, min_rr_s: float = 0.3, search_back_s: float = 0.25) -> np.ndarray:
    """Beat onset (foot) indices from the SSF peaks with an adaptive threshold."""
    ssf = slope_sum(abp, fs)
    thr = 0.4 * np.percentile(ssf, 98)
    peaks, _ = find_peaks(ssf, height=thr, distance=int(min_rr_s * fs))
    back = int(search_back_s * fs)
    onsets = []
    for pk in peaks:
        lo = max(0, pk - back)
        seg = abp[lo:pk + 1]
        if seg.size:
            onsets.append(lo + int(np.argmin(seg)))
    return np.unique(np.asarray(onsets, dtype=int))


def find_dicrotic_notch(beat: np.ndarray, fs: float, peak_idx: int,
                        min_after_s: float = 0.05, max_after_s: float = 0.45) -> int:
    """End-of-ejection index: maximum of the smoothed second derivative after the systolic peak."""
    lo = min(len(beat) - 1, peak_idx + int(min_after_s * fs))
    hi = min(len(beat) - 1, peak_idx + int(max_after_s * fs))
    if hi - lo < 3:
        return min(len(beat) - 1, peak_idx + int(0.3 * fs))
    k = max(3, int(0.03 * fs) | 1)
    sm = np.convolve(beat, np.ones(k) / k, mode="same")
    d2 = np.gradient(np.gradient(sm))
    return lo + int(np.argmax(d2[lo:hi]))


def diastolic_tau(beat: np.ndarray, fs: float, notch_idx: int) -> float:
    """Time constant (s) of the exponential diastolic decay from a log-linear fit after the notch."""
    seg = beat[notch_idx + int(0.05 * fs):]
    seg = seg[seg > 1.0]
    if seg.size < 5:
        return np.nan
    x = np.arange(seg.size) / fs
    slope = np.polyfit(x, np.log(seg), 1)[0]
    return float(-1.0 / slope) if slope < 0 else np.nan


def beat_features(abp: np.ndarray, fs: float, onsets: np.ndarray) -> pd.DataFrame:
    """One row per beat: ``onset, sbp, dbp, map, pp, hr, t_sys, sys_area, notch_p, dpdt_max, tau, ok``.

    ``sys_area`` is the area above end-diastolic pressure from onset to the notch (mmHg*s).
    ``ok`` flags beats inside the physiologic ranges in ``PHYSIOLOGIC``.
    """
    rows = []
    for i in range(len(onsets) - 1):
        a, b = int(onsets[i]), int(onsets[i + 1])
        beat = abp[a:b]
        if beat.size < int(0.2 * fs):
            continue
        dbp = float(beat[0])
        pk = int(np.argmax(beat))
        sbp = float(beat[pk])
        notch = find_dicrotic_notch(beat, fs, pk)
        sys_area = float(np.trapezoid(beat[: notch + 1] - dbp, dx=1 / fs)) if hasattr(np, "trapezoid") else float(np.trapz(beat[: notch + 1] - dbp, dx=1 / fs))
        hr = 60.0 * fs / (b - a)
        row = {
            "onset": a, "sbp": sbp, "dbp": dbp, "map": float(beat.mean()), "pp": sbp - dbp, "hr": hr,
            "t_sys": notch / fs, "sys_area": sys_area, "notch_p": float(beat[notch]),
            "dpdt_max": float(np.max(np.diff(beat[: pk + 1])) * fs) if pk > 0 else np.nan,
            "tau": diastolic_tau(beat, fs, notch),
        }
        row["ok"] = all(PHYSIOLOGIC[k][0] <= row[k] <= PHYSIOLOGIC[k][1] for k in PHYSIOLOGIC)
        rows.append(row)
    return pd.DataFrame(rows)
