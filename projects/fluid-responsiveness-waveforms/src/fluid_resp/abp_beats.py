"""Arterial-pressure beat detection, per-beat pressures, signal quality and a synthetic ABP generator.

The detector is deliberately simple and transparent (prominence-based systolic peak detection with a
refractory period, diastolic minimum between consecutive peaks) so that its failure modes can be audited; a
signal-quality mask removes beats outside physiologic ranges, implausible beat-to-beat jumps and flat-line
segments.  The synthetic generator produces a beat train whose pulse pressure is modulated at the
respiratory frequency with a known PPV, which the tests use to check that the PPV estimator recovers it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_beats(abp: np.ndarray, fs: float, min_hr: float = 30.0, max_hr: float = 220.0, prominence: float = 8.0) -> pd.DataFrame:
    """Detect systolic peaks and preceding diastolic minima.

    Returns a DataFrame with one row per beat: ``i_sys`` (sample), ``t_sys`` (s), ``sbp``, ``dbp``, ``pp``,
    ``map`` (DBP + PP/3), ``rr_s`` (interval to previous systolic peak; NaN for the first beat).
    """
    x = np.asarray(abp, dtype=float)
    if x.ndim != 1 or x.size < int(fs):
        raise ValueError("abp must be a 1-D array of at least 1 s")
    distance = int(round(fs * 60.0 / max_hr))
    peaks, _ = find_peaks(x, distance=max(distance, 1), prominence=prominence)
    if peaks.size < 2:
        return pd.DataFrame(columns=["i_sys", "t_sys", "sbp", "dbp", "pp", "map", "rr_s"])
    max_gap = fs * 60.0 / min_hr
    rows = []
    prev = None
    for k, p in enumerate(peaks):
        lo = peaks[k - 1] if k > 0 else max(0, p - int(max_gap))
        seg = x[lo:p + 1]
        i_dia = lo + int(np.argmin(seg))
        dbp = float(x[i_dia])
        sbp = float(x[p])
        rr = (p - prev) / fs if prev is not None else np.nan
        rows.append({"i_sys": int(p), "t_sys": p / fs, "sbp": sbp, "dbp": dbp, "pp": sbp - dbp, "map": dbp + (sbp - dbp) / 3.0, "rr_s": rr})
        prev = p
    return pd.DataFrame(rows)


def sqi_mask(
    beats: pd.DataFrame,
    sbp_range: Tuple[float, float] = (40.0, 250.0),
    dbp_range: Tuple[float, float] = (20.0, 160.0),
    pp_min: float = 10.0,
    max_rel_jump: float = 0.5,
    hr_range: Tuple[float, float] = (30.0, 220.0),
) -> np.ndarray:
    """Boolean mask of beats passing simple physiologic and continuity checks.

    A beat fails if SBP/DBP/PP are out of range, if its SBP or PP differs from the previous beat by more than
    ``max_rel_jump`` (relative), or if the RR interval implies a heart rate outside ``hr_range``.
    """
    if len(beats) == 0:
        return np.zeros(0, dtype=bool)
    sbp, dbp, pp, rr = (beats[c].to_numpy(dtype=float) for c in ("sbp", "dbp", "pp", "rr_s"))
    ok = (sbp >= sbp_range[0]) & (sbp <= sbp_range[1]) & (dbp >= dbp_range[0]) & (dbp <= dbp_range[1]) & (pp >= pp_min)
    hr = 60.0 / rr
    ok &= np.isnan(rr) | ((hr >= hr_range[0]) & (hr <= hr_range[1]))
    jump = np.zeros_like(ok)
    jump[1:] = (np.abs(np.diff(sbp)) / np.maximum(sbp[:-1], 1.0) > max_rel_jump) | (np.abs(np.diff(pp)) / np.maximum(pp[:-1], 1.0) > max_rel_jump)
    return ok & ~jump


def flatline_fraction(abp: np.ndarray, fs: float, window_s: float = 2.0, tol: float = 0.5) -> float:
    """Fraction of ``window_s`` windows whose peak-to-peak range is below ``tol`` mmHg (flat-line / disconnect)."""
    x = np.asarray(abp, dtype=float)
    n = int(window_s * fs)
    if n <= 0 or x.size < n:
        return float(np.ptp(x) < tol)
    k = x.size // n
    w = x[: k * n].reshape(k, n)
    return float(np.mean(np.ptp(w, axis=1) < tol))


def rr_irregularity(beats: pd.DataFrame, mask: Optional[np.ndarray] = None, jump_s: float = 0.15) -> float:
    """Fraction of successive RR-interval differences larger than ``jump_s`` (a crude atrial-fibrillation index)."""
    rr = beats["rr_s"].to_numpy(dtype=float)
    if mask is not None:
        rr = rr[np.asarray(mask, dtype=bool)]
    rr = rr[np.isfinite(rr)]
    if rr.size < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(rr)) > jump_s))


# ------------------------------------------------------------------------------------------------ synthetic
def _pulse_template(phase: np.ndarray) -> np.ndarray:
    """Normalised pressure pulse on phase in [0, 1): fast upstroke, exponential decay with a small dicrotic notch."""
    up = 0.12
    y = np.where(phase < up, np.sin(0.5 * np.pi * phase / up), np.exp(-(phase - up) / 0.35))
    notch = 0.08 * np.exp(-((phase - 0.42) ** 2) / (2 * 0.02**2))
    return y + notch


def synthetic_abp(
    fs: float = 125.0,
    duration_s: float = 60.0,
    hr_bpm: float = 80.0,
    rr_bpm: float = 15.0,
    ppv_pct: float = 12.0,
    sbp_base: float = 120.0,
    dbp_base: float = 70.0,
    noise_sd: float = 0.5,
    hr_jitter_s: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic ABP whose beat-to-beat pulse pressure is modulated sinusoidally at ``rr_bpm`` with the given PPV.

    With PP_i = PP0 * (1 + a sin(2 pi f_r t_i)), PPV = 100 (PPmax - PPmin) / ((PPmax + PPmin)/2) = 200 a, so
    ``a = ppv_pct / 200``. Returns ``(t, abp)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0, duration_s, 1.0 / fs)
    abp = np.full_like(t, dbp_base)
    pp0 = sbp_base - dbp_base
    a = ppv_pct / 200.0
    f_r = rr_bpm / 60.0
    t_beat = 0.0
    while t_beat < duration_s:
        rr = 60.0 / hr_bpm + (rng.normal(0, hr_jitter_s) if hr_jitter_s > 0 else 0.0)
        rr = max(rr, 0.3)
        pp = pp0 * (1.0 + a * np.sin(2 * np.pi * f_r * t_beat))
        i0, i1 = int(t_beat * fs), min(int((t_beat + rr) * fs), t.size)
        if i1 > i0:
            phase = (np.arange(i0, i1) / fs - t_beat) / rr
            abp[i0:i1] = dbp_base + pp * _pulse_template(phase)
        t_beat += rr
    abp += rng.normal(0, noise_sd, size=abp.size)
    return t, abp
