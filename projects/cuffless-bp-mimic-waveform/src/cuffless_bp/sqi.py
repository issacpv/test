"""Signal-quality indices (SQI) for 10-s PPG, ECG and ABP windows.

PPG: skewness / kurtosis SQI (Elgendi, 2016 Bioengineering), perfusion index, beat-template
correlation (Orphanidou et al., 2015 IEEE JBHI), flatline and clipping fractions.
ECG: kSQI, pSQI, basSQI (Li, Clifford & Rajagopalan, 2008 Physiol Meas), flatline.
ABP: physiological-plausibility rules in the spirit of the "jSQI" of Sun, Reisner & Mark (2006
Computers in Cardiology): pressure ranges, pulse pressure, dP/dt, beat-to-beat jumps, flatline.

All functions return a dict of the component values plus an ``accept`` boolean so that the
acceptance rule can be audited and its thresholds tuned on training subjects only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .beats import abp_beats, bandpass, ppg_fiducials


def _flatline_fraction(x: np.ndarray, fs: float, min_s: float = 0.5, tol: float = 1e-6) -> float:
    """Fraction of samples inside runs (>= ``min_s`` s) where the signal does not change."""
    d = np.abs(np.diff(x)) <= tol
    if not d.any():
        return 0.0
    run_len = int(min_s * fs)
    flat = np.zeros(len(x), bool)
    i = 0
    while i < len(d):
        if d[i]:
            j = i
            while j < len(d) and d[j]:
                j += 1
            if j - i >= run_len:
                flat[i:j + 1] = True
            i = j
        else:
            i += 1
    return float(flat.mean())


def _clipping_fraction(x: np.ndarray, tol: float = 1e-9) -> float:
    lo, hi = np.min(x), np.max(x)
    return float(np.mean((np.abs(x - lo) < tol) | (np.abs(x - hi) < tol)))


def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    m = (f >= lo) & (f < hi)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(p[m], f[m])) if m.any() else 0.0


def ppg_sqi(ppg: np.ndarray, fs: float, min_template_corr: float = 0.8, max_flat: float = 0.05,
            max_clip: float = 0.02, hr_range: tuple[float, float] = (30.0, 200.0)) -> dict:
    """PPG quality: returns skewness, kurtosis, perfusion, template correlation, flat/clip fractions."""
    x = np.asarray(ppg, dtype=float)
    out = {"flat": _flatline_fraction(x, fs), "clip": _clipping_fraction(x)}
    if np.std(x) < 1e-9:
        out.update(skew=np.nan, kurt=np.nan, perfusion=0.0, template_corr=0.0, n_beats=0, hr=np.nan, accept=False)
        return out
    xf = bandpass(x, fs, 0.5, 8.0)
    out["skew"] = float(skew(xf))
    out["kurt"] = float(kurtosis(xf))
    dc = float(np.mean(x))
    out["perfusion"] = float((np.max(xf) - np.min(xf)) / abs(dc)) if abs(dc) > 1e-9 else np.nan
    fid = ppg_fiducials(x, fs)
    peaks = fid["peak"]
    out["n_beats"] = int(len(peaks))
    if len(peaks) >= 3:
        ibi = np.diff(peaks) / fs
        out["hr"] = float(60.0 / np.median(ibi))
        # template correlation: each beat (foot to foot) vs the mean beat, resampled to 100 points
        beats = []
        feet = fid["foot"]
        for a, b in zip(feet[:-1], feet[1:]):
            seg = xf[a:b]
            if len(seg) >= 10:
                beats.append(np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg))
        if len(beats) >= 3:
            B = np.array(beats)
            tmpl = B.mean(axis=0)
            corr = [np.corrcoef(b, tmpl)[0, 1] for b in B]
            out["template_corr"] = float(np.nanmedian(corr))
        else:
            out["template_corr"] = 0.0
    else:
        out["hr"] = np.nan
        out["template_corr"] = 0.0
    out["accept"] = bool(out["template_corr"] >= min_template_corr and out["flat"] <= max_flat
                         and out["clip"] <= max_clip and np.isfinite(out["hr"]) and hr_range[0] <= out["hr"] <= hr_range[1])
    return out


def ecg_sqi(ecg: np.ndarray, fs: float, k_min: float = 5.0, p_range: tuple[float, float] = (0.5, 0.8),
            bas_min: float = 0.95, max_flat: float = 0.05) -> dict:
    """ECG quality: kSQI (kurtosis), pSQI (QRS band power ratio), basSQI (baseline power), flatline."""
    x = np.asarray(ecg, dtype=float)
    out = {"flat": _flatline_fraction(x, fs)}
    if np.std(x) < 1e-9:
        out.update(kSQI=np.nan, pSQI=np.nan, basSQI=np.nan, accept=False)
        return out
    out["kSQI"] = float(kurtosis(x))
    p5_15 = _band_power(x, fs, 5, 15)
    p5_40 = _band_power(x, fs, 5, 40)
    p0_1 = _band_power(x, fs, 0, 1)
    p0_40 = _band_power(x, fs, 0, 40)
    out["pSQI"] = float(p5_15 / p5_40) if p5_40 > 0 else np.nan
    out["basSQI"] = float(1 - p0_1 / p0_40) if p0_40 > 0 else np.nan
    out["accept"] = bool(out["kSQI"] >= k_min and p_range[0] <= out["pSQI"] <= p_range[1]
                         and out["basSQI"] >= bas_min and out["flat"] <= max_flat)
    return out


def abp_sqi(abp: np.ndarray, fs: float, sbp_range: tuple[float, float] = (40.0, 250.0),
            dbp_range: tuple[float, float] = (20.0, 180.0), map_range: tuple[float, float] = (30.0, 200.0),
            min_pp: float = 15.0, max_dpdt: float = 1500.0, max_beat_jump: float = 30.0,
            max_flat: float = 0.05, hr_range: tuple[float, float] = (30.0, 200.0)) -> dict:
    """ABP plausibility: pressure ranges, pulse pressure, max |dP/dt| (mmHg/s), beat-to-beat SBP jump."""
    x = np.asarray(abp, dtype=float)
    out = {"flat": _flatline_fraction(x, fs), "clip": _clipping_fraction(x)}
    b = abp_beats(x, fs)
    sbp, dbp, mp = b["sbp"], b["dbp"], b["map"]
    out["n_beats"] = int(len(sbp))
    if len(sbp) < 3:
        out.update(sbp=np.nan, dbp=np.nan, map=np.nan, pp=np.nan, dpdt_max=np.nan, max_jump=np.nan, hr=np.nan, accept=False)
        return out
    out["sbp"], out["dbp"], out["map"] = float(np.median(sbp)), float(np.median(dbp)), float(np.nanmedian(mp))
    out["pp"] = float(np.median(sbp - dbp))
    out["dpdt_max"] = float(np.max(np.abs(np.diff(x))) * fs)
    out["max_jump"] = float(np.max(np.abs(np.diff(sbp))))
    out["hr"] = float(60.0 * fs / np.median(np.diff(b["peak"])))
    ok = (sbp_range[0] <= np.min(sbp) and np.max(sbp) <= sbp_range[1]
          and dbp_range[0] <= np.min(dbp) and np.max(dbp) <= dbp_range[1]
          and map_range[0] <= out["map"] <= map_range[1]
          and out["pp"] >= min_pp and out["dpdt_max"] <= max_dpdt and out["max_jump"] <= max_beat_jump
          and out["flat"] <= max_flat and hr_range[0] <= out["hr"] <= hr_range[1])
    out["accept"] = bool(ok)
    return out


def window_quality(ecg: np.ndarray, ppg: np.ndarray, abp: np.ndarray, fs: float, max_hr_disagreement: float = 10.0) -> dict:
    """Combined SQI for one window. Also requires PPG- and ABP-derived heart rates to agree."""
    e, p, a = ecg_sqi(ecg, fs), ppg_sqi(ppg, fs), abp_sqi(abp, fs)
    hr_ok = np.isfinite(p["hr"]) and np.isfinite(a["hr"]) and abs(p["hr"] - a["hr"]) <= max_hr_disagreement
    return {"ecg": e, "ppg": p, "abp": a, "hr_agree": bool(hr_ok),
            "accept": bool(e["accept"] and p["accept"] and a["accept"] and hr_ok)}
