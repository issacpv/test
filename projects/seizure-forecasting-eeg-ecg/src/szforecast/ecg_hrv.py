"""ECG R-peak detection and windowed heart-rate-variability (HRV) features.

The detector is a Pan-Tompkins-style pipeline (band-pass, derivative, squaring,
moving-window integration, adaptive peak picking) implemented with scipy only,
followed by refinement of each peak on the band-passed signal.  HRV features
follow the Task Force (1996) conventions: time domain (mean HR, SDNN, RMSSD,
pNN50), frequency domain from a Lomb-Scargle periodogram of the unevenly
sampled RR series (LF 0.04-0.15 Hz, HF 0.15-0.40 Hz), plus sample entropy.
A signal-quality index (fraction of beats rejected by the RR cleaning rules)
is returned so that low-quality windows can be excluded downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal as sps

RR_MIN_S = 0.3
RR_MAX_S = 2.0


def bandpass_ecg(ecg: np.ndarray, fs: float, lo: float = 5.0, hi: float = 30.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass used both for detection and peak refinement."""
    nyq = 0.5 * fs
    hi = min(hi, 0.95 * nyq)
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(ecg, dtype=float))


def detect_r_peaks(ecg: np.ndarray, fs: float, refractory_s: float = 0.30, integ_win_s: float = 0.15) -> np.ndarray:
    """Return sample indices of R peaks.

    Steps: band-pass 5-30 Hz -> derivative -> square -> moving-window integration
    (``integ_win_s``) -> ``scipy.signal.find_peaks`` with a refractory distance and an
    adaptive prominence threshold (30% of the 98th percentile of the integrated signal,
    recomputed in 10-s blocks so that amplitude drifts do not drop beats) -> refine each
    detection to the largest absolute deflection of the band-passed signal within +/-50 ms.
    """
    x = bandpass_ecg(ecg, fs)
    d = np.gradient(x)
    sq = d ** 2
    w = max(1, int(round(integ_win_s * fs)))
    integ = np.convolve(sq, np.ones(w) / w, mode="same")

    block = int(10 * fs)
    thr = np.empty_like(integ)
    for s in range(0, integ.size, block):
        seg = integ[s:s + block]
        thr[s:s + block] = 0.3 * np.percentile(seg, 98) if seg.size else 0.0
    cand, _ = sps.find_peaks(integ, distance=max(1, int(refractory_s * fs)), height=thr)

    half = int(0.05 * fs)
    peaks = []
    for c in cand:
        lo, hi = max(0, c - half), min(x.size, c + half + 1)
        peaks.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    peaks = np.unique(np.asarray(peaks, dtype=int))
    if peaks.size > 1:  # enforce refractory after refinement
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= refractory_s * fs:
                keep.append(p)
            elif abs(x[p]) > abs(x[keep[-1]]):
                keep[-1] = p
        peaks = np.asarray(keep)
    return peaks


@dataclass
class RRSeries:
    """Cleaned RR-interval series: beat times (s, of the *second* beat of each interval), RR (s), and mask."""
    t: np.ndarray
    rr: np.ndarray
    valid: np.ndarray

    @property
    def sqi(self) -> float:
        """Fraction of intervals rejected by the cleaning rules (0 = perfect)."""
        return float(1.0 - self.valid.mean()) if self.valid.size else 1.0


def rr_from_peaks(peaks: np.ndarray, fs: float, max_dev: float = 0.2, med_win: int = 5) -> RRSeries:
    """Convert R-peak indices to a cleaned RR series.

    Intervals outside [0.3, 2.0] s or deviating by more than ``max_dev`` (fraction)
    from a running median of ``med_win`` beats are flagged invalid and linearly
    interpolated (so that spectral features see a continuous series), but the
    ``valid`` mask is kept for the SQI and for time-domain features.
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size < 3:
        return RRSeries(np.array([]), np.array([]), np.array([], dtype=bool))
    t = peaks[1:] / fs
    rr = np.diff(peaks) / fs
    valid = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    med = sps.medfilt(rr, kernel_size=med_win if med_win % 2 else med_win + 1)
    valid &= np.abs(rr - med) <= max_dev * np.maximum(med, 1e-6)
    rr_clean = rr.copy()
    if valid.sum() >= 2 and (~valid).any():
        rr_clean[~valid] = np.interp(t[~valid], t[valid], rr[valid])
    return RRSeries(t=t, rr=rr_clean, valid=valid)


def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """Sample entropy (Richman & Moorman, 2000) of a 1-D series; ``r`` defaults to 0.2*SD."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < m + 2:
        return float("nan")
    r = 0.2 * np.std(x) if r is None else r
    if r <= 0:
        return float("nan")

    def _count(mm: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=-1)
        c = (d <= r).sum() - emb.shape[0]  # remove self-matches
        return int(c)

    b, a = _count(m), _count(m + 1)
    if a == 0 or b == 0:
        return float("nan")
    return float(-np.log(a / b))


def hrv_features(rr: RRSeries, f_lf=(0.04, 0.15), f_hf=(0.15, 0.40)) -> Dict[str, float]:
    """HRV feature dictionary for one window given a cleaned RR series."""
    out = {k: np.nan for k in ("mean_hr", "sdnn", "rmssd", "pnn50", "lf", "hf", "lf_hf", "total_power",
                                "sampen", "n_beats", "sqi")}
    if rr.rr.size < 8:
        out["n_beats"], out["sqi"] = float(rr.rr.size), rr.sqi
        return out
    r = rr.rr[rr.valid] if rr.valid.sum() >= 8 else rr.rr
    out["mean_hr"] = 60.0 / np.mean(r)
    out["sdnn"] = 1000.0 * np.std(r, ddof=1)
    dr = np.diff(r)
    out["rmssd"] = 1000.0 * np.sqrt(np.mean(dr ** 2))
    out["pnn50"] = float(np.mean(np.abs(dr) > 0.05))
    # Lomb-Scargle on the detrended, unevenly sampled RR series
    y = rr.rr - rr.rr.mean()
    freqs = np.linspace(0.01, 0.5, 200)
    pgram = sps.lombscargle(rr.t, y, 2 * np.pi * freqs, normalize=False)
    pgram = pgram * 2.0 / y.size * 1e6  # ms^2 per Hz-ish scale
    df = freqs[1] - freqs[0]
    lf = pgram[(freqs >= f_lf[0]) & (freqs < f_lf[1])].sum() * df
    hf = pgram[(freqs >= f_hf[0]) & (freqs < f_hf[1])].sum() * df
    out["lf"], out["hf"] = float(lf), float(hf)
    out["lf_hf"] = float(lf / hf) if hf > 0 else np.nan
    out["total_power"] = float(pgram[(freqs >= 0.003) & (freqs < 0.4)].sum() * df)
    out["sampen"] = sample_entropy(r[:600])
    out["n_beats"] = float(rr.rr.size)
    out["sqi"] = rr.sqi
    return out


def sliding_hrv(ecg: np.ndarray, fs: float, win_s: float = 300.0, step_s: float = 60.0) -> pd.DataFrame:
    """Windowed HRV features over a whole ECG channel.

    Returns a DataFrame with one row per window (``t_start`` in seconds) and the
    columns of :func:`hrv_features`.  R peaks are detected once on the full signal.
    """
    peaks = detect_r_peaks(ecg, fs)
    n = np.asarray(ecg).size
    rows = []
    starts = np.arange(0.0, max(0.0, n / fs - win_s) + 1e-9, step_s)
    for s in starts:
        lo, hi = int(s * fs), int((s + win_s) * fs)
        pk = peaks[(peaks >= lo) & (peaks < hi)] - lo
        feats = hrv_features(rr_from_peaks(pk, fs))
        feats["t_start"] = float(s)
        rows.append(feats)
    return pd.DataFrame(rows)
