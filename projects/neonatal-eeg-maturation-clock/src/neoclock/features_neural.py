"""Maturational neonatal EEG features (NEURAL-inspired; O'Toole & Boylan, 2017).

All functions take signals in microvolts.  Per-channel features are computed on
one epoch (typically 5 min or 1 h) and aggregated with the median across
channels by :func:`record_features`.

* Spectral: log absolute and relative power in delta1 (0.5-2), delta2 (2-4),
  theta (4-7), alpha (7-13), beta (13-30) Hz; spectral edge 90/95 %; log-log
  spectral slope over 2-20 Hz.
* rEEG (range EEG; O'Reilly et al., 2012): peak-to-peak amplitude in 2-s windows,
  summarized by the 5th/50th/95th percentiles and an asymmetry index.
* Burst / interburst structure: bursts are runs where a smoothed amplitude
  envelope exceeds ``thr_uv`` for at least ``min_burst_s``; interburst intervals
  (IBI) are the gaps; continuity = burst fraction.
* Interhemispheric synchrony: correlation of 2-s envelopes across homologous pairs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, signal as sps

BANDS: Dict[str, Tuple[float, float]] = {
    "delta1": (0.5, 2.0), "delta2": (2.0, 4.0), "theta": (4.0, 7.0), "alpha": (7.0, 13.0), "beta": (13.0, 30.0),
}


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 30.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    sos = sps.butter(order, [lo / nyq, min(hi, 0.95 * nyq) / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, float), axis=-1)


def spectral_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    x = np.asarray(x, float)
    nperseg = min(x.size, int(8 * fs))
    f, P = sps.welch(x, fs=fs, nperseg=nperseg)
    df = f[1] - f[0]
    sel = (f >= 0.5) & (f <= 30.0)
    tot = P[sel].sum() * df + 1e-12
    out: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        s = (f >= lo) & (f < hi)
        bp = P[s].sum() * df
        out[f"log_{name}"] = float(np.log(bp + 1e-12))
        out[f"rel_{name}"] = float(bp / tot)
    cum = np.cumsum(P[sel])
    cum = cum / (cum[-1] + 1e-12)
    out["sef90"] = float(f[sel][(cum >= 0.90).argmax()])
    out["sef95"] = float(f[sel][(cum >= 0.95).argmax()])
    s2 = (f >= 2.0) & (f <= 20.0) & (P > 0)
    if s2.sum() >= 5:
        slope = np.polyfit(np.log10(f[s2]), np.log10(P[s2]), 1)[0]
        out["spectral_slope"] = float(slope)
    else:
        out["spectral_slope"] = float("nan")
    return out


def amplitude_envelope(x: np.ndarray, fs: float, smooth_s: float = 0.5) -> np.ndarray:
    """Smoothed absolute amplitude of the 0.5-30 Hz signal (moving average over ``smooth_s``)."""
    xf = bandpass(x, fs)
    w = max(1, int(smooth_s * fs))
    return ndimage.uniform_filter1d(np.abs(xf), size=w, mode="nearest")


def reeg(x: np.ndarray, fs: float, win_s: float = 2.0) -> Dict[str, float]:
    """Range-EEG summary: percentiles of the 2-s peak-to-peak amplitude and asymmetry (Q95-Q50)/(Q50-Q5)."""
    xf = bandpass(x, fs)
    L = int(win_s * fs)
    n = xf.size // L
    if n < 2:
        return {"reeg_p5": np.nan, "reeg_p50": np.nan, "reeg_p95": np.nan, "reeg_asym": np.nan}
    seg = xf[: n * L].reshape(n, L)
    ptp = seg.max(1) - seg.min(1)
    p5, p50, p95 = np.percentile(ptp, [5, 50, 95])
    return {"reeg_p5": float(p5), "reeg_p50": float(p50), "reeg_p95": float(p95),
            "reeg_asym": float((p95 - p50) / (p50 - p5 + 1e-9))}


def burst_mask(x: np.ndarray, fs: float, thr_uv: float = 15.0, min_burst_s: float = 1.0, min_gap_s: float = 0.5) -> np.ndarray:
    """Boolean per-sample burst mask from the amplitude envelope (short gaps are bridged, short bursts dropped)."""
    env = amplitude_envelope(x, fs)
    m = env > thr_uv
    m = ndimage.binary_closing(m, structure=np.ones(max(1, int(min_gap_s * fs))))
    m = ndimage.binary_opening(m, structure=np.ones(max(1, int(min_burst_s * fs))))
    return m


def interburst_stats(mask: np.ndarray, fs: float) -> Dict[str, float]:
    """Continuity (burst fraction), IBI median/95th percentile/max (s), bursts per minute."""
    m = np.r_[False, np.asarray(mask, bool), False].astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    n_bursts = starts.size
    if n_bursts >= 2:
        ibi = (starts[1:] - ends[:-1]) / fs
        ibi = ibi[ibi > 0]
    else:
        ibi = np.array([])
    dur_min = mask.size / fs / 60.0
    return {
        "continuity": float(np.mean(mask)),
        "ibi_median_s": float(np.median(ibi)) if ibi.size else 0.0,
        "ibi_p95_s": float(np.percentile(ibi, 95)) if ibi.size else 0.0,
        "ibi_max_s": float(ibi.max()) if ibi.size else 0.0,
        "bursts_per_min": float(n_bursts / dur_min) if dur_min > 0 else np.nan,
    }


def interhemispheric_synchrony(xl: np.ndarray, xr: np.ndarray, fs: float, win_s: float = 2.0) -> float:
    """Mean correlation of 2-s amplitude envelopes between a homologous channel pair."""
    el, er = amplitude_envelope(xl, fs), amplitude_envelope(xr, fs)
    L = int(win_s * fs)
    n = min(el.size, er.size) // L
    if n < 2:
        return float("nan")
    a = el[: n * L].reshape(n, L).mean(1)
    b = er[: n * L].reshape(n, L).mean(1)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def channel_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    out = spectral_features(x, fs)
    out.update(reeg(x, fs))
    out.update(interburst_stats(burst_mask(x, fs), fs))
    return out


def record_features(X: np.ndarray, fs: float, channel_names: Optional[Sequence[str]] = None,
                    homologous_pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, float]:
    """Median across channels of per-channel features plus mean interhemispheric synchrony."""
    X = np.asarray(X, float)
    per = [channel_features(X[c], fs) for c in range(X.shape[0])]
    keys = list(per[0].keys())
    out = {k: float(np.nanmedian([p[k] for p in per])) for k in keys}
    if homologous_pairs:
        syn = [interhemispheric_synchrony(X[i], X[j], fs) for i, j in homologous_pairs]
        out["synchrony"] = float(np.nanmean(syn))
    return out


def feature_names(include_synchrony: bool = True) -> List[str]:
    names = list(spectral_features(np.random.default_rng(0).normal(size=2048), 256.0).keys())
    names += ["reeg_p5", "reeg_p50", "reeg_p95", "reeg_asym", "continuity", "ibi_median_s", "ibi_p95_s", "ibi_max_s",
              "bursts_per_min"]
    if include_synchrony:
        names.append("synchrony")
    return names


def features_to_matrix(rows: Sequence[Dict[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[r.get(n, np.nan) for n in names] for r in rows], float)
