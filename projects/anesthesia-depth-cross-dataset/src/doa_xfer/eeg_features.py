"""Frontal-EEG features for depth of anaesthesia.

Per 4-s epoch (default): Welch PSD over 0.5-47 Hz; log absolute and relative
band powers (slow, delta, theta, alpha, beta, gamma); spectral edge frequency
95% (SEF95); spectral entropy; aperiodic exponent/offset from a robust log-log
fit with iterative peak exclusion (a FOOOF-lite; Donoghue et al., 2020; Lendner
et al., 2020); alpha peak power above the aperiodic fit; alpha/delta ratio;
burst-suppression ratio (Rampil, 1998: fraction of time with |x| below an
amplitude threshold for at least 0.5 s).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sps

BANDS: Dict[str, Tuple[float, float]] = {
    "slow": (0.1, 1.0), "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 47.0),
}
SIX_FEATURES: List[str] = ["log_alpha", "log_delta", "aperiodic_exponent", "sef95", "bsr", "alpha_delta_ratio"]


def welch_psd(epochs: np.ndarray, fs: float, fmax: float = 47.0) -> Tuple[np.ndarray, np.ndarray]:
    """PSD (n_epochs, n_freq) for 0 <= f <= fmax with 2-s Hann segments."""
    e = np.asarray(epochs, float)
    nperseg = min(e.shape[-1], int(2 * fs))
    f, P = sps.welch(e, fs=fs, nperseg=nperseg, axis=-1)
    sel = f <= fmax
    return f[sel], P[..., sel]


def aperiodic_fit(f: np.ndarray, P: np.ndarray, fmin: float = 2.0, fmax: float = 40.0, n_iter: int = 3,
                  excl_sd: float = 1.0) -> Tuple[float, float]:
    """Robust log-log line fit log10(P) = offset - exponent*log10(f), iteratively excluding peaks.

    Returns (exponent, offset).  Points more than ``excl_sd`` residual-SDs *above* the
    line are dropped at each iteration (peaks only), so oscillatory bumps do not bias
    the slope.
    """
    sel = (f >= fmin) & (f <= fmax) & (P > 0)
    x, y = np.log10(f[sel]), np.log10(P[sel])
    if x.size < 5:
        return float("nan"), float("nan")
    keep = np.ones(x.size, bool)
    slope, inter = 0.0, 0.0
    for _ in range(n_iter):
        A = np.column_stack([np.ones(keep.sum()), x[keep]])
        (inter, slope), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        resid = y - (inter + slope * x)
        sd = resid[keep].std() + 1e-12
        new_keep = resid <= excl_sd * sd
        if new_keep.sum() < 5 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(-slope), float(inter)


def burst_suppression_ratio(x: np.ndarray, fs: float, thr_uv: float = 5.0, min_dur_s: float = 0.5) -> float:
    """Fraction of the epoch spent in suppression (|x| < thr for >= min_dur_s)."""
    x = np.asarray(x, float)
    below = np.abs(x) < thr_uv
    m = np.r_[False, below, False].astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    L = int(min_dur_s * fs)
    sup = sum((e - s) for s, e in zip(starts, ends) if (e - s) >= L)
    return float(sup / x.size) if x.size else float("nan")


def epoch_features(epochs: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """Feature dictionary (each value: array of length n_epochs)."""
    e = np.asarray(epochs, float)
    if e.ndim == 1:
        e = e[None, :]
    f, P = welch_psd(e, fs)
    df = f[1] - f[0]
    tot = P[:, f >= 0.5].sum(-1) * df + 1e-12
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        sel = (f >= lo) & (f < hi)
        bp = P[:, sel].sum(-1) * df
        out[f"log_{name}"] = np.log(bp + 1e-12)
        out[f"rel_{name}"] = bp / tot
    sel = f >= 0.5
    cum = np.cumsum(P[:, sel], axis=-1)
    cum = cum / (cum[:, -1:] + 1e-12)
    out["sef95"] = f[sel][(cum >= 0.95).argmax(-1)]
    pn = P[:, sel] / (P[:, sel].sum(-1, keepdims=True) + 1e-12)
    out["spectral_entropy"] = -(pn * np.log(pn + 1e-12)).sum(-1) / np.log(pn.shape[1])
    expo = np.empty(e.shape[0])
    offs = np.empty(e.shape[0])
    alpha_peak = np.empty(e.shape[0])
    a_sel = (f >= 8) & (f <= 13)
    for i in range(e.shape[0]):
        expo[i], offs[i] = aperiodic_fit(f, P[i])
        if np.isfinite(expo[i]):
            fit = 10 ** (offs[i] - expo[i] * np.log10(np.maximum(f[a_sel], 1e-3)))
            alpha_peak[i] = np.log10(P[i, a_sel].max() + 1e-12) - np.log10(fit[np.argmax(P[i, a_sel])] + 1e-12)
        else:
            alpha_peak[i] = np.nan
    out["aperiodic_exponent"], out["aperiodic_offset"], out["alpha_peak_above_ap"] = expo, offs, alpha_peak
    out["alpha_delta_ratio"] = out["log_alpha"] - out["log_delta"]
    out["bsr"] = np.asarray([burst_suppression_ratio(row, fs) for row in e])
    return out


def features_to_matrix(feats: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    return np.column_stack([np.asarray(feats[n], float) for n in names])


def smooth_trajectory(x: np.ndarray, step_s: float, win_s: float = 30.0) -> np.ndarray:
    """Centered moving average of a per-epoch trajectory (NaN-aware)."""
    x = np.asarray(x, float)
    k = max(1, int(round(win_s / step_s)))
    v = np.where(np.isfinite(x), x, 0.0)
    w = np.isfinite(x).astype(float)
    num = np.convolve(v, np.ones(k), mode="same")
    den = np.convolve(w, np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
