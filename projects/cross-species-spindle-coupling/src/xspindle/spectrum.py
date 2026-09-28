"""Spectral parameterisation: Welch PSD, aperiodic fit and individualised spindle-peak detection.

A minimal 'specparam-lite': the aperiodic component is fitted as log10 P(f) = b - chi * log10 f over a fit
range (optionally excluding candidate peak bands), and peaks are read from the residual. This is enough to
individualise the spindle band per recording without depending on the full specparam package.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch


def welch_psd(x: np.ndarray, fs: float, nperseg_s: float = 4.0, fmax: Optional[float] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with Hann windows of ``nperseg_s`` seconds (50% overlap)."""
    nperseg = int(round(nperseg_s * fs))
    f, p = welch(np.asarray(x, dtype=float), fs=fs, nperseg=min(nperseg, len(x)))
    if fmax is not None:
        keep = f <= fmax
        f, p = f[keep], p[keep]
    return f, p


def aperiodic_fit(f: np.ndarray, pxx: np.ndarray, fit_range: Tuple[float, float] = (1.0, 40.0),
                  exclude: Sequence[Tuple[float, float]] = ()) -> Dict[str, object]:
    """Fit log10 P = b - chi log10 f by least squares; return offset, exponent and the residual over all f.

    ``exclude`` lists (lo, hi) bands ignored during the fit (e.g., the expected spindle band) so that a large
    peak does not bias the slope.
    """
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    keep = (f >= fit_range[0]) & (f <= fit_range[1]) & (pxx > 0)
    for lo, hi in exclude:
        keep &= ~((f >= lo) & (f <= hi))
    lf = np.log10(f[keep])
    lp = np.log10(pxx[keep])
    A = np.column_stack([np.ones_like(lf), -lf])
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    b, chi = float(coef[0]), float(coef[1])
    with np.errstate(divide="ignore"):
        fit = b - chi * np.log10(np.where(f > 0, f, np.nan))
        resid = np.log10(np.where(pxx > 0, pxx, np.nan)) - fit
    return {"offset": b, "exponent": chi, "residual": resid, "fit": fit}


def find_spectral_peak(f: np.ndarray, residual: np.ndarray, search_range: Tuple[float, float],
                       min_height: float = 0.3) -> Dict[str, float]:
    """Highest residual peak within ``search_range``; returns NaNs if below ``min_height`` (log10 units)."""
    f = np.asarray(f, dtype=float)
    r = np.asarray(residual, dtype=float)
    sel = (f >= search_range[0]) & (f <= search_range[1]) & np.isfinite(r)
    if not sel.any():
        return {"peak_freq": np.nan, "peak_height": np.nan, "found": 0.0}
    idx = np.flatnonzero(sel)
    i = idx[np.argmax(r[idx])]
    # require a local maximum (not the edge of the search window) and a minimum height
    interior = (i > idx[0]) and (i < idx[-1]) and r[i] >= r[i - 1] and r[i] >= r[i + 1]
    if not interior or r[i] < min_height:
        return {"peak_freq": np.nan, "peak_height": float(r[i]), "found": 0.0}
    # parabolic interpolation for sub-bin precision
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    denom = (y0 - 2 * y1 + y2)
    shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    df = f[1] - f[0]
    return {"peak_freq": float(f[i] + shift * df), "peak_height": float(y1), "found": 1.0}


def sigma_peak(x: np.ndarray, fs: float, search_range: Tuple[float, float] = (8.0, 18.0),
               fit_range: Tuple[float, float] = (1.0, 40.0), nperseg_s: float = 4.0,
               min_height: float = 0.3) -> Dict[str, float]:
    """Individualised spindle frequency of an NREM signal: aperiodic-corrected peak in ``search_range``.

    ``min_height`` is in log10 units above the aperiodic fit (0.3 = twice the 1/f power); pure 1/f noise
    produces residual fluctuations of ~ +/- 0.15 with 4-s Welch segments, so 0.3 rejects them."""
    f, p = welch_psd(x, fs, nperseg_s=nperseg_s, fmax=min(fit_range[1], fs / 2))
    ap = aperiodic_fit(f, p, fit_range=fit_range, exclude=[search_range, (0.0, 2.0)])
    peak = find_spectral_peak(f, ap["residual"], search_range, min_height=min_height)
    peak["aperiodic_exponent"] = ap["exponent"]
    return peak


def pink_noise(n: int, fs: float, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent noise via spectral shaping (unit variance). Used by tests and simulations."""
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.ones_like(freqs)
    amp[1:] = 1.0 / np.power(freqs[1:], exponent / 2.0)
    amp[0] = 0.0
    spec = amp * (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs)))
    x = np.fft.irfft(spec, n=n)
    return (x - x.mean()) / (x.std() + 1e-12)
