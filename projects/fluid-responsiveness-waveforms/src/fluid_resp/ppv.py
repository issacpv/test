"""Windowed pulse-pressure variation (PPV), systolic-pressure variation (SPV) and PP-derived respiratory rate.

PPV follows the usual definition (Michard & Teboul): over one respiratory cycle,
``PPV = 100 * (PPmax - PPmin) / ((PPmax + PPmin) / 2)``.  Because the respiratory period is not known beat by
beat, PPV is computed over sliding windows of ~8 s (>= 2 respiratory cycles at 15/min) stepped by 2 s and then
summarised by the median over the analysis window, as automated monitors do.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _windowed_variation(t: np.ndarray, v: np.ndarray, window_s: float, step_s: float, min_beats: int, relative: bool) -> pd.DataFrame:
    if t.size == 0:
        return pd.DataFrame(columns=["t_center", "value", "n_beats"])
    t0, t1 = float(t.min()), float(t.max())
    rows = []
    start = t0
    while start + window_s <= t1 + 1e-9:
        sel = (t >= start) & (t < start + window_s)
        n = int(sel.sum())
        if n >= min_beats:
            vmax, vmin = float(v[sel].max()), float(v[sel].min())
            val = 100.0 * (vmax - vmin) / ((vmax + vmin) / 2.0) if relative else (vmax - vmin)
        else:
            val = np.nan
        rows.append({"t_center": start + window_s / 2.0, "value": val, "n_beats": n})
        start += step_s
    return pd.DataFrame(rows)


def ppv_series(t_beats: np.ndarray, pp: np.ndarray, window_s: float = 8.0, step_s: float = 2.0, min_beats: int = 4) -> pd.DataFrame:
    """PPV (%) per sliding window. Columns: ``t_center, ppv_pct, n_beats``."""
    d = _windowed_variation(np.asarray(t_beats, float), np.asarray(pp, float), window_s, step_s, min_beats, relative=True)
    return d.rename(columns={"value": "ppv_pct"})


def spv_series(t_beats: np.ndarray, sbp: np.ndarray, window_s: float = 8.0, step_s: float = 2.0, min_beats: int = 4) -> pd.DataFrame:
    """SPV (mmHg) per sliding window. Columns: ``t_center, spv_mmhg, n_beats``."""
    d = _windowed_variation(np.asarray(t_beats, float), np.asarray(sbp, float), window_s, step_s, min_beats, relative=False)
    return d.rename(columns={"value": "spv_mmhg"})


def summarise_index(series: pd.DataFrame, col: str, min_fraction_valid: float = 0.8) -> dict:
    """Median / IQR of a windowed index; NaN when fewer than ``min_fraction_valid`` of windows are valid."""
    v = series[col].to_numpy(dtype=float)
    valid = np.isfinite(v)
    frac = float(valid.mean()) if v.size else 0.0
    if v.size == 0 or frac < min_fraction_valid:
        return {"median": np.nan, "iqr": np.nan, "fraction_valid": frac, "n_windows": int(v.size)}
    return {"median": float(np.median(v[valid])), "iqr": float(np.subtract(*np.percentile(v[valid], [75, 25]))), "fraction_valid": frac, "n_windows": int(v.size)}


def ppv_from_beats(beats: pd.DataFrame, mask: Optional[np.ndarray] = None, **kw) -> dict:
    """Convenience: apply a quality mask, compute the PPV series and summarise it."""
    b = beats if mask is None else beats[np.asarray(mask, dtype=bool)]
    s = ppv_series(b["t_sys"].to_numpy(), b["pp"].to_numpy(), **kw)
    return summarise_index(s, "ppv_pct")


def respiratory_rate_from_pp(t_beats: np.ndarray, pp: np.ndarray, fs_resample: float = 4.0, band: tuple = (0.1, 0.7)) -> float:
    """Dominant frequency (breaths/min) of the beat-to-beat PP series within ``band`` Hz.

    Resamples the irregular PP series to ``fs_resample`` Hz by linear interpolation, detrends and takes the
    FFT peak. Returns NaN for fewer than 20 beats.
    """
    t = np.asarray(t_beats, float)
    v = np.asarray(pp, float)
    if t.size < 20:
        return float("nan")
    tt = np.arange(t.min(), t.max(), 1.0 / fs_resample)
    x = np.interp(tt, t, v)
    x = x - np.polyval(np.polyfit(tt, x, 1), tt)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_resample)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any():
        return float("nan")
    return float(freqs[sel][np.argmax(spec[sel])] * 60.0)
