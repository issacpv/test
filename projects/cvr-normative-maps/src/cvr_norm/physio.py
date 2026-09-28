"""Physiological regressors for breath-hold CVR mapping.

* :func:`end_tidal_co2` – exhalation-end peaks of a CO2 trace, interpolated to a PetCO2 time course.
* :func:`rvt` – respiration volume per time from a belt trace (Birn et al., 2006, NeuroImage).
* :func:`breath_hold_compliance` – fraction of hold blocks in which breathing actually stopped.
* :func:`boxcar`, :func:`canonical_hrf`, :func:`convolve_hrf`, :func:`resample_to_tr` – task regressors.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy import stats as sps


def end_tidal_co2(co2: np.ndarray, fs: float, min_breath_interval: float = 1.5, prominence: float | None = None,
                  out_times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect end-tidal peaks of a CO2 trace and interpolate a PetCO2 time course.

    Parameters
    ----------
    co2 : CO2 trace (mmHg or arbitrary units), sampled at ``fs`` Hz.
    min_breath_interval : minimum time between peaks (s).
    prominence : minimum peak prominence (default 25 % of the trace's 5-95 % range).
    out_times : times (s) at which to evaluate PetCO2; default = every sample.

    Returns
    -------
    (peak_times_s, peak_values, petco2_at_out_times)
    """
    co2 = np.asarray(co2, float)
    if prominence is None:
        lo, hi = np.percentile(co2, [5, 95])
        prominence = 0.25 * (hi - lo)
    peaks, _ = signal.find_peaks(co2, distance=max(1, int(min_breath_interval * fs)), prominence=prominence)
    if peaks.size < 2:
        raise ValueError("fewer than two end-tidal peaks detected; check units / prominence")
    t_peaks = peaks / fs
    v_peaks = co2[peaks]
    t_out = np.arange(co2.size) / fs if out_times is None else np.asarray(out_times, float)
    petco2 = np.interp(t_out, t_peaks, v_peaks, left=v_peaks[0], right=v_peaks[-1])
    return t_peaks, v_peaks, petco2


def rvt(resp: np.ndarray, fs: float, min_breath_interval: float = 1.5,
        out_times: np.ndarray | None = None) -> np.ndarray:
    """Respiration volume per time: (peak - trough amplitude) / breath period, interpolated to ``out_times``."""
    resp = np.asarray(resp, float)
    dist = max(1, int(min_breath_interval * fs))
    pk, _ = signal.find_peaks(resp, distance=dist)
    tr_, _ = signal.find_peaks(-resp, distance=dist)
    if pk.size < 2 or tr_.size < 1:
        raise ValueError("could not find breaths in the respiration trace")
    t = np.arange(resp.size) / fs
    trough_vals = np.interp(t[pk], t[tr_], resp[tr_])
    amp = resp[pk] - trough_vals
    period = np.gradient(t[pk]) if pk.size > 2 else np.diff(t[pk], prepend=t[pk][0] - min_breath_interval)
    period = np.clip(period, 1e-3, None)
    r = amp / period
    t_out = t if out_times is None else np.asarray(out_times, float)
    return np.interp(t_out, t[pk], r, left=r[0], right=r[-1])


def breath_hold_compliance(resp: np.ndarray, fs: float, onsets: np.ndarray, durations: np.ndarray,
                           baseline_window: float = 10.0, ratio: float = 0.25) -> tuple[float, np.ndarray]:
    """Fraction of hold blocks where the belt variance falls below ``ratio`` × pre-hold baseline variance.

    Returns ``(fraction_compliant, per_block_ratio)``.
    """
    resp = np.asarray(resp, float)
    ratios = []
    for on, dur in zip(onsets, durations):
        b0, b1 = int(max(0, (on - baseline_window) * fs)), int(on * fs)
        h0, h1 = int(on * fs), int(min(resp.size, (on + dur) * fs))
        base = resp[b0:b1].var() if b1 > b0 + 2 else np.nan
        hold = resp[h0:h1].var() if h1 > h0 + 2 else np.nan
        ratios.append(hold / base if base and np.isfinite(base) else np.nan)
    ratios_arr = np.asarray(ratios, float)
    ok = ratios_arr < ratio
    return float(np.nanmean(ok)) if ratios_arr.size else float("nan"), ratios_arr


def boxcar(n_samples: int, fs: float, onsets: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """0/1 block regressor sampled at ``fs`` Hz."""
    x = np.zeros(n_samples)
    for on, dur in zip(onsets, durations):
        x[int(on * fs):int((on + dur) * fs)] = 1.0
    return x


def canonical_hrf(fs: float, duration: float = 32.0, peak_delay: float = 6.0, undershoot_delay: float = 16.0,
                  ratio: float = 6.0) -> np.ndarray:
    """SPM-style double-gamma HRF sampled at ``fs`` Hz, unit area."""
    t = np.arange(0, duration, 1.0 / fs)
    h = sps.gamma.pdf(t, peak_delay) - sps.gamma.pdf(t, undershoot_delay) / ratio
    return h / h.sum()


def convolve_hrf(regressor: np.ndarray, fs: float, **hrf_kwargs) -> np.ndarray:
    """Causal convolution with the canonical HRF, same length as ``regressor``."""
    h = canonical_hrf(fs, **hrf_kwargs)
    return np.convolve(regressor, h)[: regressor.size]


def resample_to_tr(x: np.ndarray, fs: float, tr: float, n_vols: int, shift: float = 0.0,
                   slice_time_offset: float = 0.0) -> np.ndarray:
    """Sample a fine-resolution regressor at volume acquisition times ``k*tr + offset - shift``.

    Positive ``shift`` delays the regressor (the BOLD response to a CO2 change arrives later).
    """
    t_fine = np.arange(x.size) / fs
    t_vol = np.arange(n_vols) * tr + slice_time_offset - shift
    return np.interp(t_vol, t_fine, x, left=x[0], right=x[-1])
