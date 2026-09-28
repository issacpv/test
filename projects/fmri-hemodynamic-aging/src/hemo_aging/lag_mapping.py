"""Systemic low-frequency-oscillation (sLFO) blood-arrival lag mapping.

A lightweight NumPy re-implementation of the core of the rapidtide algorithm
(Tong, Hocke & Frederick, 2019, Front. Neurosci.): band-pass the data, cross-correlate
each voxel/parcel time series against a reference regressor over a bounded lag range,
refine the reference by time-shifting and averaging well-correlated voxels, iterate.

Conventions
-----------
* ``data`` is ``(n_series, n_timepoints)``; ``tr`` in seconds.
* A positive lag means the voxel signal *follows* the reference (blood arrives later).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class LagResult:
    """Output of :func:`estimate_lag_map`."""

    lag: np.ndarray        # seconds, shape (n_series,)
    corr: np.ndarray       # maximum cross-correlation, shape (n_series,)
    reference: np.ndarray  # final refined reference regressor, shape (n_timepoints,)
    n_passes: int


def bandpass(data: np.ndarray, tr: float, low: float = 0.009, high: float = 0.15, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (frequencies in Hz)."""
    nyq = 0.5 / tr
    high = min(high, 0.95 * nyq)
    sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def zscore(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """Standardize along ``axis``; constant series become zeros instead of NaN."""
    mu = data.mean(axis=axis, keepdims=True)
    sd = data.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (data - mu) / sd


def shift_series(ts: np.ndarray, shift_seconds: float, tr: float) -> np.ndarray:
    """Shift a series in time by ``shift_seconds`` (positive = delay) with linear interpolation.

    Values outside the recorded window are filled with the series edge value.
    """
    t = np.arange(ts.shape[-1]) * tr
    return np.interp(t - shift_seconds, t, ts, left=ts[0], right=ts[-1])


def crosscorr_lag(x: np.ndarray, reference: np.ndarray, tr: float, max_lag: float = 10.0,
                  min_lag: float | None = None) -> tuple[float, float]:
    """Lag (s) and peak normalized cross-correlation of ``x`` relative to ``reference``.

    Integer-sample cross-correlation followed by parabolic interpolation of the peak.
    Positive lag = ``x`` lags the reference.
    """
    min_lag = -max_lag if min_lag is None else min_lag
    x = zscore(np.asarray(x, dtype=float))
    r = zscore(np.asarray(reference, dtype=float))
    n = x.size
    full = signal.correlate(x, r, mode="full") / n
    lags = np.arange(-n + 1, n) * tr
    keep = (lags >= min_lag) & (lags <= max_lag)
    full, lags = full[keep], lags[keep]
    k = int(np.argmax(full))
    if 0 < k < full.size - 1:
        y0, y1, y2 = full[k - 1], full[k], full[k + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
        delta = float(np.clip(delta, -1, 1))
        lag = lags[k] + delta * tr
        peak = y1 - 0.25 * (y0 - y2) * delta
    else:
        lag, peak = lags[k], full[k]
    return float(lag), float(peak)


def estimate_lag_map(data: np.ndarray, tr: float, reference: np.ndarray | None = None,
                     max_lag: float = 10.0, min_lag: float | None = None, n_passes: int = 3,
                     corr_threshold: float = 0.3, do_filter: bool = True) -> LagResult:
    """Iteratively refined sLFO lag map.

    Parameters
    ----------
    data : (n_series, n_timepoints) array of BOLD time series (voxels or parcels).
    tr : repetition time in seconds.
    reference : optional initial regressor; default is the mean of all series.
    max_lag, min_lag : search window in seconds (``min_lag`` defaults to ``-max_lag``).
    n_passes : number of refinement passes (rapidtide default is 3).
    corr_threshold : series with peak correlation above this contribute to the refined reference.
    do_filter : band-pass data to the sLFO band before correlation.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be (n_series, n_timepoints)")
    if do_filter:
        data = bandpass(data, tr)
    data = zscore(data)
    ref = data.mean(axis=0) if reference is None else zscore(np.asarray(reference, dtype=float))
    if do_filter and reference is not None:
        ref = zscore(bandpass(ref, tr))
    lag = np.zeros(data.shape[0])
    corr = np.zeros(data.shape[0])
    for p in range(n_passes):
        for i in range(data.shape[0]):
            lag[i], corr[i] = crosscorr_lag(data[i], ref, tr, max_lag=max_lag, min_lag=min_lag)
        good = corr > corr_threshold
        if p == n_passes - 1 or good.sum() < 2:
            break
        # realign each well-correlated series to the reference and average
        aligned = np.stack([shift_series(data[i], -lag[i], tr) for i in np.flatnonzero(good)])
        ref = zscore(aligned.mean(axis=0))
    return LagResult(lag=lag, corr=corr, reference=ref, n_passes=p + 1)


def regress_out_slfo(data: np.ndarray, result: LagResult, tr: float) -> np.ndarray:
    """Remove the lagged sLFO regressor from every series (voxel-specific delay)."""
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        reg = shift_series(result.reference, result.lag[i], tr)
        X = np.column_stack([np.ones_like(reg), reg])
        beta, *_ = np.linalg.lstsq(X, data[i], rcond=None)
        out[i] = data[i] - X[:, 1] * beta[1]
    return out


def lag_summaries(lag: np.ndarray, corr: np.ndarray, corr_threshold: float = 0.3,
                  extreme_fraction: float = 0.1) -> dict[str, float]:
    """Scalar features of a lag map used as aging / vascular-risk markers.

    Returns the interquartile range, skewness, the arteriovenous gradient
    (median of the latest ``extreme_fraction`` minus median of the earliest), the
    fraction of series with a usable sLFO fit, and the median lag among usable series.
    """
    lag, corr = np.asarray(lag, float), np.asarray(corr, float)
    ok = corr > corr_threshold
    if ok.sum() < 5:
        return {k: float("nan") for k in ("iqr", "skew", "av_gradient", "fit_fraction", "median")} | {
            "fit_fraction": float(ok.mean())}
    l = np.sort(lag[ok])
    q25, q50, q75 = np.percentile(l, [25, 50, 75])
    k = max(1, int(extreme_fraction * l.size))
    m3 = np.mean((l - l.mean()) ** 3)
    sd = l.std()
    return {
        "iqr": float(q75 - q25),
        "skew": float(m3 / sd**3) if sd > 0 else 0.0,
        "av_gradient": float(np.median(l[-k:]) - np.median(l[:k])),
        "fit_fraction": float(ok.mean()),
        "median": float(q50),
    }


def parcellate(data: np.ndarray, labels: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    """Average ``(n_voxels, n_time)`` series within integer parcel ``labels`` (0 = background)."""
    labels = np.asarray(labels).ravel()
    n_labels = int(labels.max()) if n_labels is None else n_labels
    out = np.full((n_labels, data.shape[1]), np.nan)
    for k in range(1, n_labels + 1):
        m = labels == k
        if m.any():
            out[k - 1] = data[m].mean(axis=0)
    return out
