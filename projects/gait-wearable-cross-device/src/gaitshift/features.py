"""Gait features from acceleration and vertical ground-reaction force, plus a synthetic gait generator."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 2.x renamed trapz


def synthesize_gait(
    fs: float = 100.0,
    duration_s: float = 20.0,
    step_freq_hz: float = 1.8,
    amp_m_s2: float = 2.0,
    asymmetry: float = 0.0,
    noise: float = 0.2,
    fog_segments: list[tuple[float, float]] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic ``[T, 3]`` accelerometer signal (m/s^2) with gravity on the third axis.

    The vertical channel carries the step-frequency fundamental and its
    second harmonic (stride); ``asymmetry`` adds a stride-frequency
    component (left/right difference). ``fog_segments`` (start, end in s)
    replace locomotion with 3-8 Hz trembling.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    stride = step_freq_hz / 2.0
    vert = amp_m_s2 * np.sin(2 * np.pi * step_freq_hz * t) + 0.3 * amp_m_s2 * np.sin(4 * np.pi * step_freq_hz * t)
    vert += asymmetry * amp_m_s2 * np.sin(2 * np.pi * stride * t)
    ap = 0.5 * amp_m_s2 * np.sin(2 * np.pi * step_freq_hz * t + 0.5)
    ml = 0.3 * amp_m_s2 * np.sin(2 * np.pi * stride * t)
    x = np.column_stack([ap, ml, vert + 9.80665])
    for (a, b) in fog_segments or []:
        m = (t >= a) & (t < b)
        trem = 0.8 * amp_m_s2 * np.sin(2 * np.pi * 6.0 * t[m])
        x[m, 0] = 0.2 * trem
        x[m, 1] = 0.2 * trem
        x[m, 2] = 9.80665 + trem
    return x + noise * rng.normal(size=x.shape)


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = welch(np.asarray(x, float), fs=fs, nperseg=min(len(x), int(4 * fs)))
    m = (f >= lo) & (f < hi)
    return float(_trapz(p[m], f[m])) if m.any() else 0.0


def freeze_index(x: np.ndarray, fs: float) -> float:
    """Moore et al. (2008) freeze index: power in 3-8 Hz over power in 0.5-3 Hz."""
    loco = band_power(x, fs, 0.5, 3.0)
    freeze = band_power(x, fs, 3.0, 8.0)
    return float(freeze / max(loco, 1e-12))


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    return ac / max(ac[0], 1e-12)


def cadence_hz(x: np.ndarray, fs: float, lo_hz: float = 1.2, hi_hz: float = 3.0) -> float:
    """Step frequency from the first dominant autocorrelation peak in the step-lag range."""
    ac = _autocorr(x)
    lo, hi = int(fs / hi_hz), int(fs / lo_hz)
    seg = ac[lo:hi + 1]
    if len(seg) == 0:
        return float("nan")
    peaks, props = find_peaks(seg, height=0.0)
    if len(peaks) == 0:
        return float("nan")
    best = peaks[np.argmax(props["peak_heights"])]
    return float(fs / (lo + best))


def stride_regularity(x: np.ndarray, fs: float, step_freq_hz: float | None = None) -> dict[str, float]:
    """Step and stride regularity (autocorrelation at step and stride lags; Moe-Nilssen & Helbostad, 2004)."""
    ac = _autocorr(x)
    sf = step_freq_hz or cadence_hz(x, fs)
    if not np.isfinite(sf) or sf <= 0:
        return {"step_regularity": np.nan, "stride_regularity": np.nan, "symmetry": np.nan}
    step_lag = int(round(fs / sf))
    stride_lag = 2 * step_lag
    if stride_lag >= len(ac):
        return {"step_regularity": np.nan, "stride_regularity": np.nan, "symmetry": np.nan}

    def local_max(lag: int, w: int = 3) -> float:
        return float(ac[max(0, lag - w): lag + w + 1].max())

    step_r, stride_r = local_max(step_lag), local_max(stride_lag)
    return {"step_regularity": step_r, "stride_regularity": stride_r, "symmetry": float(abs(step_r - stride_r))}


def harmonic_ratio(x: np.ndarray, fs: float, stride_freq_hz: float, n_harm: int = 10) -> float:
    """Harmonic ratio: sum of even over sum of odd harmonics of the stride frequency (vertical/AP axes)."""
    x = np.asarray(x, float) - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    even = odd = 0.0
    for k in range(1, 2 * n_harm + 1):
        idx = int(np.argmin(np.abs(freqs - k * stride_freq_hz)))
        if k % 2 == 0:
            even += spec[idx]
        else:
            odd += spec[idx]
    return float(even / max(odd, 1e-12))


def spectral_entropy(x: np.ndarray, fs: float) -> float:
    f, p = welch(np.asarray(x, float), fs=fs, nperseg=min(len(x), int(4 * fs)))
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def feature_vector(channels: np.ndarray, fs: float) -> dict[str, float]:
    """Features from harmonised ``[T, 3]`` channels (magnitude, vertical, horizontal)."""
    mag, vert, hor = channels[:, 0], channels[:, 1], channels[:, 2]
    sf = cadence_hz(vert, fs)
    reg = stride_regularity(vert, fs, sf if np.isfinite(sf) else None)
    stride_f = sf / 2 if np.isfinite(sf) else np.nan
    feats = {
        "rms_mag": float(np.sqrt(np.mean(mag**2))),
        "rms_vert": float(np.sqrt(np.mean(vert**2))),
        "jerk_rms": float(np.sqrt(np.mean(np.diff(vert) ** 2)) * fs),
        "step_freq_hz": sf,
        "freeze_index": freeze_index(vert, fs),
        "harmonic_ratio": harmonic_ratio(vert, fs, stride_f) if np.isfinite(stride_f) else np.nan,
        "spectral_entropy": spectral_entropy(vert, fs),
        "power_loco": band_power(vert, fs, 0.5, 3.0),
        "power_freeze": band_power(vert, fs, 3.0, 8.0),
        "hor_to_vert": float(np.sqrt(np.mean(hor**2)) / max(np.sqrt(np.mean(vert**2)), 1e-9)),
    }
    feats.update(reg)
    return feats


def vgrf_stride_times(total_force: np.ndarray, fs: float, threshold_frac: float = 0.1, min_stride_s: float = 0.6) -> np.ndarray:
    """Stride intervals (s) from heel-strike detection on one foot's total VGRF (rising threshold crossings)."""
    f = np.asarray(total_force, float)
    thr = threshold_frac * np.percentile(f, 95)
    on = (f[1:] > thr) & (f[:-1] <= thr)
    idx = np.where(on)[0] + 1
    if len(idx) < 2:
        return np.array([])
    strides = np.diff(idx) / fs
    return strides[strides >= min_stride_s]


def stride_time_cv(stride_times: np.ndarray) -> float:
    s = np.asarray(stride_times, float)
    if len(s) < 3:
        return float("nan")
    return float(100.0 * np.std(s, ddof=1) / np.mean(s))
