"""ECG feature extraction and a synthetic 12-lead generator.

The synthetic generator models each beat as a sum of Gaussian bumps (P, Q,
R, S, T) with lead-specific gains, so that beat morphology depends on a few
physiological knobs (heart rate, QRS duration, R/S amplitudes). It is used
for unit tests and for counterfactual ECG rendering from the twin state.

Feature extraction (R-peak detection, QRS duration, axis, Sokolow-Lyon
voltage) is deliberately simple and transparent; PTB-XL+ engineered
features are the reference for validating it on real data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Relative R-wave and S-wave gains per lead for a "normal axis" heart (dimensionless).
_R_GAIN = np.array([0.7, 1.0, 0.4, -0.5, 0.3, 0.7, 0.2, 0.5, 0.9, 1.3, 1.2, 1.0])
_S_GAIN = np.array([0.1, 0.15, 0.1, 0.2, 0.1, 0.1, 1.0, 1.2, 0.8, 0.4, 0.2, 0.1])


@dataclass(frozen=True)
class EcgFeatures:
    hr_bpm: float
    qrs_ms: float
    axis_deg: float
    sokolow_lyon_mv: float
    n_beats: int


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def synthesize_12lead(
    fs: int = 500,
    duration_s: float = 10.0,
    hr_bpm: float = 70.0,
    qrs_ms: float = 90.0,
    r_amp_mv: float = 1.0,
    s_amp_mv: float = 1.0,
    t_amp_mv: float = 0.3,
    noise_mv: float = 0.01,
    axis_shift_deg: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Synthetic 12-lead ECG ``[T, 12]`` in mV.

    ``r_amp_mv`` scales R waves in the lateral leads and ``s_amp_mv`` the S
    wave in V1/V2 (together they set the Sokolow-Lyon index). ``qrs_ms``
    sets the width of the QRS complex. ``axis_shift_deg`` rotates the frontal
    plane gains (positive = rightward).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    rr = 60.0 / hr_bpm
    beats = np.arange(0.3, duration_s, rr)
    qrs_w = (qrs_ms / 1000.0) / 5.0  # Gaussian sigma so that ~5 sigma spans the QRS
    # frontal-axis rotation of limb-lead R gains (I, II, III, aVR, aVL, aVF)
    limb_angles = np.deg2rad(np.array([0, 60, 120, -150, -30, 90]))
    axis = np.deg2rad(60.0 + axis_shift_deg)
    r_gain = _R_GAIN.copy()
    r_gain[:6] = np.cos(limb_angles - axis)
    sig = np.zeros((len(t), 12))
    for b in beats:
        p = 0.1 * _gauss(t, b - 0.16, 0.02)
        q = -0.1 * _gauss(t, b - 0.6 * qrs_w * 2.5, qrs_w * 0.6)
        r = _gauss(t, b, qrs_w)
        s = -_gauss(t, b + 0.6 * qrs_w * 2.5, qrs_w * 0.6)
        tw = _gauss(t, b + 0.30, 0.05)
        for k in range(12):
            sig[:, k] += 0.5 * p * r_gain[k] + 0.3 * q * r_gain[k] + r_amp_mv * r_gain[k] * r + s_amp_mv * _S_GAIN[k] * s + t_amp_mv * r_gain[k] * tw
    sig += noise_mv * rng.normal(size=sig.shape)
    return sig, list(LEADS)


def bandpass(x: np.ndarray, fs: int, lo: float = 5.0, hi: float = 20.0, order: int = 3) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def detect_r_peaks(lead: np.ndarray, fs: int) -> np.ndarray:
    """R-peak indices from one lead: band-pass, square, find peaks with 250 ms refractory."""
    f = bandpass(lead, fs)
    energy = f**2
    thr = 0.3 * np.percentile(energy, 99)
    peaks, _ = find_peaks(energy, height=thr, distance=int(0.25 * fs))
    # refine to the local maximum of |lead| within +/- 40 ms
    w = int(0.04 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - w), min(len(lead), p + w + 1)
        refined.append(lo + int(np.argmax(np.abs(lead[lo:hi]))))
    return np.array(sorted(set(refined)), dtype=int)


def heart_rate_bpm(r_peaks: np.ndarray, fs: int) -> float:
    if len(r_peaks) < 2:
        return float("nan")
    return float(60.0 / (np.median(np.diff(r_peaks)) / fs))


def qrs_duration_ms(lead: np.ndarray, fs: int, r_peaks: np.ndarray, frac: float = 0.05) -> float:
    """Median QRS duration: onset/offset where slope energy drops below ``frac`` of the beat maximum."""
    d = np.gradient(lead) ** 2
    w = int(0.10 * fs)
    durs = []
    for p in r_peaks:
        lo, hi = max(0, p - w), min(len(lead), p + w + 1)
        seg = d[lo:hi]
        thr = frac * seg.max()
        above = np.where(seg > thr)[0]
        if len(above) == 0:
            continue
        durs.append((above[-1] - above[0]) / fs * 1000.0)
    return float(np.median(durs)) if durs else float("nan")


def _net_deflection(lead: np.ndarray, fs: int, r_peaks: np.ndarray) -> float:
    w = int(0.06 * fs)
    vals = []
    for p in r_peaks:
        seg = lead[max(0, p - w): p + w + 1]
        vals.append(seg.max() + seg.min())  # positive peak plus negative trough
    return float(np.median(vals)) if vals else float("nan")


def qrs_axis_deg(sig: np.ndarray, leads: list[str], fs: int, r_peaks: np.ndarray) -> float:
    """Frontal QRS axis from net deflections in leads I and aVF."""
    i = _net_deflection(sig[:, leads.index("I")], fs, r_peaks)
    avf = _net_deflection(sig[:, leads.index("aVF")], fs, r_peaks)
    return float(np.rad2deg(np.arctan2(avf, i)))


def sokolow_lyon_mv(sig: np.ndarray, leads: list[str], fs: int, r_peaks: np.ndarray) -> float:
    """S(V1) + max(R(V5), R(V6)) in mV (LVH voltage criterion, >= 3.5 mV)."""
    w = int(0.06 * fs)

    def beat_stat(lead: np.ndarray, fn) -> float:
        vals = [fn(lead[max(0, p - w): p + w + 1]) for p in r_peaks]
        return float(np.median(vals)) if vals else float("nan")

    s_v1 = -beat_stat(sig[:, leads.index("V1")], np.min)
    r_v5 = beat_stat(sig[:, leads.index("V5")], np.max)
    r_v6 = beat_stat(sig[:, leads.index("V6")], np.max)
    return float(max(s_v1, 0.0) + max(r_v5, r_v6, 0.0))


def extract_features(sig: np.ndarray, leads: list[str], fs: int) -> EcgFeatures:
    """Compute the feature set used by the twin from a ``[T, 12]`` signal."""
    ref = sig[:, leads.index("II")]
    peaks = detect_r_peaks(ref, fs)
    return EcgFeatures(
        hr_bpm=heart_rate_bpm(peaks, fs),
        qrs_ms=qrs_duration_ms(ref, fs, peaks),
        axis_deg=qrs_axis_deg(sig, leads, fs, peaks),
        sokolow_lyon_mv=sokolow_lyon_mv(sig, leads, fs, peaks),
        n_beats=int(len(peaks)),
    )
