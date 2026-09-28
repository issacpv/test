"""Spindle and slow-oscillation detection: YASA wrappers plus a fallback detector.

Both backends return DataFrames with the same columns so downstream coupling code
is backend-agnostic.

Spindles: ``Start, Peak, End, Duration, Amplitude, Frequency, Stage`` (seconds, uV, Hz).
Slow oscillations: ``Start, NegPeak, MidCrossing, PosPeak, End, Duration, ValNegPeak, ValPosPeak, PTP, Frequency, Stage``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .io_edf import N2, N3, UNK

SPINDLE_COLS = ["Start", "Peak", "End", "Duration", "Amplitude", "Frequency", "Stage"]
SO_COLS = ["Start", "NegPeak", "MidCrossing", "PosPeak", "End", "Duration", "ValNegPeak", "ValPosPeak", "PTP", "Frequency", "Stage"]


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _segments(mask: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.column_stack([np.where(d == 1)[0], np.where(d == -1)[0]])


# --------------------------------------------------------------------------- #
# YASA backend
# --------------------------------------------------------------------------- #
def detect_spindles_yasa(x: np.ndarray, fs: float, hypno_samples: Optional[np.ndarray] = None,
                         include: Sequence[int] = (N2, N3), freq_sp: Tuple[float, float] = (12.0, 15.0),
                         **kwargs) -> pd.DataFrame:
    """YASA ``spindles_detect`` on one channel; returns the standardized spindle table."""
    import yasa  # lazy

    sp = yasa.spindles_detect(x, fs, hypno=hypno_samples, include=list(include), freq_sp=freq_sp, verbose="error", **kwargs)
    if sp is None:
        return pd.DataFrame(columns=SPINDLE_COLS)
    s = sp.summary()
    out = pd.DataFrame({"Start": s["Start"], "Peak": s["Peak"], "End": s["End"], "Duration": s["Duration"],
                        "Amplitude": s["Amplitude"], "Frequency": s["Frequency"],
                        "Stage": s["Stage"] if "Stage" in s else UNK})
    return out.reset_index(drop=True)


def detect_sos_yasa(x: np.ndarray, fs: float, hypno_samples: Optional[np.ndarray] = None,
                    include: Sequence[int] = (N2, N3), freq_sw: Tuple[float, float] = (0.3, 1.5), **kwargs) -> pd.DataFrame:
    """YASA ``sw_detect`` on one channel; returns the standardized SO table."""
    import yasa  # lazy

    sw = yasa.sw_detect(x, fs, hypno=hypno_samples, include=list(include), freq_sw=freq_sw, verbose="error", **kwargs)
    if sw is None:
        return pd.DataFrame(columns=SO_COLS)
    s = sw.summary()
    out = pd.DataFrame({c: s[c] for c in SO_COLS if c in s})
    for c in SO_COLS:
        if c not in out:
            out[c] = np.nan
    return out[SO_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# fallback (numpy) backend
# --------------------------------------------------------------------------- #
def detect_spindles_simple(x: np.ndarray, fs: float, hypno_samples: Optional[np.ndarray] = None,
                           include: Sequence[int] = (N2, N3), freq_sp: Tuple[float, float] = (11.0, 16.0),
                           thresh_sd: float = 2.0, min_dur: float = 0.5, max_dur: float = 2.5,
                           smooth_s: float = 0.2, merge_gap: float = 0.2) -> pd.DataFrame:
    """Envelope-threshold spindle detector (sigma-band Hilbert envelope > mean + ``thresh_sd`` SD in NREM).

    Meant for tests and as a detector-sensitivity reference, not as a replacement
    for YASA/Luna.
    """
    x = np.asarray(x, float)
    n = len(x)
    stage = np.full(n, N2) if hypno_samples is None else np.asarray(hypno_samples)[:n]
    nrem = np.isin(stage, list(include))
    sig = _bandpass(x, fs, *freq_sp)
    env = np.abs(hilbert(sig))
    k = max(int(round(smooth_s * fs)), 1)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    if nrem.sum() < fs:
        return pd.DataFrame(columns=SPINDLE_COLS)
    thr = env[nrem].mean() + thresh_sd * env[nrem].std()
    mask = (env > thr) & nrem
    segs = _segments(mask)
    # merge close segments
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < merge_gap * fs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rows = []
    for a, b in merged:
        dur = (b - a) / fs
        if not (min_dur <= dur <= max_dur):
            continue
        pk = a + int(np.argmax(env[a:b]))
        seg = sig[a:b]
        zc = np.sum(np.diff(np.sign(seg)) != 0)
        freq = zc / 2.0 / dur if dur > 0 else np.nan
        rows.append(dict(Start=a / fs, Peak=pk / fs, End=b / fs, Duration=dur,
                         Amplitude=float(np.ptp(seg)), Frequency=float(freq), Stage=int(stage[pk])))
    return pd.DataFrame(rows, columns=SPINDLE_COLS)


def detect_sos_simple(x: np.ndarray, fs: float, hypno_samples: Optional[np.ndarray] = None,
                      include: Sequence[int] = (N2, N3), freq_sw: Tuple[float, float] = (0.3, 1.5),
                      dur_range: Tuple[float, float] = (0.8, 2.0), neg_amp_min: Optional[float] = 40.0,
                      ptp_min: Optional[float] = 75.0, relative_sd: Optional[float] = None) -> pd.DataFrame:
    """Zero-crossing slow-oscillation detector (negative half-wave followed by positive half-wave).

    Candidate waves lie between consecutive positive-to-negative zero crossings of
    the 0.3-1.5 Hz signal; they are kept if the duration is in ``dur_range``, the
    negative peak is below ``-neg_amp_min`` uV and the peak-to-peak amplitude exceeds
    ``ptp_min`` uV. If ``relative_sd`` is given, amplitude thresholds are instead
    ``relative_sd`` times the SD of the filtered NREM signal (useful for synthetic data).
    """
    x = np.asarray(x, float)
    n = len(x)
    stage = np.full(n, N2) if hypno_samples is None else np.asarray(hypno_samples)[:n]
    nrem = np.isin(stage, list(include))
    sig = _bandpass(x, fs, *freq_sw)
    if relative_sd is not None and nrem.any():
        sd = sig[nrem].std()
        neg_amp_min, ptp_min = relative_sd * sd, 1.5 * relative_sd * sd
    sign = np.sign(sig)
    sign[sign == 0] = 1
    pn = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1  # positive-to-negative crossings
    rows = []
    for a, b in zip(pn[:-1], pn[1:]):
        dur = (b - a) / fs
        if not (dur_range[0] <= dur <= dur_range[1]) or not nrem[a:b].all():
            continue
        seg = sig[a:b]
        neg_idx = a + int(np.argmin(seg))
        mid_rel = np.where(np.diff(np.sign(seg)) > 0)[0]
        if len(mid_rel) == 0:
            continue
        mid = a + int(mid_rel[0]) + 1
        pos_idx = mid + int(np.argmax(sig[mid:b])) if b > mid else neg_idx
        vneg, vpos = float(sig[neg_idx]), float(sig[pos_idx])
        if (neg_amp_min is not None and -vneg < neg_amp_min) or (ptp_min is not None and vpos - vneg < ptp_min):
            continue
        rows.append(dict(Start=a / fs, NegPeak=neg_idx / fs, MidCrossing=mid / fs, PosPeak=pos_idx / fs, End=b / fs,
                         Duration=dur, ValNegPeak=vneg, ValPosPeak=vpos, PTP=vpos - vneg, Frequency=1.0 / dur,
                         Stage=int(stage[neg_idx])))
    return pd.DataFrame(rows, columns=SO_COLS)


# --------------------------------------------------------------------------- #
# dispatch and summaries
# --------------------------------------------------------------------------- #
def _has_yasa() -> bool:
    try:
        import yasa  # noqa: F401
        return True
    except ImportError:
        return False


def detect_events(x: np.ndarray, fs: float, hypno_samples: Optional[np.ndarray] = None,
                  backend: str = "auto", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect spindles and SOs with ``backend`` in {'auto', 'yasa', 'simple'}; returns ``(spindles, sos)``."""
    if backend == "auto":
        backend = "yasa" if _has_yasa() else "simple"
    if backend == "yasa":
        return detect_spindles_yasa(x, fs, hypno_samples, **kwargs.get("spindle_kwargs", {})), \
            detect_sos_yasa(x, fs, hypno_samples, **kwargs.get("so_kwargs", {}))
    return detect_spindles_simple(x, fs, hypno_samples, **kwargs.get("spindle_kwargs", {})), \
        detect_sos_simple(x, fs, hypno_samples, **kwargs.get("so_kwargs", {}))


def spindle_density(spindles: pd.DataFrame, hypno: np.ndarray, epoch_len: float,
                    stages: Sequence[int] = (N2, N3)) -> float:
    """Spindles per minute of the given stages (per-epoch hypnogram)."""
    minutes = np.isin(hypno, list(stages)).sum() * epoch_len / 60.0
    return float(len(spindles) / minutes) if minutes > 0 else np.nan


def spindle_summary(spindles: pd.DataFrame, hypno: np.ndarray, epoch_len: float) -> dict:
    """Classic per-night spindle metrics (density, mean duration/amplitude/frequency)."""
    return {
        "sp_density": spindle_density(spindles, hypno, epoch_len),
        "sp_density_n2": spindle_density(spindles[spindles["Stage"] == N2] if len(spindles) else spindles, hypno, epoch_len, (N2,)),
        "sp_duration": float(spindles["Duration"].mean()) if len(spindles) else np.nan,
        "sp_amplitude": float(spindles["Amplitude"].mean()) if len(spindles) else np.nan,
        "sp_frequency": float(spindles["Frequency"].mean()) if len(spindles) else np.nan,
        "sp_count": int(len(spindles)),
    }
