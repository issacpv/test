"""Mining fast-flush (square-wave) tests from recorded ABP to label damping objectively.

Nurses flush arterial lines routinely; the 300 mmHg square wave and the
ringing that follows its release are recorded in the raw waveform. From the
ringing we recover the natural frequency ``fn`` and damping coefficient
``zeta`` (Gardner, 1981): with successive opposite-direction extrema
amplitudes ``A1, A2`` (half a period apart) the logarithmic decrement is
``d = ln(A1/A2)`` and ``zeta = d / sqrt(pi^2 + d^2)``; the damped frequency is
``fd = 1 / T`` and ``fn = fd / sqrt(1 - zeta^2)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from .transfer import gardner_adequacy


@dataclass
class FlushResult:
    release_idx: int
    fn_hz: float
    zeta: float
    n_extrema: int
    adequacy: str


def detect_flush_events(abp: np.ndarray, fs: float, plateau_mmhg: float = 250.0,
                        min_plateau_s: float = 0.2, merge_gap_s: float = 1.0) -> list[tuple[int, int]]:
    """Contiguous runs above ``plateau_mmhg`` lasting >= ``min_plateau_s``; nearby runs are merged.

    Returns ``[(start, end)]`` with ``end`` = first index after the plateau (the release point).
    """
    above = np.asarray(abp) > plateau_mmhg
    if not above.any():
        return []
    edges = np.diff(above.astype(int), prepend=0, append=0)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= int(min_plateau_s * fs)]
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= int(merge_gap_s * fs):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _highpass(x: np.ndarray, fs: float, fc: float = 3.0) -> np.ndarray:
    if len(x) < 12:
        return x - np.median(x)
    b, a = butter(2, fc / (fs / 2), btype="high")
    return filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def ringing_parameters(abp: np.ndarray, fs: float, release_idx: int, window_s: float = 0.6,
                       min_amplitude_mmhg: float = 3.0) -> FlushResult:
    """Estimate fn and zeta from the ringing after a flush release.

    The post-release segment is high-passed (3 Hz) to remove the returning arterial pulse, and
    the first extrema of alternating sign are used. Fewer than two usable extrema means the
    response is over-damped (no oscillation): ``zeta`` is reported as NaN and ``adequacy`` as
    ``overdamped``.
    """
    seg = np.asarray(abp[release_idx: release_idx + int(window_s * fs)], float)
    if seg.size < 8:
        return FlushResult(release_idx, np.nan, np.nan, 0, "unknown")
    x = _highpass(seg, fs)
    dist = max(1, int(0.01 * fs))
    pk_max, _ = find_peaks(x, prominence=min_amplitude_mmhg, distance=dist)
    pk_min, _ = find_peaks(-x, prominence=min_amplitude_mmhg, distance=dist)
    ext = sorted([(i, x[i]) for i in pk_max] + [(i, x[i]) for i in pk_min])
    # keep strictly alternating signs
    alt: list[tuple[int, float]] = []
    for i, v in ext:
        if not alt or np.sign(v) != np.sign(alt[-1][1]):
            alt.append((i, v))
    if len(alt) < 2:
        return FlushResult(release_idx, np.nan, np.nan, len(alt), "overdamped")
    a1, a2 = abs(alt[0][1]), abs(alt[1][1])
    if a2 >= a1 or a2 <= 0:
        return FlushResult(release_idx, np.nan, np.nan, len(alt), "unknown")
    d = np.log(a1 / a2)
    zeta = float(d / np.sqrt(np.pi ** 2 + d ** 2))
    half_period = (alt[1][0] - alt[0][0]) / fs
    if len(alt) >= 3:
        period = (alt[2][0] - alt[0][0]) / fs
    else:
        period = 2 * half_period
    fd = 1.0 / period
    fn = float(fd / np.sqrt(max(1 - zeta ** 2, 1e-6)))
    return FlushResult(release_idx, fn, zeta, len(alt), gardner_adequacy(fn, zeta))


def flush_labels(abp: np.ndarray, fs: float, **kw) -> pd.DataFrame:
    """All flush events in a record with their (fn, zeta, adequacy) labels; one row per release."""
    rows = []
    for _, end in detect_flush_events(abp, fs):
        r = ringing_parameters(abp, fs, end, **kw)
        rows.append({"release_idx": r.release_idx, "t_release_s": r.release_idx / fs, "fn_hz": r.fn_hz,
                     "zeta": r.zeta, "n_extrema": r.n_extrema, "adequacy": r.adequacy})
    return pd.DataFrame(rows, columns=["release_idx", "t_release_s", "fn_hz", "zeta", "n_extrema", "adequacy"])


def propagate_labels(labels: pd.DataFrame, t_grid_s: np.ndarray, max_age_s: float = 4 * 3600.0) -> np.ndarray:
    """Assign each grid time the adequacy of the most recent flush within ``max_age_s`` (else 'unlabelled')."""
    out = np.full(len(t_grid_s), "unlabelled", dtype=object)
    if labels.empty:
        return out
    lab = labels.sort_values("t_release_s")
    idx = np.searchsorted(lab["t_release_s"].to_numpy(), t_grid_s, side="right") - 1
    for k, i in enumerate(idx):
        if i >= 0 and t_grid_s[k] - lab["t_release_s"].iloc[i] <= max_age_s:
            out[k] = lab["adequacy"].iloc[i]
    return out
