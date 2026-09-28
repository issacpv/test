"""Unified EDF loading with montage harmonization to a common bipolar set.

CHB-MIT ships bipolar channels ("FP1-F7"), Siena/TUSZ/Helsinki ship referential
channels ("EEG Fp1", "EEG FP1-REF", "EEG FP1-LE"). This module normalizes names,
derives the 18 longitudinal-bipolar ("double banana") pairs from whatever is
available, resamples to a common rate and returns a masked array so downstream
code never has to know which corpus a record came from.

MNE is imported lazily so the harmonization logic can be unit-tested without it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt

#: Common 18-pair longitudinal bipolar montage (old 10-20 names T3/T4/T5/T6).
CANONICAL_BIPOLAR_18: Tuple[str, ...] = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FZ-CZ", "CZ-PZ",
)

#: Electrode aliases (new 10-20 names -> old names used in the canonical montage).
_ALIASES: Dict[str, str] = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "M1": "A1", "M2": "A2", "FPZ": "FPZ", "FP1": "FP1", "FP2": "FP2",
}
_PREFIX = re.compile(r"^(EEG|ECG|EKG|EMG|EOG)\s*", re.IGNORECASE)
_SUFFIX = re.compile(r"-(REF|LE|AVG|AR)$", re.IGNORECASE)


def normalize_electrode(name: str) -> str:
    """Normalize a single electrode label: ``'Fp1'`` -> ``'FP1'``, ``'T7'`` -> ``'T3'``."""
    n = name.strip().upper().replace(" ", "")
    n = _SUFFIX.sub("", n)
    return _ALIASES.get(n, n)


def normalize_channel_name(name: str) -> str:
    """Normalize a raw EDF channel label to ``'A'`` (referential) or ``'A-B'`` (bipolar).

    Examples
    --------
    >>> normalize_channel_name("EEG FP1-REF")
    'FP1'
    >>> normalize_channel_name("FP1-F7")
    'FP1-F7'
    >>> normalize_channel_name("EEG Fp1")
    'FP1'
    >>> normalize_channel_name("T7-P7")
    'T3-T5'
    """
    n = _PREFIX.sub("", name.strip()).strip().upper().replace(" ", "")
    if _SUFFIX.search(n):
        return normalize_electrode(n)
    if "-" in n:
        a, b = n.split("-", 1)
        return f"{normalize_electrode(a)}-{normalize_electrode(b)}"
    return normalize_electrode(n)


def parse_channel(name: str) -> Tuple[str, Optional[str]]:
    """Return ``(anode, cathode)`` for a normalized label; cathode is None if referential."""
    n = normalize_channel_name(name)
    if "-" in n:
        a, b = n.split("-", 1)
        return a, b
    return n, None


def harmonize_to_bipolar(
    data: np.ndarray,
    ch_names: Sequence[str],
    pairs: Sequence[str] = CANONICAL_BIPOLAR_18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive a fixed bipolar montage from referential and/or bipolar channels.

    Parameters
    ----------
    data : (n_channels, n_samples) array
    ch_names : raw channel labels aligned with ``data`` rows
    pairs : target bipolar pairs, e.g. ``"FP1-F7"``

    Returns
    -------
    out : (len(pairs), n_samples) array; rows that cannot be derived are NaN
    mask : (len(pairs),) bool array, True where the pair is available

    Notes
    -----
    Resolution order for a pair A-B: (1) an existing channel ``A-B``;
    (2) an existing channel ``B-A`` negated; (3) referential ``A`` minus
    referential ``B``. Duplicate labels keep the first occurrence.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(ch_names):
        raise ValueError("data must be (n_channels, n_samples) aligned with ch_names")
    idx: Dict[str, int] = {}
    for i, nm in enumerate(ch_names):
        idx.setdefault(normalize_channel_name(nm), i)
    out = np.full((len(pairs), data.shape[1]), np.nan)
    mask = np.zeros(len(pairs), dtype=bool)
    for k, pair in enumerate(pairs):
        a, b = pair.split("-")
        if pair in idx:
            out[k] = data[idx[pair]]
        elif f"{b}-{a}" in idx:
            out[k] = -data[idx[f"{b}-{a}"]]
        elif a in idx and b in idx:
            out[k] = data[idx[a]] - data[idx[b]]
        else:
            continue
        mask[k] = True
    return out, mask


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float, axis: int = -1) -> np.ndarray:
    """Polyphase resampling of ``x`` from ``fs_in`` to ``fs_out`` Hz (NaN rows preserved)."""
    if np.isclose(fs_in, fs_out):
        return np.asarray(x, dtype=float)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    x = np.asarray(x, dtype=float)
    nan_rows = np.isnan(x).all(axis=axis) if x.ndim == 2 else None
    xf = np.nan_to_num(x)
    y = resample_poly(xf, frac.numerator, frac.denominator, axis=axis)
    if nan_rows is not None and nan_rows.any():
        y[nan_rows] = np.nan
    return y


def bandpass(x: np.ndarray, fs: float, l_freq: float = 0.5, h_freq: Optional[float] = 70.0,
             notch: Optional[float] = None, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (and optional notch) along the last axis; NaN rows kept."""
    x = np.asarray(x, dtype=float)
    nan_rows = np.isnan(x).all(axis=-1) if x.ndim == 2 else np.zeros((), bool)
    xf = np.nan_to_num(x)
    nyq = fs / 2.0
    hi = min(h_freq, 0.99 * nyq) if h_freq else None
    if hi:
        sos = butter(order, [l_freq / nyq, hi / nyq], btype="band", output="sos")
    else:
        sos = butter(order, l_freq / nyq, btype="high", output="sos")
    y = sosfiltfilt(sos, xf, axis=-1)
    if notch and notch < nyq:
        b, a = iirnotch(notch / nyq, Q=30.0)
        y = filtfilt(b, a, y, axis=-1)
    if np.ndim(nan_rows) and nan_rows.any():
        y[nan_rows] = np.nan
    return y


@dataclass
class Recording:
    """A harmonized EEG recording."""

    data: np.ndarray          # (n_pairs, n_samples), NaN rows for missing pairs
    fs: float
    pairs: Tuple[str, ...]
    mask: np.ndarray          # (n_pairs,) availability
    subject_id: str
    record_id: str
    dataset: str
    path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.fs

    def iter_windows(self, win_s: float, step_s: float) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield ``(start_time_s, window)`` with window shape (n_pairs, win_samples)."""
        for start, stop in window_bounds(self.data.shape[1], self.fs, win_s, step_s):
            yield start / self.fs, self.data[:, start:stop]


def window_bounds(n_samples: int, fs: float, win_s: float, step_s: float) -> List[Tuple[int, int]]:
    """Sample index bounds of fixed-length windows covering ``n_samples``."""
    w = int(round(win_s * fs))
    s = int(round(step_s * fs))
    if w <= 0 or s <= 0:
        raise ValueError("win_s and step_s must be positive")
    return [(st, st + w) for st in range(0, n_samples - w + 1, s)]


# --------------------------------------------------------------------------- #
# subject-id inference per corpus (used for leakage-safe splitting)
# --------------------------------------------------------------------------- #
def infer_dataset(path: str | Path) -> str:
    p = str(path).lower()
    if "chbmit" in p or re.search(r"chb\d\d", p):
        return "chbmit"
    if "siena" in p or re.search(r"pn\d\d", p):
        return "siena"
    if "tusz" in p or "tuh_eeg" in p:
        return "tusz"
    if "helsinki" in p or re.search(r"eeg\d+\.edf$", p):
        return "helsinki"
    return "unknown"


def infer_subject_id(path: str | Path, dataset: Optional[str] = None) -> str:
    """Extract the patient identifier from a corpus-specific path.

    CHB-MIT: ``chb01_03.edf`` -> ``chb01`` (chb21 is the same child as chb01 and is
    mapped to ``chb01`` to avoid leakage). Siena: ``PN00-1.edf`` -> ``PN00``.
    TUSZ: ``.../edf/train/aaaaaaac/s001_2002/01_tcp_ar/aaaaaaac_s001_t000.edf`` -> ``aaaaaaac``.
    Helsinki: ``eeg12.edf`` -> ``eeg12``.
    """
    p = Path(path)
    ds = dataset or infer_dataset(p)
    name = p.stem
    if ds == "chbmit":
        m = re.match(r"(chb\d\d)", name, re.IGNORECASE)
        sid = m.group(1).lower() if m else name
        return "chb01" if sid == "chb21" else sid
    if ds == "siena":
        m = re.match(r"(PN\d\d)", name, re.IGNORECASE)
        return m.group(1).upper() if m else name
    if ds == "tusz":
        m = re.match(r"([a-z]{8})_s\d+", name)
        if m:
            return m.group(1)
        return p.parents[2].name if len(p.parents) > 2 else name
    if ds == "helsinki":
        return name
    return name


def load_edf_harmonized(
    path: str | Path,
    target_fs: float = 256.0,
    pairs: Sequence[str] = CANONICAL_BIPOLAR_18,
    l_freq: float = 0.5,
    h_freq: Optional[float] = 70.0,
    notch: Optional[float] = None,
    dataset: Optional[str] = None,
) -> Recording:
    """Read an EDF with MNE and return a :class:`Recording` in the canonical montage.

    Requires ``mne``. Filtering is applied after re-referencing, then the
    signal is resampled to ``target_fs``. Units are microvolts.
    """
    import mne  # lazy import

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="error")
    fs_in = float(raw.info["sfreq"])
    data = raw.get_data() * 1e6  # V -> uV
    harmonized, mask = harmonize_to_bipolar(data, raw.ch_names, pairs)
    harmonized = bandpass(harmonized, fs_in, l_freq, h_freq, notch)
    harmonized = resample_signal(harmonized, fs_in, target_fs)
    ds = dataset or infer_dataset(path)
    return Recording(
        data=harmonized, fs=float(target_fs), pairs=tuple(pairs), mask=mask,
        subject_id=infer_subject_id(path, ds), record_id=Path(path).stem, dataset=ds,
        path=str(path), meta={"fs_original": fs_in, "n_channels_original": len(raw.ch_names),
                             "meas_date": str(raw.info.get("meas_date"))},
    )


def montage_coverage(ch_names: Sequence[str], pairs: Sequence[str] = CANONICAL_BIPOLAR_18) -> Dict[str, bool]:
    """Which canonical pairs can be derived from a channel list (no data needed)."""
    dummy = np.zeros((len(ch_names), 1))
    _, mask = harmonize_to_bipolar(dummy, ch_names, pairs)
    return dict(zip(pairs, mask.tolist()))
