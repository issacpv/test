"""Multi-dataset 12-lead ECG loading with acquisition harmonisation.

Every record is converted to a float32 array of shape ``(12, n_samples)`` at a
common sampling rate, with a common lead order and duration, so that models
and shift diagnostics see identical tensors regardless of the source.

Supported on-disk formats
-------------------------
* WFDB (``.hea`` + ``.dat``/``.mat``): PTB-XL, ecg-arrhythmia (Chapman/Ningbo),
  Challenge-2021 (Georgia, CPSC), MIMIC-IV-ECG.  Requires the ``wfdb`` package.
* CODE-15% HDF5 (``exams_partN.hdf5`` with dataset ``tracings`` of shape
  (N, 4096, 12) at 400 Hz).  Requires ``h5py``.

The pure-numpy helpers (`harmonize_leads`, `resample_signal`, `crop_or_pad`,
`bandpass`, `standardize_record`) do not need either dependency and are what
the tests exercise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy import signal as sps

STANDARD_LEADS: tuple[str, ...] = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
"""Canonical lead order used throughout the package."""

_LEAD_ALIASES = {
    "DI": "I", "DII": "II", "DIII": "III", "LEAD I": "I", "LEAD II": "II", "LEAD III": "III",
    "MLI": "I", "MLII": "II", "MLIII": "III", "AVR": "AVR", "AVL": "AVL", "AVF": "AVF",
}

TARGET_FS = 500
TARGET_SECONDS = 10.0


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one ECG source."""

    name: str
    country: str
    root: Path
    fs: int
    fmt: str = "wfdb"  # "wfdb" or "code15"
    lead_names: tuple[str, ...] = STANDARD_LEADS
    glob: str = "**/*.hea"
    notes: str = ""


# Sources in the population-shift grid; roots are relative to the data dir.
DEFAULT_SPECS: dict[str, DatasetSpec] = {
    "ptbxl": DatasetSpec("ptbxl", "DE", Path("ptbxl/records500"), 500, glob="**/*_hr.hea"),
    "chapman": DatasetSpec("chapman", "CN", Path("ecg-arrhythmia/WFDBRecords"), 500,
                           notes="records JS00001-JS10646"),
    "ningbo": DatasetSpec("ningbo", "CN", Path("ecg-arrhythmia/WFDBRecords"), 500,
                          notes="records JS10647-JS45551"),
    "georgia": DatasetSpec("georgia", "US", Path("challenge2021/training/georgia"), 500),
    "cpsc": DatasetSpec("cpsc", "CN", Path("challenge2021/training/cpsc_2018"), 500),
    "code15": DatasetSpec("code15", "BR", Path("code15"), 400, fmt="code15",
                          lead_names=("DI", "DII", "DIII", "AVR", "AVL", "AVF",
                                      "V1", "V2", "V3", "V4", "V5", "V6"), glob="exams_part*.hdf5"),
    "mimic": DatasetSpec("mimic", "US", Path("mimic-iv-ecg/files"), 500),
}


def normalize_lead_name(name: str) -> str:
    """Map vendor / dataset lead spellings onto :data:`STANDARD_LEADS` names."""
    key = name.strip().upper().replace("-", "")
    return _LEAD_ALIASES.get(key, key)


def harmonize_leads(x: np.ndarray, lead_names: Sequence[str], fill_missing: bool = True) -> np.ndarray:
    """Reorder ``x`` (leads x samples) into :data:`STANDARD_LEADS` order.

    Missing limb leads are reconstructed from Einthoven/Goldberger relations
    when possible (III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2);
    any other missing lead is zero-filled when ``fill_missing`` else raises.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected (leads, samples)")
    have = {normalize_lead_name(n): i for i, n in enumerate(lead_names)}
    out = np.zeros((12, x.shape[1]), dtype=np.float32)
    for j, lead in enumerate(STANDARD_LEADS):
        if lead in have:
            out[j] = x[have[lead]]
    missing = [ld for ld in STANDARD_LEADS if ld not in have]
    if missing and "I" in have and "II" in have:
        I, II = x[have["I"]], x[have["II"]]
        derived = {"III": II - I, "AVR": -(I + II) / 2, "AVL": I - II / 2, "AVF": II - I / 2}
        for ld in list(missing):
            if ld in derived:
                out[STANDARD_LEADS.index(ld)] = derived[ld]
                missing.remove(ld)
    if missing and not fill_missing:
        raise ValueError(f"missing leads {missing}")
    return out


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling along the last axis (exact for rational ratios)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    from fractions import Fraction

    frac = Fraction(int(round(fs_out)), int(round(fs_in)))
    y = sps.resample_poly(np.asarray(x, dtype=np.float64), frac.numerator, frac.denominator, axis=-1)
    return y.astype(np.float32)


def crop_or_pad(x: np.ndarray, n_samples: int, mode: str = "center") -> np.ndarray:
    """Crop (centre or start) or zero-pad the last axis to ``n_samples``."""
    n = x.shape[-1]
    if n == n_samples:
        return x
    if n > n_samples:
        start = (n - n_samples) // 2 if mode == "center" else 0
        return x[..., start:start + n_samples]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n_samples - n)]
    return np.pad(x, pad, mode="constant")


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    nyq = 0.5 * fs
    high = min(high, 0.95 * nyq)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1).astype(np.float32)


def zscore(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray:
    """Per-lead z-scoring; pass source-domain ``mean``/``std`` (shape (12,1)) to avoid target leakage."""
    if mean is None:
        mean = x.mean(axis=-1, keepdims=True)
    if std is None:
        std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def standardize_record(
    x: np.ndarray,
    fs: float,
    lead_names: Sequence[str] = STANDARD_LEADS,
    target_fs: int = TARGET_FS,
    seconds: float = TARGET_SECONDS,
    filt: bool = True,
) -> np.ndarray:
    """Full harmonisation: lead order -> resample -> band-pass -> crop/pad. Returns (12, target_fs*seconds)."""
    x = harmonize_leads(x, lead_names)
    x = resample_signal(x, fs, target_fs)
    if filt:
        x = bandpass(x, target_fs)
    return crop_or_pad(x, int(round(target_fs * seconds)))


# ----------------------------------------------------------------------------
# WFDB header parsing (no wfdb dependency needed for metadata)
# ----------------------------------------------------------------------------
_HDR_RE = re.compile(r"^#\s*(Age|Sex|Dx|Rx|Hx|Sx)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class RecordMeta:
    """Metadata parsed from a WFDB header (Challenge / ecg-arrhythmia format)."""

    record: str
    fs: float
    n_samples: int
    lead_names: list[str]
    age: float | None = None
    sex: str | None = None
    dx: list[str] = field(default_factory=list)


def parse_wfdb_header(path: Path | str) -> RecordMeta:
    """Parse a ``.hea`` file: sampling rate, length, lead names, and ``#Age/#Sex/#Dx`` comments."""
    lines = Path(path).read_text().splitlines()
    first = lines[0].split()
    record = first[0]
    n_sig = int(first[1])
    fs = float(first[2]) if len(first) > 2 else 500.0
    n_samples = int(first[3]) if len(first) > 3 else 0
    leads: list[str] = []
    for ln in lines[1:1 + n_sig]:
        parts = ln.split()
        leads.append(parts[-1] if parts else f"ch{len(leads)}")
    meta = RecordMeta(record, fs, n_samples, leads)
    for ln in lines[1 + n_sig:]:
        m = _HDR_RE.match(ln.strip())
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "age":
            try:
                meta.age = float(val)
            except ValueError:
                meta.age = None
        elif key == "sex":
            v = val.lower()
            meta.sex = "F" if v.startswith("f") else "M" if v.startswith("m") else None
        elif key == "dx":
            meta.dx = [c.strip() for c in val.split(",") if c.strip() and c.strip().lower() != "nan"]
    return meta


def load_wfdb_record(path: Path | str) -> tuple[np.ndarray, float, list[str]]:
    """Read a WFDB record with ``wfdb``; returns ``(signal (leads, samples), fs, lead_names)``."""
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install wfdb") from exc
    p = str(path)
    if p.endswith(".hea"):
        p = p[:-4]
    rec = wfdb.rdrecord(p)
    sig = np.asarray(rec.p_signal, dtype=np.float32).T
    sig = np.nan_to_num(sig)
    return sig, float(rec.fs), list(rec.sig_name)


def iter_code15(hdf5_path: Path | str, batch: int = 256) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(exam_ids, tracings)`` batches from a CODE-15% part file; tracings are (B, 12, 4096) at 400 Hz."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install h5py") from exc
    with h5py.File(hdf5_path, "r") as f:
        ids = np.asarray(f["exam_id"])
        tr = f["tracings"]
        for s in range(0, len(ids), batch):
            block = np.asarray(tr[s:s + batch], dtype=np.float32)  # (B, 4096, 12)
            yield ids[s:s + batch], np.transpose(block, (0, 2, 1))


def trim_zero_padding(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Remove leading/trailing all-zero samples (CODE-15% pads shorter exams with zeros)."""
    nz = np.where(np.abs(x).sum(axis=0) > eps)[0]
    if nz.size == 0:
        return x
    return x[:, nz[0]:nz[-1] + 1]


def iter_dataset(spec: DatasetSpec, data_dir: Path | str, limit: int | None = None) -> Iterator[tuple[str, np.ndarray, RecordMeta | None]]:
    """Iterate harmonised ``(record_id, signal(12, 5000), meta)`` tuples for a source."""
    root = Path(data_dir) / spec.root
    n = 0
    if spec.fmt == "code15":
        for part in sorted(root.glob(spec.glob)):
            for ids, block in iter_code15(part):
                for eid, x in zip(ids, block):
                    x = trim_zero_padding(x)
                    yield str(int(eid)), standardize_record(x, spec.fs, spec.lead_names), None
                    n += 1
                    if limit and n >= limit:
                        return
        return
    for hea in sorted(root.glob(spec.glob)):
        rid = hea.stem
        if spec.name == "chapman" and not (rid.startswith("JS") and int(rid[2:]) <= 10646):
            continue
        if spec.name == "ningbo" and not (rid.startswith("JS") and int(rid[2:]) > 10646):
            continue
        meta = parse_wfdb_header(hea)
        sig, fs, leads = load_wfdb_record(hea)
        yield rid, standardize_record(sig, fs, leads), meta
        n += 1
        if limit and n >= limit:
            return
