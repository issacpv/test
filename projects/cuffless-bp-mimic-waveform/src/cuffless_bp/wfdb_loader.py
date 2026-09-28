"""Streaming waveform windows from PhysioNet (WFDB), VitalDB and PulseDB.

The PhysioNet loaders never download whole records: they read ``chunk_s`` seconds at a time with
``wfdb.rdrecord(..., pn_dir=..., sampfrom=, sampto=)`` and yield fixed-length windows that contain
one ECG lead, the PPG (``PLETH``) and the arterial pressure (``ABP``/``ART``). Each window keeps the
record name, the subject id and the absolute (de-identified, shifted) timestamp so that it can be
linked to clinical events such as MIMIC-IV ``inputevents``.

``wfdb``, ``vitaldb`` and ``h5py`` are imported lazily so that the feature / evaluation code can be
used without them (e.g. in the unit tests).
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from scipy.signal import resample_poly

ECG_NAMES = ("II", "I", "III", "V", "V1", "V2", "V5", "MCL", "MCL1", "AVR", "AVL", "AVF", "ECG", "ECG_II")
PPG_NAMES = ("PLETH", "Pleth", "PPG")
ABP_NAMES = ("ABP", "ART", "Art", "AoP", "ABP1")

DEFAULT_FS = 125.0


@dataclass
class Window:
    """One fixed-length multi-signal window (all arrays at the same ``fs``)."""

    source: str          # "mimic3wdb", "mimic4wdb", "vitaldb", "pulsedb"
    record: str          # record name (or VitalDB caseid)
    subject_id: int | str
    start_s: float       # seconds from record start
    fs: float
    ecg: np.ndarray
    ppg: np.ndarray
    abp: np.ndarray
    timestamp: _dt.datetime | None = None  # absolute time of window start if the header has one

    @property
    def duration_s(self) -> float:
        return len(self.ppg) / self.fs


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def _wfdb():
    try:
        import wfdb
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install wfdb to stream PhysioNet records") from e
    return wfdb


def split_record_path(rec: str) -> tuple[str, str]:
    """'p00/p000020/p000020-2183-04-28-17-47' -> ('p00/p000020', 'p000020-2183-04-28-17-47')."""
    rec = rec.strip("/")
    if "/" not in rec:
        return "", rec
    d, b = rec.rsplit("/", 1)
    return d, b


def subject_id_from_record(rec: str) -> int | None:
    """MIMIC-III ids are 'p' + 6 digits, MIMIC-IV ids 'p' + 8 digits."""
    m = re.search(r"p(\d{8})(?!\d)", rec) or re.search(r"p(\d{6})(?!\d)", rec)
    return int(m.group(1)) if m else None


def signal_roles(sig_names: Sequence[str]) -> dict[str, int]:
    """Map roles ('ecg', 'ppg', 'abp') to channel indices, preferring lead II for ECG."""
    roles: dict[str, int] = {}
    for pref in ECG_NAMES:
        if pref in sig_names:
            roles["ecg"] = list(sig_names).index(pref)
            break
    for pref in PPG_NAMES:
        if pref in sig_names:
            roles["ppg"] = list(sig_names).index(pref)
            break
    for pref in ABP_NAMES:
        if pref in sig_names:
            roles["abp"] = list(sig_names).index(pref)
            break
    return roles


def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from fractions import Fraction

    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator)


# ----------------------------------------------------------------------------------------------
# PhysioNet (MIMIC-III matched subset, MIMIC-IV waveform DB)
# ----------------------------------------------------------------------------------------------


def describe_record(rec: str, db_dir: str) -> dict:
    """Read only the header(s) of a record and report which signals it has.

    Multi-segment records (all MIMIC waveform records) declare their channels in a ``*_layout``
    header; this function follows it. Works with ``pn_dir`` streaming (no download).
    """
    wfdb = _wfdb()
    d, base = split_record_path(rec)
    pn_dir = f"{db_dir}/{d}" if d else db_dir
    hdr = wfdb.rdheader(base, pn_dir=pn_dir)
    sig_names: list[str] = list(getattr(hdr, "sig_name", None) or [])
    if not sig_names and getattr(hdr, "seg_name", None):
        layout = [s for s in hdr.seg_name if s.endswith("_layout")]
        if layout:
            sig_names = list(wfdb.rdheader(layout[0], pn_dir=pn_dir).sig_name)
    roles = signal_roles(sig_names)
    return {
        "record": base, "pn_dir": pn_dir, "subject_id": subject_id_from_record(rec),
        "has_ecg": "ecg" in roles, "has_pleth": "ppg" in roles, "has_abp": "abp" in roles,
        "sig_names": sig_names, "fs": float(hdr.fs), "n_samples": int(hdr.sig_len or 0),
        "base_datetime": getattr(hdr, "base_datetime", None),
    }


def iter_windows(rec: str, db_dir: str, source: str = "mimic3wdb", window_s: float = 10.0, hop_s: float = 5.0,
                 fs_target: float = DEFAULT_FS, chunk_s: float = 600.0, sampfrom: int = 0, sampto: int | None = None,
                 max_windows: int | None = None) -> Iterator[Window]:
    """Stream ``window_s``-second ECG/PPG/ABP windows from one PhysioNet record.

    Reads the record in chunks of ``chunk_s`` seconds; windows containing NaN (segment gaps) are
    skipped. Signals are resampled to ``fs_target``.
    """
    wfdb = _wfdb()
    info = describe_record(rec, db_dir)
    if not (info["has_ecg"] and info["has_pleth"] and info["has_abp"]):
        return
    names = info["sig_names"]
    roles = signal_roles(names)
    chans = [names[roles["ecg"]], names[roles["ppg"]], names[roles["abp"]]]
    fs = info["fs"]
    n_total = info["n_samples"] if sampto is None else min(sampto, info["n_samples"])
    chunk = int(chunk_s * fs)
    win, hop = int(window_s * fs), int(hop_s * fs)
    base_dt = info["base_datetime"]
    n_out = 0
    pos = sampfrom
    carry = np.empty((0, 3))
    carry_start = pos
    while pos < n_total:
        stop = min(pos + chunk, n_total)
        try:
            r = wfdb.rdrecord(info["record"], pn_dir=info["pn_dir"], sampfrom=pos, sampto=stop, channel_names=chans)
            x = np.asarray(r.p_signal, dtype=float)
            # rdrecord returns channels in the record's order; reorder to ecg, ppg, abp
            order = [list(r.sig_name).index(c) for c in chans]
            x = x[:, order]
        except Exception:  # segment without the requested channels, network hiccup, ...
            pos = stop
            carry = np.empty((0, 3))
            carry_start = pos
            continue
        x = np.vstack([carry, x])
        start_idx = carry_start
        i = 0
        while i + win <= len(x):
            seg = x[i:i + win]
            if not np.isnan(seg).any():
                ecg, ppg, abp = (_resample(seg[:, k], fs, fs_target) for k in range(3))
                t0 = (start_idx + i) / fs
                ts = base_dt + _dt.timedelta(seconds=t0) if base_dt else None
                yield Window(source, info["record"], info["subject_id"], t0, fs_target, ecg, ppg, abp, ts)
                n_out += 1
                if max_windows and n_out >= max_windows:
                    return
            i += hop
        carry = x[i:]
        carry_start = start_idx + i
        pos = stop


# ----------------------------------------------------------------------------------------------
# VitalDB
# ----------------------------------------------------------------------------------------------

VITALDB_TRACKS = ("SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART")


def load_vitaldb_case(caseid: int, fs: float = DEFAULT_FS, tracks: Sequence[str] = VITALDB_TRACKS) -> np.ndarray:
    """Return an (n, 3) array [ECG, PPG, ART] at ``fs`` Hz for one VitalDB case (open API)."""
    try:
        import vitaldb
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install vitaldb") from e
    arr = vitaldb.load_case(caseid, list(tracks), 1.0 / fs)
    return np.asarray(arr, dtype=float)


def iter_vitaldb_windows(caseid: int, window_s: float = 10.0, hop_s: float = 5.0, fs: float = DEFAULT_FS,
                         max_windows: int | None = None) -> Iterator[Window]:
    x = load_vitaldb_case(caseid, fs)
    yield from iter_array_windows(x, fs, source="vitaldb", record=str(caseid), subject_id=caseid,
                                  window_s=window_s, hop_s=hop_s, max_windows=max_windows)


def iter_array_windows(x: np.ndarray, fs: float, source: str, record: str, subject_id: int | str,
                       window_s: float = 10.0, hop_s: float = 5.0, max_windows: int | None = None) -> Iterator[Window]:
    """Cut an in-memory (n, 3) [ecg, ppg, abp] array into windows (used for VitalDB and caches)."""
    win, hop = int(window_s * fs), int(hop_s * fs)
    n_out = 0
    for i in range(0, len(x) - win + 1, hop):
        seg = x[i:i + win]
        if np.isnan(seg).any():
            continue
        yield Window(source, record, subject_id, i / fs, fs, seg[:, 0].copy(), seg[:, 1].copy(), seg[:, 2].copy())
        n_out += 1
        if max_windows and n_out >= max_windows:
            return


# ----------------------------------------------------------------------------------------------
# PulseDB (.mat v7.3 = HDF5)
# ----------------------------------------------------------------------------------------------

PULSEDB_FIELDS = ("ECG_F", "PPG_F", "ABP_F", "SegSBP", "SegDBP", "SubjectID", "CaseID", "WinID", "Age", "Gender")


def load_pulsedb_file(path: str, fields: Sequence[str] = PULSEDB_FIELDS, max_segments: int | None = None) -> dict[str, np.ndarray]:
    """Read selected fields from a PulseDB ``.mat`` (v7.3/HDF5) file into numpy arrays.

    PulseDB stores a struct ``Subj_Wins`` whose fields are HDF5 object-reference arrays (one
    reference per 10-s segment). Field names are passed explicitly because they differ between
    the per-subject files and the pre-built train/test files; unknown fields are skipped with a
    warning rather than raising.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install h5py") from e
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        root = f["Subj_Wins"] if "Subj_Wins" in f else f
        for name in fields:
            if name not in root:
                print(f"[pulsedb] field {name!r} not in {path}; available: {list(root.keys())[:20]}")
                continue
            ds = root[name]
            refs = ds[()]
            if refs.dtype.kind == "O":  # object references, one per segment
                refs = refs.ravel()
                if max_segments:
                    refs = refs[:max_segments]
                vals = []
                for r in refs:
                    v = f[r][()]
                    if v.dtype.kind in "iu" and name in ("SubjectID", "CaseID"):
                        v = "".join(chr(c) for c in v.ravel())  # MATLAB char arrays
                    vals.append(np.squeeze(v))
                try:
                    out[name] = np.stack(vals)
                except ValueError:
                    out[name] = np.array(vals, dtype=object)
            else:
                out[name] = np.asarray(refs)[:max_segments] if max_segments else np.asarray(refs)
    return out


# ----------------------------------------------------------------------------------------------
# MIMIC-IV clinical linkage for bolus events
# ----------------------------------------------------------------------------------------------

# MIMIC-IV icu/inputevents itemids for vasoactive drugs (d_items labels: "Phenylephrine",
# "Norepinephrine", "Epinephrine", "Vasopressin"). Ephedrine pushes are charted in hosp/emar,
# not inputevents; link them separately via emar.charttime if needed.
VASOACTIVE_ITEMIDS = {
    "phenylephrine": 221749, "norepinephrine": 221906, "epinephrine": 221289, "vasopressin": 222315,
}


def mimiciv_bolus_sql(root: str, itemids: Sequence[int] = (221749, 221906, 221289, 222315)) -> str:
    """DuckDB SQL for vasoactive *push* events in MIMIC-IV ``icu/inputevents`` (v3.1).

    Boluses are identified by ``ordercategorydescription`` in ('Drug Push', 'Bolus') or by an
    infusion whose duration is <= 2 min. Joins ``icustays`` to carry ``subject_id`` for linkage
    with waveform record names (``p<subject_id>``). Columns: subject_id, stay_id, itemid, label,
    starttime, endtime, amount, amountuom, rate, category.
    """
    r = root.rstrip("/")
    ids = ", ".join(map(str, itemids))
    return f"""
    SELECT i.subject_id, i.stay_id, i.itemid, d.label, i.starttime, i.endtime, i.amount, i.amountuom,
           i.rate, i.ordercategorydescription AS category
    FROM read_csv_auto('{r}/icu/inputevents.csv.gz') i
    JOIN read_csv_auto('{r}/icu/d_items.csv.gz') d USING (itemid)
    WHERE i.itemid IN ({ids})
      AND (i.ordercategorydescription IN ('Drug Push', 'Bolus')
           OR date_diff('second', i.starttime, i.endtime) <= 120)
    ORDER BY i.subject_id, i.starttime
    """
