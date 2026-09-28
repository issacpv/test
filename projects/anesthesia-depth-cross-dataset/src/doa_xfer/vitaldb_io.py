"""VitalDB case selection, track alignment and EEG epoching.

Everything here works on plain CSV rows (as returned by the VitalDB REST API and
cached by ``scripts/download_data.py``) so that no network access is needed once
the files are on disk.  Agent arms:

* ``propofol``   - general anaesthesia with a propofol TCI track (``Orchestra/PPF20_CE``)
                   and no volatile agent track with meaningful exposure.
* ``sevoflurane`` - general anaesthesia with ``Primus/EXP_SEVO`` and no propofol TCI track
                   (an induction bolus recorded in the clinical table is allowed).
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EEG_TRACKS = ("BIS/EEG1_WAV", "BIS/EEG2_WAV")
PROPOFOL_TRACK = "Orchestra/PPF20_CE"
SEVO_TRACK = "Primus/EXP_SEVO"
DES_TRACK = "Primus/EXP_DES"
EEG_FS = 128.0


def _tracks_by_case(trks: Iterable[dict]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for t in trks:
        out.setdefault(str(t["caseid"]), set()).add(t["tname"])
    return out


def agent_of(track_names: set) -> str:
    """'propofol', 'sevoflurane', 'mixed' or 'other' from the set of available track names."""
    has_ppf = PROPOFOL_TRACK in track_names
    has_vol = SEVO_TRACK in track_names or DES_TRACK in track_names
    if has_ppf and not has_vol:
        return "propofol"
    if has_vol and not has_ppf:
        return "sevoflurane" if SEVO_TRACK in track_names else "other"
    if has_ppf and has_vol:
        return "mixed"
    return "other"


def select_cases(cases: Iterable[dict], trks: Iterable[dict], min_age: float = 18.0) -> List[dict]:
    """Eligible general-anaesthesia cases with EEG, tagged with their agent arm."""
    by_case = _tracks_by_case(trks)
    out = []
    for c in cases:
        cid = str(c.get("caseid", ""))
        names = by_case.get(cid, set())
        try:
            age = float(c.get("age", "nan"))
        except (TypeError, ValueError):
            age = float("nan")
        if str(c.get("ane_type", "")).strip().lower() != "general":
            continue
        if not np.isfinite(age) or age < min_age:
            continue
        has_eeg = any(t in names for t in EEG_TRACKS)
        if not has_eeg:
            continue
        agent = agent_of(names)
        if agent not in ("propofol", "sevoflurane"):
            continue
        out.append(dict(caseid=cid, subjectid=str(c.get("subjectid", "")), age=age, sex=c.get("sex", ""),
                        asa=c.get("asa", ""), agent=agent, has_eeg=int(has_eeg), n_tracks=len(names),
                        anestart=c.get("anestart", ""), aneend=c.get("aneend", ""),
                        opstart=c.get("opstart", ""), opend=c.get("opend", "")))
    return out


def read_track(path: Path) -> pd.DataFrame:
    """Read a cached track CSV(.gz): returns DataFrame with columns ``t`` (s) and ``v``."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh)
    df = df.rename(columns={df.columns[0]: "t", df.columns[1]: "v"})
    return df[["t", "v"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["t"])


def waveform_to_array(df: pd.DataFrame, fs: float = EEG_FS) -> Tuple[np.ndarray, float]:
    """Waveform tracks are stored as (Time, value) rows at the device rate; return (signal, t0)."""
    v = df["v"].to_numpy(float)
    t0 = float(df["t"].iloc[0]) if len(df) else 0.0
    return v, t0


def align_numeric(df: pd.DataFrame, grid_t: np.ndarray, max_gap_s: float = 60.0) -> np.ndarray:
    """Last-observation-carried-forward alignment of a numeric track onto ``grid_t`` (NaN beyond ``max_gap_s``)."""
    d = df.dropna(subset=["v"]).sort_values("t")
    t, v = d["t"].to_numpy(float), d["v"].to_numpy(float)
    if t.size == 0:
        return np.full(grid_t.shape, np.nan)
    idx = np.searchsorted(t, grid_t, side="right") - 1
    out = np.full(grid_t.shape, np.nan)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    gap = grid_t[ok] - t[idx[ok]]
    out[np.where(ok)[0][gap > max_gap_s]] = np.nan
    return out


def epoch_eeg(x: np.ndarray, fs: float, epoch_s: float = 4.0, step_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a 1-D EEG signal into (n_epochs, n_samples) epochs; returns (epochs, epoch start times in s)."""
    L, S = int(epoch_s * fs), int(step_s * fs)
    x = np.asarray(x, float)
    n = (x.size - L) // S + 1 if x.size >= L else 0
    if n <= 0:
        return np.zeros((0, L)), np.zeros(0)
    idx = np.arange(n)[:, None] * S + np.arange(L)[None, :]
    return x[idx], np.arange(n) * step_s


def artifact_flags(epochs: np.ndarray, amp_max_uv: float = 500.0, flat_thr_uv: float = 0.1) -> np.ndarray:
    """True for epochs that are saturated/large-amplitude, flat, or contain NaN."""
    e = np.asarray(epochs, float)
    nan = ~np.isfinite(e).all(axis=1)
    big = np.nanmax(np.abs(e), axis=1) > amp_max_uv
    flat = np.nanstd(e, axis=1) < flat_thr_uv
    return nan | big | flat


def frontal_channel_picks(channel_names: Sequence[str]) -> List[int]:
    """Indices of frontal channels approximating the BIS sensor montage (Fp1/Fp2/F7/F8/AF7/AF8/Fpz)."""
    wanted = {"FP1", "FP2", "FPZ", "F7", "F8", "AF7", "AF8", "AF3", "AF4"}
    return [i for i, c in enumerate(channel_names) if c.upper().replace("EEG ", "").strip() in wanted]


def case_grid(anestart: float, aneend: float, step_s: float = 1.0) -> np.ndarray:
    """1-s time grid spanning the anaesthesia period (VitalDB times are seconds relative to case start)."""
    return np.arange(float(anestart), float(aneend), step_s)
