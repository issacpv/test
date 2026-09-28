"""Load human single-unit NWB sessions into a plain, dataset-agnostic container.

The only object downstream code needs is :class:`SessionData`: a list of spike-time
arrays (seconds), a trial table with at least ``stim_id`` and ``onset`` columns, and
per-unit metadata (region, electrode/wire id, quality metrics). Reading NWB requires
``pynwb`` (imported lazily so the statistics modules and tests run without it).

Column names differ between dandisets, so a mapping dictionary (``NwbMapping``)
tells the loader which NWB trial columns hold stimulus identity and onset time, and
which ``units`` columns hold region and quality metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class NwbMapping:
    """How to read one dandiset's NWB files.

    Attributes
    ----------
    stim_col : trial column with the stimulus identity (category or image id).
    onset_col : trial column with stimulus onset time (s). Falls back to ``start_time``.
    region_col : units column with the recording region label.
    electrode_col : units column with the electrode / microwire id.
    quality_cols : units columns with sorting-quality metrics to carry along.
    block_col : optional trial column with block index (for block permutations).
    """

    stim_col: str = "stimCategory"
    onset_col: str = "start_time"
    region_col: str = "location"
    electrode_col: str = "electrodes"
    quality_cols: Sequence[str] = ("SNR", "IsolationDist", "ISIViolations")
    block_col: Optional[str] = None


@dataclass
class SessionData:
    """Spike times and trial table for one recording session."""

    session_id: str
    spike_times: List[np.ndarray]
    trials: pd.DataFrame
    unit_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    dataset: str = ""
    patient_id: str = ""

    @property
    def n_units(self) -> int:
        return len(self.spike_times)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def validate(self) -> None:
        if "stim_id" not in self.trials.columns or "onset" not in self.trials.columns:
            raise ValueError("trials must have 'stim_id' and 'onset' columns")
        if len(self.unit_meta) not in (0, self.n_units):
            raise ValueError("unit_meta must be empty or have one row per unit")
        for st in self.spike_times:
            if np.any(np.diff(st) < 0):
                raise ValueError("spike times must be sorted")


def spike_counts(spikes: np.ndarray, onsets: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Spike counts of one unit in ``[onset + t0, onset + t1)`` for each trial onset.

    Uses ``searchsorted`` on the sorted spike-time array, so it is O((n_spikes + n_trials) log n).
    """
    spikes = np.asarray(spikes, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    lo = np.searchsorted(spikes, onsets + t0, side="left")
    hi = np.searchsorted(spikes, onsets + t1, side="left")
    return (hi - lo).astype(int)


def binned_counts(spikes: np.ndarray, onsets: np.ndarray, t_start: float, t_stop: float,
                  bin_width: float = 0.1, step: float = 0.05) -> np.ndarray:
    """Trials x bins spike-count matrix with sliding bins of ``bin_width`` every ``step`` s.

    Returns an array of shape ``(n_trials, n_bins)``; bin ``k`` covers
    ``[t_start + k*step, t_start + k*step + bin_width)`` relative to each onset.
    """
    if bin_width <= 0 or step <= 0:
        raise ValueError("bin_width and step must be positive")
    starts = np.arange(t_start, t_stop - bin_width + 1e-12, step)
    out = np.empty((len(onsets), len(starts)), dtype=int)
    for k, s in enumerate(starts):
        out[:, k] = spike_counts(spikes, onsets, s, s + bin_width)
    return out


def rates_matrix(session: SessionData, t0: float, t1: float) -> np.ndarray:
    """Units x trials firing-rate matrix (Hz) for the window ``[t0, t1)`` after onset."""
    onsets = session.trials["onset"].to_numpy(dtype=float)
    dur = float(t1 - t0)
    if dur <= 0:
        raise ValueError("window must have positive duration")
    return np.vstack([spike_counts(st, onsets, t0, t1) / dur for st in session.spike_times])


# ----------------------------------------------------------------------------- NWB reading
def _as_array(col: Any) -> np.ndarray:
    try:
        return np.asarray(col[:])
    except TypeError:
        return np.asarray(col)


def load_nwb_session(path: str, mapping: Optional[NwbMapping] = None, dataset: str = "",
                     session_id: Optional[str] = None) -> SessionData:
    """Read one NWB file (local path or an open ``h5py.File``) into :class:`SessionData`.

    Requires ``pynwb``. Unmapped columns are ignored but listed in ``unit_meta.attrs``-like
    metadata (``SessionData.unit_meta.attrs['unmapped']``) so mapping gaps are visible.
    """
    try:
        import pynwb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only with real data
        raise ImportError("pynwb is required to read NWB files: pip install pynwb") from exc

    mapping = mapping or NwbMapping()
    with pynwb.NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units
        trials = nwb.trials
        if units is None or trials is None:
            raise ValueError(f"{path}: NWB file has no units or trials table")

        spike_times = [np.sort(np.asarray(units["spike_times"][i], dtype=float)) for i in range(len(units))]

        tdf = trials.to_dataframe()
        stim_col = mapping.stim_col if mapping.stim_col in tdf.columns else None
        if stim_col is None:
            raise KeyError(f"trial column {mapping.stim_col!r} not in {list(tdf.columns)}")
        onset_col = mapping.onset_col if mapping.onset_col in tdf.columns else "start_time"
        out_trials = pd.DataFrame({
            "stim_id": _as_array(tdf[stim_col]),
            "onset": _as_array(tdf[onset_col]).astype(float),
            "trial_index": np.arange(len(tdf)),
        })
        if mapping.block_col and mapping.block_col in tdf.columns:
            out_trials["block"] = _as_array(tdf[mapping.block_col])

        udf = units.to_dataframe()
        meta: Dict[str, Any] = {}
        unmapped = []
        for name, col in (("region", mapping.region_col), ("electrode", mapping.electrode_col)):
            if col in udf.columns:
                values = udf[col]
                if name == "electrode":
                    # the electrodes column is a DynamicTableRegion -> take the first row index/label
                    values = values.apply(lambda v: (v.index[0] if hasattr(v, "index") and len(v) else v))
                meta[name] = _as_array(values)
            else:
                unmapped.append(col)
        for col in mapping.quality_cols:
            if col in udf.columns:
                meta[col] = _as_array(udf[col]).astype(float)
            else:
                unmapped.append(col)
        unit_meta = pd.DataFrame(meta, index=np.arange(len(spike_times)))
        unit_meta.attrs["unmapped"] = unmapped
        subject = getattr(nwb, "subject", None)
        patient = getattr(subject, "subject_id", "") if subject is not None else ""
        sid = session_id or getattr(nwb, "identifier", None) or str(path)

    sess = SessionData(session_id=str(sid), spike_times=spike_times, trials=out_trials,
                       unit_meta=unit_meta, dataset=dataset, patient_id=str(patient))
    sess.validate()
    return sess


def save_session_npz(session: SessionData, path: str) -> None:
    """Persist a session as a compressed npz (spike times as an object array)."""
    np.savez_compressed(
        path,
        session_id=session.session_id,
        dataset=session.dataset,
        patient_id=session.patient_id,
        spike_times=np.array(session.spike_times, dtype=object),
        trials=session.trials.to_records(index=False),
        unit_meta=session.unit_meta.to_records(index=False),
    )


def load_session_npz(path: str) -> SessionData:
    """Inverse of :func:`save_session_npz`."""
    with np.load(path, allow_pickle=True) as z:
        trials = pd.DataFrame.from_records(z["trials"])
        unit_meta = pd.DataFrame.from_records(z["unit_meta"]) if z["unit_meta"].size else pd.DataFrame()
        sess = SessionData(session_id=str(z["session_id"]), spike_times=[np.asarray(s, dtype=float) for s in z["spike_times"]],
                           trials=trials, unit_meta=unit_meta, dataset=str(z["dataset"]), patient_id=str(z["patient_id"]))
    sess.validate()
    return sess


__all__ = ["NwbMapping", "SessionData", "spike_counts", "binned_counts", "rates_matrix",
           "load_nwb_session", "save_session_npz", "load_session_npz"]
