"""Session loading: DANDI/NWB streaming, AllenSDK and IBL adapters, and a synthetic generator.

All heavy dependencies (dandi, pynwb, remfile, h5py, allensdk, ONE) are imported
lazily inside the functions that need them, so the container class and the
synthetic generator work with numpy/pandas alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SessionData:
    """Container for one recording session.

    Attributes
    ----------
    session_id : identifier
    units : DataFrame indexed by unit id with at least ``area`` and any quality metrics
    spike_times : mapping unit id -> sorted spike times (s)
    stimulus : DataFrame with ``start_time``, ``stop_time``, ``table`` and condition columns
               (``condition`` is an integer label; extra columns are kept as-is)
    meta : free-form metadata (dataset, sorter, mouse id ...)
    """

    session_id: str
    units: pd.DataFrame
    spike_times: Dict[int, np.ndarray]
    stimulus: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def unit_ids(self) -> np.ndarray:
        return np.asarray(self.units.index)

    def subset_units(self, unit_ids: Sequence[int]) -> "SessionData":
        ids = [u for u in unit_ids if u in self.spike_times]
        return SessionData(self.session_id, self.units.loc[ids].copy(),
                           {u: self.spike_times[u] for u in ids}, self.stimulus.copy(), dict(self.meta))

    @property
    def t_range(self) -> tuple:
        lo = min((st[0] for st in self.spike_times.values() if len(st)), default=0.0)
        hi = max((st[-1] for st in self.spike_times.values() if len(st)), default=0.0)
        return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# DANDI / NWB streaming
# --------------------------------------------------------------------------- #
def resolve_dandi_asset_url(dandiset_id: str, path_substring: str, version: str = "draft") -> str:
    """Return the S3 URL of the first asset in ``dandiset_id`` whose path contains ``path_substring``."""
    from dandi.dandiapi import DandiAPIClient  # lazy

    with DandiAPIClient() as client:
        ds = client.get_dandiset(dandiset_id, version)
        for asset in ds.get_assets():
            if path_substring in asset.path:
                return asset.get_content_url(follow_redirects=1, strip_query=True)
    raise FileNotFoundError(f"no asset containing '{path_substring}' in DANDI:{dandiset_id}")


def open_nwb_streaming(url: str):
    """Open a remote NWB file for reading over HTTP. Returns ``(io, nwbfile)``; close ``io`` when done."""
    import h5py  # lazy
    import remfile
    from pynwb import NWBHDF5IO

    rem = remfile.File(url)
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io, io.read()


def _intervals_to_frame(name: str, table) -> pd.DataFrame:
    df = table.to_dataframe()
    df = df.reset_index(drop=True)
    df["table"] = name
    return df


def load_session_from_nwb(nwb, session_id: Optional[str] = None,
                          condition_columns: Sequence[str] = ("frame", "orientation", "image_name", "stimulus_condition_id")) -> SessionData:
    """Build a :class:`SessionData` from an open NWB file (Allen ecephys layout, generic fallback).

    Units: ``nwb.units`` with ``spike_times`` and quality columns; area from
    ``location``/``ecephys_structure_acronym`` if present. Stimulus: all
    ``nwb.intervals`` tables (Allen: ``*_presentations``) plus ``nwb.trials``.
    ``condition`` is taken from the first available column in ``condition_columns``
    (integer-coded).
    """
    units_df = nwb.units.to_dataframe()
    spike_times = {int(i): np.asarray(row["spike_times"], float) for i, row in units_df.iterrows()}
    units = units_df.drop(columns=[c for c in ("spike_times", "waveform_mean", "spike_amplitudes") if c in units_df])
    for cand in ("ecephys_structure_acronym", "location", "area"):
        if cand in units:
            units["area"] = units[cand].astype(str)
            break
    else:
        units["area"] = "unknown"

    frames = []
    for name, tbl in getattr(nwb, "intervals", {}).items():
        try:
            frames.append(_intervals_to_frame(name, tbl))
        except Exception:  # pragma: no cover - odd tables
            continue
    if getattr(nwb, "trials", None) is not None and "trials" not in [f["table"].iloc[0] for f in frames if len(f)]:
        frames.append(_intervals_to_frame("trials", nwb.trials))
    stim = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["start_time", "stop_time", "table"])
    stim = assign_condition(stim, condition_columns)
    sid = session_id or str(getattr(nwb, "identifier", "session"))
    meta = {"dataset": "nwb", "subject": str(getattr(getattr(nwb, "subject", None), "subject_id", ""))}
    return SessionData(sid, units, spike_times, stim, meta)


def assign_condition(stim: pd.DataFrame, condition_columns: Sequence[str]) -> pd.DataFrame:
    """Add an integer ``condition`` column from the first usable column in ``condition_columns``."""
    stim = stim.copy()
    stim["condition"] = -1
    for col in condition_columns:
        if col in stim:
            vals = stim[col]
            ok = vals.notna() & (vals.astype(str) != "null")
            codes, _ = pd.factorize(vals[ok].astype(str))
            stim.loc[ok, "condition"] = codes
            break
    return stim


def load_session_from_allensdk(session, session_id: str) -> SessionData:
    """Adapter for AllenSDK ``EcephysSession`` / Visual Behavior ecephys session objects."""
    units = session.units.copy()
    if "ecephys_structure_acronym" in units:
        units["area"] = units["ecephys_structure_acronym"].astype(str)
    elif "structure_acronym" in units:
        units["area"] = units["structure_acronym"].astype(str)
    spike_times = {int(u): np.asarray(t, float) for u, t in session.spike_times.items()}
    stim = session.stimulus_presentations.copy().reset_index()
    stim = stim.rename(columns={"stimulus_presentation_id": "presentation_id"})
    stim["table"] = stim.get("stimulus_name", "stimulus").astype(str)
    stim = assign_condition(stim, ("frame", "orientation", "image_name", "stimulus_condition_id"))
    return SessionData(session_id, units, spike_times, stim, {"dataset": "allensdk"})


def load_session_from_ibl(spikes, clusters, trials, session_id: str) -> SessionData:
    """Adapter for IBL ``SpikeSortingLoader`` outputs and a ``trials`` object.

    ``condition`` is the signed stimulus side (0 = left, 1 = right); other task
    variables are kept as columns (contrast, choice, block prior, feedback).
    """
    cl = pd.DataFrame({k: np.asarray(v) for k, v in dict(clusters).items() if np.ndim(v) == 1})
    cl.index.name = "unit_id"
    cl["area"] = cl["acronym"].astype(str) if "acronym" in cl else "unknown"
    st = np.asarray(spikes["times"]); sc = np.asarray(spikes["clusters"])
    order = np.argsort(sc, kind="stable")
    st, sc = st[order], sc[order]
    bounds = np.searchsorted(sc, np.arange(len(cl) + 1))
    spike_times = {u: np.sort(st[bounds[u]:bounds[u + 1]]) for u in range(len(cl))}
    tr = pd.DataFrame({k: np.asarray(v) for k, v in dict(trials).items() if np.ndim(v) == 1})
    left = tr.get("contrastLeft"); right = tr.get("contrastRight")
    side = np.where(np.isnan(np.asarray(right, float)), 0, 1) if right is not None else np.zeros(len(tr), int)
    stim = pd.DataFrame({
        "start_time": tr["stimOn_times"], "stop_time": tr["stimOn_times"] + 0.2,
        "table": "trials", "condition": side,
        "contrast": np.where(side == 1, right, left) if right is not None else np.nan,
        "choice": tr.get("choice"), "prob_left": tr.get("probabilityLeft"), "feedback": tr.get("feedbackType"),
    })
    return SessionData(session_id, cl, spike_times, stim, {"dataset": "ibl"})


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache_session(sess: SessionData, path: Path) -> None:
    """Persist a session as a compressed npz + parquet/csv side tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(sess.spike_times)
    lengths = np.array([len(sess.spike_times[u]) for u in ids])
    concat = np.concatenate([sess.spike_times[u] for u in ids]) if ids else np.empty(0)
    np.savez_compressed(path, unit_ids=np.array(ids), lengths=lengths, spikes=concat,
                        session_id=sess.session_id)
    sess.units.to_csv(path.with_suffix(".units.csv"))
    sess.stimulus.to_csv(path.with_suffix(".stim.csv"), index=False)


def load_cached_session(path: Path) -> SessionData:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    ids = z["unit_ids"]; lengths = z["lengths"]; concat = z["spikes"]
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    spike_times = {int(u): concat[bounds[i]:bounds[i + 1]] for i, u in enumerate(ids)}
    units = pd.read_csv(path.with_suffix(".units.csv"), index_col=0)
    stim = pd.read_csv(path.with_suffix(".stim.csv"))
    return SessionData(str(z["session_id"]), units, spike_times, stim, {"dataset": "cache"})


# --------------------------------------------------------------------------- #
# synthetic data (for tests and method validation)
# --------------------------------------------------------------------------- #
def synthetic_session(n_units: int = 40, n_conditions: int = 8, n_repeats: int = 30,
                      drift_rate: float = 0.0, base_rate: float = 5.0, tuning_gain: float = 8.0,
                      trial_dur: float = 0.5, iti: float = 0.5, areas: Sequence[str] = ("VISp", "LGd", "CA1"),
                      seed: int = 0) -> SessionData:
    """Poisson population with condition tuning that drifts as a random walk across repeats.

    ``drift_rate`` is the per-repeat standard deviation of the random walk applied to
    each unit's tuning vector (in units of ``tuning_gain``). ``drift_rate=0`` gives a
    stationary population, so any measured drift is the finite-count baseline.
    """
    rng = np.random.default_rng(seed)
    tuning = rng.random((n_units, n_conditions))
    tuning = tuning / tuning.sum(axis=1, keepdims=True) * n_conditions  # mean 1
    rows, spike_times = [], {u: [] for u in range(n_units)}
    t = 0.0
    cur = tuning.copy()
    for r in range(n_repeats):
        order = rng.permutation(n_conditions)
        for c in order:
            rate = np.clip(base_rate + tuning_gain * (cur[:, c] - 1.0), 0.2, None)
            counts = rng.poisson(rate * trial_dur)
            for u in range(n_units):
                if counts[u]:
                    spike_times[u].append(np.sort(t + rng.random(counts[u]) * trial_dur))
            rows.append(dict(start_time=t, stop_time=t + trial_dur, table="synthetic", condition=int(c), repeat=r))
            t += trial_dur + iti
        cur = cur + rng.standard_normal(cur.shape) * drift_rate
        cur = np.clip(cur, 0.0, None)
    spikes = {u: (np.concatenate(v) if v else np.empty(0)) for u, v in spike_times.items()}
    units = pd.DataFrame({
        "area": [areas[u % len(areas)] for u in range(n_units)],
        "isi_violations": rng.random(n_units) * 0.6,
        "amplitude_cutoff": rng.random(n_units) * 0.2,
        "presence_ratio": 0.85 + rng.random(n_units) * 0.15,
        "firing_rate": [len(spikes[u]) / t for u in range(n_units)],
    }, index=pd.Index(range(n_units), name="unit_id"))
    return SessionData("synthetic", units, spikes, pd.DataFrame(rows), {"dataset": "synthetic", "drift_rate": drift_rate})
