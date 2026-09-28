"""Allen Visual Coding Neuropixels cache tables: loading, depth-from-surface and unit selection.

The cache CSVs (``sessions.csv``, ``probes.csv``, ``channels.csv``, ``units.csv``) come from the public S3
bucket (see data/README.md). Only pandas is required; NWB access is optional (h5py/pynwb/allensdk).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

VISUAL_CORTEX_AREAS: Sequence[str] = ("VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl")


def load_cache_tables(cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the four cache CSVs into a dict keyed by 'sessions', 'probes', 'channels', 'units'."""
    cache_dir = Path(cache_dir)
    return {name: pd.read_csv(cache_dir / f"{name}.csv") for name in ("sessions", "probes", "channels", "units")}


def surface_vertical_position(channels: pd.DataFrame, probes: pd.DataFrame) -> pd.Series:
    """Vertical position (um from probe tip) of the cortical-surface channel for every probe id.

    Uses ``probes.surface_channel_index`` and the channel table's ``probe_vertical_position`` / ``local_index``.
    Channels are ordered along the probe by ``probe_vertical_position``; the surface channel is the channel with
    local index equal to ``surface_channel_index``. If ``local_index`` is absent, channels are ranked by vertical
    position within the probe."""
    ch = channels.copy()
    if "local_index" not in ch.columns:
        ch["local_index"] = ch.groupby("probe_id")["probe_vertical_position"].rank(method="first").astype(int) - 1
    out = {}
    for _, p in probes.iterrows():
        sel = ch[(ch["probe_id"] == p["id"]) & (ch["local_index"] == int(p["surface_channel_index"]))]
        if len(sel):
            out[p["id"]] = float(sel["probe_vertical_position"].iloc[0])
        else:  # fall back to the top channel
            top = ch.loc[ch["probe_id"] == p["id"], "probe_vertical_position"]
            out[p["id"]] = float(top.max()) if len(top) else np.nan
    return pd.Series(out, name="surface_vertical_position")


def unit_depths_from_surface(units: pd.DataFrame, channels: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """Attach probe id, structure acronym, vertical position and depth-below-surface (um, positive = deeper)
    to every unit via its ``peak_channel_id``."""
    ch = channels.set_index("id")
    u = units.copy()
    u["probe_id"] = u["peak_channel_id"].map(ch["probe_id"])
    u["structure"] = u["peak_channel_id"].map(ch["ecephys_structure_acronym"])
    u["vertical_position"] = u["peak_channel_id"].map(ch["probe_vertical_position"])
    surf = surface_vertical_position(channels, probes)
    u["depth_um"] = u["probe_id"].map(surf) - u["vertical_position"]
    return u


def select_units(units_with_depth: pd.DataFrame, session_id: Optional[int] = None,
                 areas: Iterable[str] = VISUAL_CORTEX_AREAS, qc: bool = True) -> pd.DataFrame:
    """Cortical units (optionally one session) passing the Allen default QC filters if the columns exist."""
    u = units_with_depth
    m = u["structure"].isin(list(areas))
    if session_id is not None and "ecephys_session_id" in u.columns:
        m &= u["ecephys_session_id"] == session_id
    if qc:
        if "isi_violations" in u.columns:
            m &= u["isi_violations"] < 0.5
        if "amplitude_cutoff" in u.columns:
            m &= u["amplitude_cutoff"] < 0.1
        if "presence_ratio" in u.columns:
            m &= u["presence_ratio"] > 0.9
    return u[m].copy()


def lfp_channel_order(channels: pd.DataFrame, probe_id: int) -> pd.DataFrame:
    """Channels of one probe sorted from the surface downward (descending vertical position)."""
    ch = channels[channels["probe_id"] == probe_id].sort_values("probe_vertical_position", ascending=False)
    return ch.reset_index(drop=True)


def read_lfp_nwb(path: Path, max_channels: Optional[int] = None):
    """Read LFP data, timestamps and electrode ids from an Allen probe LFP NWB via h5py (optional dependency).

    Returns (data (n_time, n_channels) float32 in volts, timestamps (n_time,), electrode ids). The dataset paths
    follow the Allen NWB layout ``acquisition/probe_<id>_lfp/probe_<id>_lfp_data``; adjust if the layout changes.
    """
    import h5py  # lazy optional import

    with h5py.File(path, "r") as f:
        acq = f["acquisition"]
        key = next(k for k in acq if k.endswith("_lfp"))
        grp = acq[key][f"{key}_data"]
        data = grp["data"]
        ts = grp["timestamps"][:]
        elec = grp["electrodes"][:]
        X = data[:, :max_channels] if max_channels else data[:]
        conv = grp["data"].attrs.get("conversion", 1.0)
    return np.asarray(X, dtype=np.float32) * np.float32(conv), ts, elec


def synthetic_cache_tables(rng: np.random.Generator, n_probes: int = 2, n_channels: int = 40, n_units: int = 60
                           ) -> Dict[str, pd.DataFrame]:
    """Tiny synthetic sessions/probes/channels/units tables mirroring the cache columns (for tests)."""
    probes = pd.DataFrame({"id": np.arange(n_probes) + 100, "ecephys_session_id": 1,
                           "surface_channel_index": n_channels - 5, "sampling_rate": 30000.0})
    chans = []
    for p in probes["id"]:
        for i in range(n_channels):
            chans.append({"id": p * 1000 + i, "probe_id": p, "local_index": i, "probe_vertical_position": 20 * i,
                          "ecephys_structure_acronym": "VISp" if i >= 5 else "white matter"})
    channels = pd.DataFrame(chans)
    units = pd.DataFrame({"id": np.arange(n_units), "ecephys_session_id": 1,
                          "peak_channel_id": rng.choice(channels["id"], n_units),
                          "isi_violations": rng.uniform(0, 1, n_units), "amplitude_cutoff": rng.uniform(0, 0.2, n_units),
                          "presence_ratio": rng.uniform(0.8, 1.0, n_units)})
    sessions = pd.DataFrame({"id": [1], "session_type": ["brain_observatory_1.1"]})
    return {"sessions": sessions, "probes": probes, "channels": channels, "units": units}
