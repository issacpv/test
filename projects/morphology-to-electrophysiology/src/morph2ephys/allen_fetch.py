"""Allen Cell Types Database access (via allensdk) and a NeuroMorpho helper for morphology-only data.

``allensdk`` is imported lazily so the rest of the package works without it (e.g. when analysing
tables that were downloaded earlier). See ``data/README.md`` for the on-disk layout.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the environment
    from allensdk.core.cell_types_cache import CellTypesCache  # type: ignore

    HAVE_ALLENSDK = True
except Exception:  # noqa: BLE001
    CellTypesCache = None  # type: ignore
    HAVE_ALLENSDK = False

#: Allen ephys feature columns used as regression targets, with short names.
EPHYS_TARGETS: Dict[str, str] = {
    "input_resistance_mohm": "rin",
    "sag": "sag",
    "tau": "tau",
    "threshold_i_long_square": "rheobase",
    "upstroke_downstroke_ratio_long_square": "ud_ratio",
    "adaptation": "adaptation",
    "f_i_curve_slope": "fi_slope",
    "vrest": "vrest",
    "latency": "latency",
    "avg_isi": "avg_isi",
}

#: Passive (geometry-dominated) vs active (channel-dominated) split used in the transfer hypotheses.
PASSIVE_TARGETS = ("rin", "tau", "sag", "vrest")
ACTIVE_TARGETS = ("rheobase", "ud_ratio", "adaptation", "fi_slope", "latency", "avg_isi")

CELL_COLUMNS = ("id", "name", "species", "structure_layer_name", "structure_area_abbrev", "dendrite_type",
                "transgenic_line", "reporter_status", "donor_id", "disease_state", "normalized_depth",
                "reconstruction_type")


class AllenCellTypesFetcher:
    """Wrapper around ``CellTypesCache`` producing tidy tables and cached files.

    Parameters
    ----------
    cache_dir : directory holding ``manifest.json`` and per-specimen folders.
    """

    def __init__(self, cache_dir: Path) -> None:
        if not HAVE_ALLENSDK:
            raise ImportError("allensdk is required: pip install allensdk")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ctc = CellTypesCache(manifest_file=str(self.cache_dir / "manifest.json"))
        self._cells: Optional[pd.DataFrame] = None
        self._ephys: Optional[pd.DataFrame] = None
        self._morph: Optional[pd.DataFrame] = None
        self.table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ tables
    def cells(self, species: Optional[Sequence[str]] = None, require_reconstruction: bool = False) -> pd.DataFrame:
        """Cell metadata; ``species`` entries like 'mouse'/'human' (case-insensitive prefix match)."""
        raw = self.ctc.get_cells(require_reconstruction=require_reconstruction)
        df = pd.DataFrame(raw)
        keep = [c for c in CELL_COLUMNS if c in df.columns]
        df = df[keep].copy()
        if species:
            wanted = [s.lower() for s in species]
            df = df[df["species"].astype(str).str.lower().apply(lambda s: any(s.startswith(w) for w in wanted))]
        self._cells = df.reset_index(drop=True)
        return self._cells

    def ephys_features(self) -> pd.DataFrame:
        if self._ephys is None:
            self._ephys = self.ctc.get_ephys_features(dataframe=True)
        return self._ephys

    def morphology_features(self) -> pd.DataFrame:
        if self._morph is None:
            self._morph = self.ctc.get_morphology_features(dataframe=True)
        return self._morph

    def build_table(self, species: Optional[Sequence[str]] = None, require_reconstruction: bool = True) -> pd.DataFrame:
        """Cells joined with ephys features (renamed to short target names) and Allen morphometrics."""
        cells = self.cells(species=species, require_reconstruction=require_reconstruction)
        ephys = self.ephys_features().rename(columns=EPHYS_TARGETS)
        ephys_cols = ["specimen_id"] + [c for c in EPHYS_TARGETS.values() if c in ephys.columns]
        morph = self.morphology_features()
        morph = morph.add_prefix("allen_morph_").rename(columns={"allen_morph_specimen_id": "specimen_id"})
        table = cells.merge(ephys[ephys_cols], left_on="id", right_on="specimen_id", how="left")
        table = table.merge(morph, on="specimen_id", how="left")
        table["species"] = table["species"].astype(str).str.split().str[0].str.lower()
        self.table = table
        return table

    def save_tables(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._cells is not None:
            self._cells.to_csv(out_dir / "cells.csv", index=False)
        if self._ephys is not None:
            self._ephys.to_csv(out_dir / "ephys_features.csv", index=False)
        if self._morph is not None:
            self._morph.to_csv(out_dir / "morphology_features.csv", index=False)
        if self.table is not None:
            self.table.to_csv(out_dir / "cells_with_features.csv", index=False)

    # ------------------------------------------------------------------ files
    def download_reconstruction(self, specimen_id: int) -> Path:
        """Ensure the SWC for ``specimen_id`` is cached and return its path."""
        self.ctc.get_reconstruction(specimen_id)  # downloads on first call
        path = self.cache_dir / f"specimen_{specimen_id}" / "reconstruction.swc"
        if not path.exists():  # manifest layout changed? fall back to a search
            hits = list(self.cache_dir.glob(f"**/*{specimen_id}*/*.swc"))
            if not hits:
                raise FileNotFoundError(f"reconstruction for {specimen_id} not found in cache")
            path = hits[0]
        return path

    def download_ephys_nwb(self, specimen_id: int) -> Path:
        """Ensure the NWB sweep file for ``specimen_id`` is cached and return its path."""
        self.ctc.get_ephys_data(specimen_id)
        path = self.cache_dir / f"specimen_{specimen_id}" / "ephys.nwb"
        if not path.exists():
            hits = list(self.cache_dir.glob(f"**/*{specimen_id}*/*.nwb"))
            if not hits:
                raise FileNotFoundError(f"NWB for {specimen_id} not found in cache")
            path = hits[0]
        return path

    def long_square_sweeps(self, specimen_id: int) -> List[Dict[str, object]]:
        """Return long-square sweeps as dicts ``{t, v, i, sweep_number}`` (seconds, mV, pA)."""
        import numpy as np

        data_set = self.ctc.get_ephys_data(specimen_id)
        sweeps = self.ctc.get_ephys_sweeps(specimen_id)
        out = []
        for sw in sweeps:
            if "Long Square" not in str(sw.get("stimulus_name", "")):
                continue
            num = int(sw["sweep_number"])
            d = data_set.get_sweep(num)
            fs = float(d["sampling_rate"])
            idx0, idx1 = d["index_range"]
            v = np.asarray(d["response"][idx0:idx1 + 1]) * 1e3  # V -> mV
            i = np.asarray(d["stimulus"][idx0:idx1 + 1]) * 1e12  # A -> pA
            t = np.arange(v.size) / fs
            out.append({"t": t, "v": v, "i": i, "sweep_number": num})
        return out


def load_cached_table(allen_dir: Path) -> pd.DataFrame:
    """Load ``cells_with_features.csv`` produced by ``AllenCellTypesFetcher.save_tables``."""
    return pd.read_csv(Path(allen_dir) / "cells_with_features.csv")


def coarse_type_labels(table: pd.DataFrame) -> pd.Series:
    """Coarse cell-type label usable for both species: dendrite type x layer (t-types come from Patch-seq)."""
    dt = table.get("dendrite_type", pd.Series(["unknown"] * len(table))).astype(str)
    layer = table.get("structure_layer_name", pd.Series(["?"] * len(table))).astype(str)
    return (dt + "|L" + layer).rename("coarse_type")


# ---------------------------------------------------------------------- NeuroMorpho (morphology-only)
def fetch_neuromorpho_swcs(out_dir: Path, species: Iterable[str] = ("mouse", "human"),
                           brain_region: Sequence[str] = ("neocortex",), max_pages: int = 2,
                           max_files: Optional[int] = None, page_size: int = 500) -> int:
    """Download neocortical CNG.swc files from NeuroMorpho.org for self-supervised pre-training."""
    from urllib.parse import quote

    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = "https://neuromorpho.org/api/neuron/select"
    n_files = 0
    with requests.Session() as sess:
        for sp in species:
            for page in range(max_pages):
                body = {"species": [sp], "brain_region": list(brain_region)}
                resp = sess.post(api, json=body, params={"page": page, "size": page_size}, timeout=60)
                resp.raise_for_status()
                records = []
                for value in resp.json().get("_embedded", {}).values():
                    if isinstance(value, list):
                        records.extend(value)
                if not records:
                    break
                for rec in records:
                    archive, name = str(rec["archive"]), str(rec["neuron_name"])
                    target = out_dir / archive / f"{name}.CNG.swc"
                    if target.exists():
                        continue
                    url = f"https://neuromorpho.org/dableFiles/{quote(archive.lower())}/CNG%20version/{quote(name)}.CNG.swc"
                    r = sess.get(url, timeout=60)
                    if r.status_code != 200:
                        LOG.debug("missing SWC %s", url)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(r.content)
                    n_files += 1
                    if max_files is not None and n_files >= max_files:
                        return n_files
    return n_files


__all__ = ["HAVE_ALLENSDK", "AllenCellTypesFetcher", "EPHYS_TARGETS", "PASSIVE_TARGETS", "ACTIVE_TARGETS",
           "load_cached_table", "coarse_type_labels", "fetch_neuromorpho_swcs"]
