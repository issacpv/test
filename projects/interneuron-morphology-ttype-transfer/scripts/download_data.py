#!/usr/bin/env python
"""Assemble the Patch-seq morphology datasets for the mouse -> human transfer study.

Examples
--------
Fetch Allen Cell Types reconstructions + metadata through allensdk (small sample)::

    python scripts/download_data.py --allen --sample

Clone the Tolias-lab mini-atlas repository::

    python scripts/download_data.py --mini-atlas

Build the unified cell table from whatever is on disk::

    python scripts/download_data.py --build-table

The Allen Patch-seq *release* archives (SWC + metadata for Gouwens 2020 and Lee 2023)
are downloaded from the portal pages printed by ``--allen`` (they are large zip
files without a stable programmatic API); this script handles the allensdk part and
the table assembly.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("download_data")

PORTAL_LINKS = [
    "https://portal.brain-map.org/explore/classes/multimodal-characterization",
    "https://celltypes.brain-map.org/",
    "https://github.com/berenslab/mini-atlas",
    "https://zenodo.org/records/3716519",
]


def fetch_allen(out: Path, sample: bool) -> int:
    """Download Cell Types Database reconstructions and metadata with allensdk."""
    try:
        from allensdk.core.cell_types_cache import CellTypesCache
    except ImportError:
        LOG.error("allensdk not installed: pip install allensdk (see data/README.md)")
        return 1
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    ctc = CellTypesCache(manifest_file=str(out / "manifest.json"))
    cells = ctc.get_cells(require_reconstruction=True)
    inh = [c for c in cells if str(c.get("dendrite_type", "")).lower() == "aspiny"]
    LOG.info("%d reconstructed cells, %d aspiny (putative interneurons)", len(cells), len(inh))
    if sample:
        inh = inh[:10]
    rows = []
    for c in inh:
        try:
            ctc.get_reconstruction(c["id"])
            rows.append({"specimen_id": c["id"], "species": c.get("species"), "structure_layer": c.get("structure_layer_name"),
                         "line": c.get("transgenic_line"), "dendrite_type": c.get("dendrite_type"),
                         "region": c.get("structure_area_abbrev")})
        except Exception as exc:  # noqa: BLE001
            LOG.warning("reconstruction failed for %s: %s", c["id"], exc)
    pd.DataFrame(rows).to_csv(out / "celltypes_reconstructed_aspiny.csv", index=False)
    LOG.info("wrote %d rows; Patch-seq release archives (t-type labels) must be fetched from:\n  %s",
             len(rows), "\n  ".join(PORTAL_LINKS[:1]))
    return 0


def clone_mini_atlas(out: Path) -> int:
    if (out / ".git").exists():
        LOG.info("mini-atlas already cloned at %s", out)
        return 0
    cmd = ["git", "clone", "--depth", "1", "https://github.com/berenslab/mini-atlas", str(out)]
    LOG.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def build_table(data_dir: Path) -> int:
    """Assemble data/cells.parquet from the per-dataset metadata files that exist."""
    import pandas as pd

    from morph_ttype.taxonomy import harmonise_subclass

    frames = []
    spec = {
        "allen_v1": (data_dir / "allen" / "mouse_v1_metadata.csv", "mouse", "VISp", data_dir / "allen" / "swc" / "mouse_v1"),
        "allen_mtg": (data_dir / "allen" / "human_mtg_metadata.csv", "human", "MTG", data_dir / "allen" / "swc" / "human_mtg"),
    }
    for name, (csv, species, region, swc_dir) in spec.items():
        if not csv.exists():
            LOG.warning("%s: %s missing (see data/README.md)", name, csv)
            continue
        df = pd.read_csv(csv)
        cols = {c.lower(): c for c in df.columns}
        ttype_col = next((cols[k] for k in cols if "t-type" in k or "t_type" in k or "ttype" in k), None)
        if ttype_col is None:
            LOG.error("%s: no t-type column found in %s", name, csv)
            continue
        tab = pd.DataFrame({
            "cell_id": df[cols.get("specimen_id", list(cols.values())[0])].astype(str),
            "dataset": name, "species": species, "region": region,
            "t_type": df[ttype_col].astype(str),
        })
        tab["subclass_harmonised"] = tab["t_type"].map(harmonise_subclass)
        tab["swc_path"] = [str(swc_dir / f"{cid}.swc") for cid in tab["cell_id"]]
        tab["has_swc"] = [Path(p).exists() for p in tab["swc_path"]]
        frames.append(tab)
    if not frames:
        LOG.error("nothing to assemble")
        return 1
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(data_dir / "cells.parquet", index=False)
    LOG.info("cells.parquet: %d rows; subclass counts:\n%s", len(table),
             table.groupby(["dataset", "subclass_harmonised"]).size())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--allen", action="store_true")
    ap.add_argument("--mini-atlas", action="store_true")
    ap.add_argument("--build-table", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (a.allen or a.mini_atlas or a.build_table):
        ap.print_help()
        print("\nManual downloads:\n  " + "\n  ".join(PORTAL_LINKS))
        return 0
    rc = 0
    if a.allen:
        rc |= fetch_allen(a.out / "allen", a.sample)
    if a.mini_atlas:
        rc |= clone_mini_atlas(a.out / "mini-atlas")
    if a.build_table:
        rc |= build_table(a.out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
