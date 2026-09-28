#!/usr/bin/env python
"""Download the open inputs: ABC Atlas MERFISH metadata, CCFv3 annotation, DSURQE atlas.

Examples
--------
    python scripts/download_data.py --abc --sample      # ABC metadata via abc_atlas_access (small)
    python scripts/download_data.py --ccf --resolution 25
    python scripts/download_data.py --dsurqe
    python scripts/download_data.py --structures        # Allen structure ontology -> data/ccf/structures.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = logging.getLogger("download_data")

CCF_BASE = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/"
STRUCTURE_GRAPH = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
# DSURQE repository (Mouse Imaging Centre). File names follow the public repo listing; adjust if they change.
DSURQE_BASE = "http://repo.mouseimaging.ca/repo/DSURQE_40micron/"
DSURQE_FILES = ["DSURQE_40micron_average.mnc", "DSURQE_40micron_labels.mnc", "DSURQE_40micron_mask.mnc",
                "DSURQE_40micron_R_mapping.csv"]


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info("exists: %s", dest)
        return dest
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for c in r.iter_content(chunk):
                fh.write(c)
    LOG.info("saved %s", dest)
    return dest


def fetch_ccf(out: Path, res: int, template: bool) -> int:
    _download(f"{CCF_BASE}annotation/ccf_2017/annotation_{res}.nrrd", out / f"annotation_{res}.nrrd")
    if template:
        _download(f"{CCF_BASE}average_template/average_template_{res}.nrrd", out / f"average_template_{res}.nrrd")
    return 0


def fetch_structures(out: Path) -> int:
    import pandas as pd

    r = requests.get(STRUCTURE_GRAPH, timeout=120)
    r.raise_for_status()
    rows = []

    def walk(node, depth=0):
        rows.append({"id": node["id"], "acronym": node["acronym"], "name": node["name"],
                     "parent_id": node.get("parent_structure_id"), "depth": depth,
                     "structure_id_path": node.get("structure_id_path")})
        for ch in node.get("children", []):
            walk(ch, depth + 1)

    for root in r.json()["msg"]:
        walk(root)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "structures.csv", index=False)
    LOG.info("structures.csv: %d structures", len(rows))
    return 0


def fetch_abc(out: Path, sample: bool) -> int:
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    except ImportError:
        LOG.error("pip install abc-atlas-access  (or use `aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/`)")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    cache = AbcProjectCache.from_s3_cache(out)
    dirs = cache.list_directories()
    LOG.info("release %s; directories: %s", getattr(cache, "current_manifest", "?"), dirs)
    merfish = [d for d in dirs if d.startswith("MERFISH-C57BL6J") and not d.endswith("CCF")]
    ccf = [d for d in dirs if d.startswith("MERFISH-C57BL6J") and d.endswith("CCF")]
    if not merfish:
        LOG.error("no MERFISH directory in manifest; inspect %s", dirs)
        return 1
    cell = cache.get_metadata_dataframe(merfish[0], "cell_metadata")
    LOG.info("cell_metadata: %s", cell.shape)
    if ccf:
        coords = cache.get_metadata_dataframe(ccf[0], "ccf_coordinates")
        cell = cell.merge(coords, left_index=True, right_index=True, how="left") if cell.index.name else \
            cell.merge(coords, on="cell_label", how="left")
    if sample:
        cell = cell.sample(min(len(cell), 50000), random_state=0)
    (out / "tables").mkdir(exist_ok=True)
    cell.to_parquet(out / "tables" / ("cells_sample.parquet" if sample else "cells.parquet"))
    LOG.info("wrote %d cells with columns %s", len(cell), list(cell.columns)[:12])
    return 0


def fetch_dsurqe(out: Path) -> int:
    rc = 0
    for f in DSURQE_FILES:
        try:
            _download(DSURQE_BASE + f, out / f)
        except Exception as exc:  # noqa: BLE001
            LOG.error("%s: %s (check the atlas page in data/README.md for current file names)", f, exc)
            rc = 1
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--abc", action="store_true")
    ap.add_argument("--ccf", action="store_true")
    ap.add_argument("--template", action="store_true", help="also fetch the CCF average template")
    ap.add_argument("--resolution", type=int, default=25, choices=[10, 25, 50, 100])
    ap.add_argument("--structures", action="store_true")
    ap.add_argument("--dsurqe", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not any([a.abc, a.ccf, a.structures, a.dsurqe]):
        ap.print_help()
        return 0
    rc = 0
    if a.ccf:
        rc |= fetch_ccf(a.out / "ccf", a.resolution, a.template)
    if a.structures:
        rc |= fetch_structures(a.out / "ccf")
    if a.abc:
        rc |= fetch_abc(a.out / "abc_atlas", a.sample)
    if a.dsurqe:
        rc |= fetch_dsurqe(a.out / "mri" / "dsurqe")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
