#!/usr/bin/env python
"""Download open inputs for morphology-to-spike-waveform.

Primary source: the Allen Cell Types Database (reconstructed morphologies, intracellular
electrophysiology features, biophysical models) via the AllenSDK or the REST API; optional
Allen Visual Coding Neuropixels opto-tagging sessions for in-vivo waveforms.

Examples
--------
Smoke test (10 reconstructed mouse cells, SWC + ephys features, REST API fallback if AllenSDK missing)::

    python scripts/download_data.py --sample

All reconstructed mouse cells and their models::

    python scripts/download_data.py --species mouse --all --models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = logging.getLogger("download_data")
API = "https://api.brain-map.org/api/v2/data/query.json"
SPECIES = {"mouse": "Mus musculus", "human": "Homo Sapiens"}


# --------------------------------------------------------------------------- REST helpers
def rma_query(criteria: str, num_rows: int = 1000) -> List[Dict]:
    rows: List[Dict] = []
    start = 0
    while True:
        r = requests.get(f"{API}?criteria={criteria},rma::options[start_row$eq{start}][num_rows$eq{num_rows}]", timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(payload.get("msg"))
        rows.extend(payload["msg"])
        if len(payload["msg"]) < num_rows:
            return rows
        start += num_rows


def list_reconstructed_cells(species: str) -> List[Dict]:
    """Cells with a 3-D reconstruction, via ApiCellTypesSpecimenDetail."""
    crit = ("model::ApiCellTypesSpecimenDetail,rma::criteria,[nr__reconstruction_type$ne'null'],"
            f"[donor__species$eq'{SPECIES[species]}']")
    return rma_query(crit)


def swc_download_link(specimen_id: int) -> Optional[str]:
    crit = (f"model::Specimen,rma::criteria,[id$eq{specimen_id}],rma::include,"
            "neuron_reconstructions(well_known_files(well_known_file_type))")
    rows = rma_query(crit)
    for spec in rows:
        for rec in spec.get("neuron_reconstructions", []):
            for wkf in rec.get("well_known_files", []):
                if "3DNeuronReconstruction" in (wkf.get("well_known_file_type") or {}).get("name", ""):
                    return "https://api.brain-map.org" + wkf["download_link"]
    return None


def download_swc_rest(cells: List[Dict], out_dir: Path, limit: Optional[int] = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in cells[:limit]:
        sid = c["specimen__id"]
        dest = out_dir / f"specimen_{sid}" / "reconstruction.swc"
        if dest.exists():
            n += 1
            continue
        link = swc_download_link(sid)
        if not link:
            LOG.warning("no SWC link for specimen %s", sid)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(link, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        n += 1
        LOG.info("SWC %s", sid)
    return n


# --------------------------------------------------------------------------- AllenSDK path
def download_with_allensdk(species: str, limit: Optional[int], models: bool) -> bool:
    try:
        from allensdk.core.cell_types_cache import CellTypesCache
    except ImportError:
        return False
    import pandas as pd

    ctc = CellTypesCache(manifest_file=str(DATA / "cell_types" / "manifest.json"))
    cells = ctc.get_cells(species=[SPECIES[species]], require_reconstruction=True)
    pd.DataFrame(cells).to_csv(DATA / "cell_types" / f"cells_{species}.csv", index=False)
    ctc.get_ephys_features(dataframe=True).to_csv(DATA / "cell_types" / "ephys_features.csv", index=False)
    ctc.get_morphology_features(dataframe=True).to_csv(DATA / "cell_types" / "morphology_features.csv", index=False)
    for c in cells[:limit]:
        ctc.get_reconstruction(c["id"])  # writes specimen_{id}/reconstruction.swc into the cache
        LOG.info("reconstruction %s (%s, %s)", c["id"], c.get("dendrite_type"), c.get("structure_layer_name"))
    if models:
        from allensdk.api.queries.biophysical_api import BiophysicalApi

        bp = BiophysicalApi()
        for c in cells[:limit]:
            rows = rma_query(f"model::NeuronalModel,rma::criteria,[specimen_id$eq{c['id']}],rma::include,neuronal_model_template")
            for m in rows:
                mdir = DATA / "models" / str(m["id"])
                if mdir.exists():
                    continue
                mdir.mkdir(parents=True, exist_ok=True)
                bp.cache_data(m["id"], working_directory=str(mdir))
                (mdir / "template.json").write_text(json.dumps(m.get("neuronal_model_template", {})))
                LOG.info("model %s for specimen %s", m["id"], c["id"])
    return True


def download_opto_sessions(n: int) -> None:
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:  # pragma: no cover
        LOG.error("allensdk is not installed")
        return
    cache = EcephysProjectCache.from_warehouse(manifest=str(DATA / "ecephys_cache_dir" / "manifest.json"))
    sessions = cache.get_session_table()
    opto = sessions[sessions["full_genotype"].str.contains("Sst|Pvalb|Vip", regex=True)]
    for sid in opto.index[:n]:
        s = cache.get_session_data(sid)
        LOG.info("opto session %s: %d units", sid, len(s.units))


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--species", choices=list(SPECIES), default="mouse")
    p.add_argument("--all", action="store_true", help="all reconstructed cells of the species")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--models", action="store_true", help="also cache biophysical models (needs AllenSDK)")
    p.add_argument("--opto-sessions", type=int, default=0, help="cache N Visual Coding opto-tagging sessions")
    p.add_argument("--rest-only", action="store_true", help="skip AllenSDK even if installed")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    limit = 10 if args.sample else (None if args.all else args.limit)
    if args.sample or args.all or args.limit:
        done = False if args.rest_only else download_with_allensdk(args.species, limit, args.models)
        if not done:
            LOG.info("AllenSDK unavailable or skipped: using the REST API")
            cells = list_reconstructed_cells(args.species)
            (DATA / "cell_types").mkdir(parents=True, exist_ok=True)
            (DATA / "cell_types" / f"cells_{args.species}.json").write_text(json.dumps(cells))
            n = download_swc_rest(cells, DATA / "cell_types", limit=limit)
            LOG.info("%d SWC files in place", n)
    if args.opto_sessions:
        download_opto_sessions(args.opto_sessions)
    if not (args.sample or args.all or args.limit or args.opto_sessions):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
