#!/usr/bin/env python
"""Download open inputs for gene-gradients-neural-timescales.

Sources: Allen Visual Coding / Visual Behavior Neuropixels (AllenSDK caches over public S3),
IBL brain-wide map (ONE API), Allen ISH expression energy (RMA API), ABC Atlas MERFISH
metadata (public S3 manifest), CCF structure centroids.

Examples
--------
Smoke test (session table + one Neuropixels session, 3 ISH genes, MERFISH metadata)::

    python scripts/download_data.py --sample

ISH energies for a gene list::

    python scripts/download_data.py --ish --gene-list data/gene_lists/channel_genes.txt

Heavy downloads are opt-in (``--visual-coding``, ``--visual-behavior``, ``--ibl``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = logging.getLogger("download_data")

ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
ABC_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
ABC_RELEASE = "20230830"


# --------------------------------------------------------------------------- Allen ISH
def rma_query(criteria: str, num_rows: int = 2000) -> List[Dict]:
    """Paginated RMA query (start_row / num_rows) returning all rows."""
    rows: List[Dict] = []
    start = 0
    while True:
        url = f"{ALLEN_API}?criteria={criteria},rma::options[start_row$eq{start}][num_rows$eq{num_rows}]"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success", False):
            raise RuntimeError(payload.get("msg"))
        rows.extend(payload["msg"])
        if len(payload["msg"]) < num_rows:
            return rows
        start += num_rows


def download_ish(genes: List[str]) -> Path:
    """Expression energy per CCF structure for each gene (coronal and sagittal experiments)."""
    import pandas as pd

    out_dir = DATA / "ish"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for gene in genes:
        sd_path = out_dir / f"section_datasets_{gene}.json"
        if sd_path.exists():
            datasets = json.loads(sd_path.read_text())
        else:
            crit = ("model::SectionDataSet,rma::criteria,[failed$eq'false'],products[abbreviation$eq'Mouse'],"
                    f"genes[acronym$eq'{gene}'],rma::include,genes,plane_of_section")
            datasets = rma_query(crit)
            sd_path.write_text(json.dumps(datasets))
        for ds in datasets:
            u_path = out_dir / f"unionize_{ds['id']}.json"
            if u_path.exists():
                unions = json.loads(u_path.read_text())
            else:
                unions = rma_query(f"model::StructureUnionize,rma::criteria,[section_data_set_id$eq{ds['id']}]")
                u_path.write_text(json.dumps(unions))
            plane = (ds.get("plane_of_section") or {}).get("name", "unknown")
            for u in unions:
                records.append({"gene": gene, "section_data_set_id": ds["id"], "plane": plane,
                                "structure_id": u["structure_id"], "expression_energy": u["expression_energy"]})
        LOG.info("ISH %s: %d experiments", gene, len(datasets))
    df = pd.DataFrame(records)
    dest = out_dir / "ish_expression_energy.csv"
    df.to_csv(dest, index=False)
    return dest


# --------------------------------------------------------------------------- ABC MERFISH
def iter_abc_files(manifest: Dict) -> Iterator[Dict]:
    for directory, kinds in manifest.get("file_listing", {}).items():
        for kind, node in kinds.items():
            stack = [node]
            while stack:
                cur = stack.pop()
                if not isinstance(cur, dict):
                    continue
                for name, rec in (cur.get("files") or {}).items():
                    if isinstance(rec, dict) and "relative_path" in rec:
                        yield {"directory": directory, "kind": kind, "name": name,
                               "relative_path": rec["relative_path"],
                               "url": rec.get("url") or f"{ABC_BASE}/{rec['relative_path']}", "size": rec.get("size")}
                stack.extend(v for k, v in cur.items() if k != "files" and isinstance(v, dict))


def download_abc_metadata(release: str = ABC_RELEASE) -> List[Path]:
    dest_manifest = DATA / "abc" / f"manifest_{release}.json"
    dest_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not dest_manifest.exists():
        r = requests.get(f"{ABC_BASE}/releases/{release}/manifest.json", timeout=120)
        r.raise_for_status()
        dest_manifest.write_text(r.text)
    manifest = json.loads(dest_manifest.read_text())
    out = []
    for rec in iter_abc_files(manifest):
        if rec["kind"] != "metadata" or rec["directory"] not in {"MERFISH-C57BL6J-638850", "MERFISH-C57BL6J-638850-CCF", "Allen-CCF-2020"}:
            continue
        dest = DATA / "abc" / rec["relative_path"]
        if dest.exists() and (rec["size"] is None or dest.stat().st_size == rec["size"]):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(rec["url"], stream=True, timeout=300) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        LOG.info("ABC metadata: %s", dest.name)
        out.append(dest)
    return out


# --------------------------------------------------------------------------- AllenSDK caches
def download_visual_coding(n_sessions: Optional[int]) -> None:
    """Session/unit tables and NWB files via AllenSDK (public S3 warehouse)."""
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:  # pragma: no cover
        LOG.error("allensdk is not installed (pip install allensdk; Python <= 3.11)")
        return
    cache_dir = DATA / "ecephys_cache_dir"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = EcephysProjectCache.from_warehouse(manifest=str(cache_dir / "manifest.json"))
    sessions = cache.get_session_table()
    sessions.to_csv(cache_dir / "sessions.csv")
    cache.get_units().to_csv(cache_dir / "units.csv")
    ids = list(sessions.index[:n_sessions]) if n_sessions else list(sessions.index)
    for sid in ids:
        s = cache.get_session_data(sid)
        LOG.info("session %s: %d units, areas %s", sid, len(s.units), sorted(set(s.units.ecephys_structure_acronym.dropna())))


def download_visual_behavior(n_sessions: Optional[int]) -> None:
    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
    except ImportError:  # pragma: no cover
        LOG.error("allensdk is not installed")
        return
    cache_dir = DATA / "vbn_cache"
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(cache_dir))
    table = cache.get_ecephys_session_table()
    table.to_csv(cache_dir / "ecephys_sessions.csv")
    for sid in list(table.index[:n_sessions] if n_sessions else table.index):
        cache.get_ecephys_session(ecephys_session_id=sid)
        LOG.info("VBN session %s cached", sid)


def download_ibl(n_insertions: Optional[int]) -> None:
    """Spike sorting for IBL brain-wide-map insertions via ONE (public Alyx)."""
    try:
        from one.api import ONE
        from brainbox.io.one import SpikeSortingLoader
    except ImportError:  # pragma: no cover
        LOG.error("ONE-api / ibllib are not installed")
        return
    one = ONE(base_url="https://openalyx.internationalbrainlab.org",
              username=os.environ.get("IBL_USER", "intbrainlab"), password=os.environ.get("IBL_PASS", "international"),
              silent=True, cache_dir=str(DATA / "ibl"))
    pids = one.search_insertions(project="brainwide", django="session__qc__lt,50")
    pids = list(pids)[:n_insertions] if n_insertions else list(pids)
    for pid in pids:
        ssl = SpikeSortingLoader(pid=pid, one=one)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)
        LOG.info("IBL %s: %d clusters", pid, len(clusters["cluster_id"]) if clusters else 0)


def write_ccf_centroids() -> Optional[Path]:
    try:
        from allensdk.core.reference_space_cache import ReferenceSpaceCache
    except ImportError:  # pragma: no cover
        LOG.error("allensdk is not installed")
        return None
    import numpy as np
    import pandas as pd

    ccf_dir = DATA / "ccf"
    ccf_dir.mkdir(parents=True, exist_ok=True)
    rsc = ReferenceSpaceCache(25, "annotation/ccf_2017", manifest=str(ccf_dir / "manifest.json"))
    annot, _ = rsc.get_annotation_volume()
    tree = rsc.get_structure_tree()
    rows = []
    for sid in np.unique(annot):
        if sid == 0:
            continue
        idx = np.argwhere(annot == sid)
        c = idx.mean(0) * 0.025
        rows.append({"structure_id": int(sid), "acronym": tree.get_structures_by_id([int(sid)])[0]["acronym"],
                     "x_mm": c[0], "y_mm": c[1], "z_mm": c[2], "n_voxels": len(idx)})
    dest = ccf_dir / "structure_centroids_mm.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--visual-coding", action="store_true")
    p.add_argument("--visual-behavior", action="store_true")
    p.add_argument("--ibl", action="store_true")
    p.add_argument("--n-sessions", type=int, default=None)
    p.add_argument("--n-insertions", type=int, default=None)
    p.add_argument("--ish", action="store_true")
    p.add_argument("--genes", nargs="*", default=None)
    p.add_argument("--gene-list", type=Path, default=None)
    p.add_argument("--merfish-metadata", action="store_true")
    p.add_argument("--ccf", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    genes = list(args.genes or [])
    if args.gene_list:
        genes += [g.strip() for g in args.gene_list.read_text().splitlines() if g.strip()]

    if args.sample:
        download_ish(genes or ["Hcn1", "Pvalb", "Sst"])
        download_abc_metadata()
        download_visual_coding(n_sessions=1)
        return 0
    if args.ish:
        if not genes:
            p.error("--ish needs --genes or --gene-list")
        download_ish(genes)
    if args.merfish_metadata:
        download_abc_metadata()
    if args.visual_coding:
        download_visual_coding(args.n_sessions)
    if args.visual_behavior:
        download_visual_behavior(args.n_sessions)
    if args.ibl:
        download_ibl(args.n_insertions)
    if args.ccf:
        write_ccf_centroids()
    if not any([args.ish, args.merfish_metadata, args.visual_coding, args.visual_behavior, args.ibl, args.ccf]):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
