#!/usr/bin/env python
"""Fetch the open inputs for the virtual mouse brain validation study.

Examples
--------
Allen connectome matrices + region centroids (needs allensdk)::

    python scripts/download_data.py --allen

Allen CCF annotation volume (plain HTTP download)::

    python scripts/download_data.py --ccf --resolution 100

OpenNeuro ds001720 (multi-centre mouse rs-fMRI) - one subject in --sample mode::

    python scripts/download_data.py --openneuro --sample

DANDI widefield dataset (give the dandiset id from the paper)::

    python scripts/download_data.py --dandi 000XXX --sample
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = logging.getLogger("download_data")

CCF_URL = ("http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/"
           "annotation/ccf_2017/annotation_{res}.nrrd")


def fetch_allen(out: Path, sample: bool) -> int:
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        LOG.error("pip install allensdk")
        return 1
    import numpy as np
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    mcc = MouseConnectivityCache(manifest_file=str(out / "manifest.json"), resolution=100)
    tree = mcc.get_structure_tree()
    summary = tree.get_structures_by_set_id([167587189])  # "summary structures"
    ids = [s["id"] for s in summary]
    pd.DataFrame([{"id": s["id"], "acronym": s["acronym"], "name": s["name"]} for s in summary]).to_csv(
        out / "regions.csv", index=False)
    exps = mcc.get_experiments(cre=False)
    if sample:
        exps = exps[:20]
    eids = [e["id"] for e in exps]
    LOG.info("%d wild-type experiments, %d summary structures", len(eids), len(ids))
    for hemi, name in ((2, "ipsi"), (1, "contra")):
        pm = mcc.get_projection_matrix(experiment_ids=eids, projection_structure_ids=ids,
                                       hemisphere_ids=[hemi], parameter="normalized_projection_volume")
        # rows = experiments; aggregate to source summary structure by injection structure
        rows = pm["rows"]
        src = []
        for e in exps:
            inj = e["structure_id"]
            anc = [a for a in tree.ancestor_ids([inj])[0] if a in ids]
            src.append(anc[0] if anc else -1)
        src = np.asarray(src)
        M = np.full((len(ids), len(ids)), np.nan)
        for k, sid in enumerate(ids):
            m = src == sid
            if m.any():
                M[k] = np.nanmean(pm["matrix"][m], axis=0)
        np.savez(out / f"connectome_ncd_{name}.npz", matrix=M, ids=np.asarray(ids),
                 acronyms=np.asarray([c["label"] for c in pm["columns"]]))
        LOG.info("wrote %s (%d rows with injections)", name, int(np.isfinite(M).any(axis=1).sum()))
    # centroids for distance nulls (voxel coordinates * resolution)
    ann, _ = mcc.get_annotation_volume()
    cent = []
    for sid in ids:
        mask = mcc.get_structure_mask(sid)[0].astype(bool)
        idx = np.argwhere(mask)
        cent.append(idx.mean(axis=0) * 100.0 if len(idx) else [np.nan] * 3)
    pd.DataFrame(cent, columns=["x_um", "y_um", "z_um"]).assign(id=ids).to_csv(out / "centroids.csv", index=False)
    return 0


def fetch_ccf(out: Path, res: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    url = CCF_URL.format(res=res)
    dest = out / f"annotation_{res}.nrrd"
    if dest.exists():
        LOG.info("%s exists", dest)
        return 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    LOG.info("saved %s", dest)
    return 0


def fetch_openneuro(out: Path, sample: bool) -> int:
    cmd = ["openneuro-py", "download", "--dataset", "ds001720", "--target-dir", str(out / "ds001720")]
    if sample:
        cmd += ["--include", "sub-0101*", "--include", "participants.tsv", "--include", "dataset_description.json"]
    LOG.info("running: %s", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        LOG.error("openneuro-py not found: pip install openneuro-py (or use datalad / aws s3, see data/README.md)")
        return 1


def fetch_dandi(out: Path, dandiset: str, sample: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    try:
        if sample:
            listing = subprocess.run(["dandi", "ls", "-r", f"DANDI:{dandiset}"], capture_output=True, text=True)
            print(listing.stdout[:4000])
            LOG.info("sample mode lists assets only; rerun without --sample to download")
            return listing.returncode
        return subprocess.call(["dandi", "download", f"DANDI:{dandiset}", "-o", str(out), "--existing", "skip"])
    except FileNotFoundError:
        LOG.error("dandi CLI not found: pip install dandi")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--allen", action="store_true")
    ap.add_argument("--ccf", action="store_true")
    ap.add_argument("--resolution", type=int, default=100, choices=[10, 25, 50, 100])
    ap.add_argument("--openneuro", action="store_true")
    ap.add_argument("--dandi", type=str, default=None, help="dandiset id, e.g. 000123")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not any([a.allen, a.ccf, a.openneuro, a.dandi]):
        ap.print_help()
        return 0
    rc = 0
    if a.allen:
        rc |= fetch_allen(a.out / "allen", a.sample)
    if a.ccf:
        rc |= fetch_ccf(a.out / "allen", a.resolution)
    if a.openneuro:
        rc |= fetch_openneuro(a.out / "openneuro", a.sample)
    if a.dandi:
        rc |= fetch_dandi(a.out / "dandi", a.dandi, a.sample)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
