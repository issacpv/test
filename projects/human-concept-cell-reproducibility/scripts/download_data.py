#!/usr/bin/env python
"""Enumerate and download human single-unit NWB datasets from the DANDI archive.

Uses the public DANDI REST API (https://api.dandiarchive.org/api/) which needs no
credentials. The ``dandi`` command-line client is used for full downloads when it is
installed; otherwise assets are fetched one by one over HTTPS.

Examples
--------
List candidate human single-unit dandisets and write ``data/dandisets.json``::

    python scripts/download_data.py --list

List the assets of one dandiset (id, path, size)::

    python scripts/download_data.py --assets 000004

Smoke test: download the smallest NWB asset of each default dandiset::

    python scripts/download_data.py --sample

Full download of selected dandisets (delegates to ``dandi download`` if available)::

    python scripts/download_data.py --dandisets 000004 000469 --full
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.dandiarchive.org/api"
DEFAULT_DANDISETS = ["000004", "000469"]
SEARCH_TERMS = ["human single neuron", "human single unit", "medial temporal lobe", "concept cells"]

LOG = logging.getLogger("download_data")


def _get(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def iter_paginated(url: str, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Iterate over DANDI's ``{"results": [...], "next": url}`` pagination."""
    params = dict(params or {})
    params.setdefault("page_size", 100)
    while url:
        data = _get(url, params=params)
        for item in data.get("results", []):
            yield item
        url = data.get("next")
        params = None  # ``next`` already carries the query string


def search_dandisets(terms: List[str]) -> List[Dict[str, Any]]:
    """Search dandisets by free text and return de-duplicated summaries."""
    found: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        for ds in iter_paginated(f"{API}/dandisets/", {"search": term, "embargoed": "false"}):
            version = ds.get("most_recent_published_version") or ds.get("draft_version") or {}
            ident = ds.get("identifier")
            if not ident:
                continue
            found[ident] = {
                "identifier": ident,
                "name": version.get("name"),
                "version": version.get("version"),
                "size_bytes": version.get("size"),
                "asset_count": version.get("asset_count"),
                "matched_terms": sorted(set(found.get(ident, {}).get("matched_terms", [])) | {term}),
            }
    return sorted(found.values(), key=lambda d: d["identifier"])


def list_assets(dandiset: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return ``[{asset_id, path, size}]`` for a dandiset (published version if any)."""
    if version is None:
        meta = _get(f"{API}/dandisets/{dandiset}/")
        v = meta.get("most_recent_published_version") or meta.get("draft_version") or {}
        version = v.get("version", "draft")
    out = []
    for a in iter_paginated(f"{API}/dandisets/{dandiset}/versions/{version}/assets/"):
        out.append({"asset_id": a["asset_id"], "path": a["path"], "size": a.get("size", 0), "version": version})
    return out


def download_asset(asset_id: str, target: Path, chunk: int = 1 << 20) -> Path:
    """Stream one asset to ``target`` (skips if the file exists with the same size)."""
    url = f"{API}/assets/{asset_id}/download/"
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        if target.exists() and total and target.stat().st_size == total:
            LOG.info("exists: %s", target)
            return target
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
        tmp.replace(target)
    LOG.info("downloaded %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def full_download(dandiset: str, out_dir: Path, jobs: int) -> int:
    """Delegate to the ``dandi`` CLI when present, else fetch every asset over HTTPS."""
    if shutil.which("dandi"):
        cmd = ["dandi", "download", f"DANDI:{dandiset}", "--output-dir", str(out_dir), "--jobs", str(jobs)]
        LOG.info("running: %s", " ".join(cmd))
        return subprocess.call(cmd)
    LOG.warning("dandi CLI not found; falling back to sequential HTTPS downloads")
    for a in list_assets(dandiset):
        download_asset(a["asset_id"], out_dir / dandiset / a["path"])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="data directory (default: ./data)")
    ap.add_argument("--list", action="store_true", help="search DANDI for human single-unit dandisets")
    ap.add_argument("--assets", metavar="DANDISET", help="list assets of one dandiset")
    ap.add_argument("--dandisets", nargs="*", default=None, help="dandiset ids (default: %s)" % " ".join(DEFAULT_DANDISETS))
    ap.add_argument("--sample", action="store_true", help="download the smallest NWB asset of each dandiset")
    ap.add_argument("--full", action="store_true", help="download the selected dandisets completely")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    dandisets = args.dandisets or DEFAULT_DANDISETS

    if args.list:
        results = search_dandisets(SEARCH_TERMS)
        path = args.out / "dandisets.json"
        path.write_text(json.dumps(results, indent=2))
        for r in results:
            size_gb = (r["size_bytes"] or 0) / 1e9
            print(f"{r['identifier']}  {size_gb:7.1f} GB  {r['asset_count'] or 0:5d} assets  {r['name']}")
        LOG.info("wrote %s (%d dandisets)", path, len(results))
        return 0

    if args.assets:
        assets = list_assets(args.assets)
        for a in sorted(assets, key=lambda d: d["size"]):
            print(f"{a['asset_id']}  {a['size'] / 1e6:9.1f} MB  {a['path']}")
        (args.out / f"assets_{args.assets}.json").write_text(json.dumps(assets, indent=2))
        return 0

    if args.sample:
        for ds in dandisets:
            assets = [a for a in list_assets(ds) if a["path"].endswith(".nwb")]
            if not assets:
                LOG.warning("no NWB assets in %s", ds)
                continue
            smallest = min(assets, key=lambda d: d["size"])
            download_asset(smallest["asset_id"], args.out / "dandi" / ds / smallest["path"])
        return 0

    if args.full:
        rc = 0
        for ds in dandisets:
            rc |= full_download(ds, args.out / "dandi", args.jobs)
        return rc

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
