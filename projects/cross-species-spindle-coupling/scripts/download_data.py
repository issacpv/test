#!/usr/bin/env python
"""Download helpers for the open data sources.

Usage
-----
    python scripts/download_data.py --dataset dandi --dandiset 000041 --list
    python scripts/download_data.py --dataset dandi --dandiset 000041 --sample        # smallest NWB asset
    python scripts/download_data.py --dataset dandi --dandiset 000041 --all           # every asset (large)
    python scripts/download_data.py --dataset sleep-edfx [--sample]

DANDI assets are listed with the REST API (paginated). If the API is unreachable (some proxies block
api.dandiarchive.org), the script falls back to the public S3 mirror of the dandiset metadata
(``dandisets/<id>/<version>/assets.yaml``) which contains paths, sizes and blob URLs.
The MNI iEEG atlas and NSRR cohorts require accepting terms / a DUA; see data/README.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DANDI_API = "https://api.dandiarchive.org/api"
DANDI_S3 = "https://dandiarchive.s3.amazonaws.com"
PHYSIONET_BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"
SAMPLE_EDFX = ["SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf", "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"]

# Versions verified on the public S3 mirror (dandisets/<id>/<version>/dandiset.yaml)
KNOWN_VERSIONS: Dict[str, str] = {"000041": "0.250624.0419", "000978": "0.240511.0247",
                                  "000166": "0.250624.0434", "000044": "0.250624.0426"}


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.relative_to(ROOT)}")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
        tmp.rename(dest)
    print(f"  {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------------------------------------------------
# DANDI
# ------------------------------------------------------------------------------------------------------

def dandi_assets_api(dandiset: str, version: str = "draft", page_size: int = 100) -> List[dict]:
    """List assets via the REST API with pagination. Returns dicts with path, size, asset_id, download_url."""
    url: Optional[str] = f"{DANDI_API}/dandisets/{dandiset}/versions/{version}/assets/?page_size={page_size}"
    out: List[dict] = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        js = r.json()
        for a in js.get("results", []):
            out.append({"path": a["path"], "size": int(a.get("size", 0)), "asset_id": a["asset_id"],
                        "download_url": f"{DANDI_API}/assets/{a['asset_id']}/download/"})
        url = js.get("next")
    return out


def dandi_assets_s3(dandiset: str, version: str) -> List[dict]:
    """Fallback: parse the public assets.yaml mirror (no API needed)."""
    import yaml  # lazy import

    url = f"{DANDI_S3}/dandisets/{dandiset}/{version}/assets.yaml"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    assets = yaml.safe_load(r.text)
    out: List[dict] = []
    for a in assets:
        urls = a.get("contentUrl", [])
        blob = next((u for u in urls if "s3.amazonaws.com" in u), urls[0] if urls else "")
        out.append({"path": a.get("path"), "size": int(a.get("contentSize", 0)),
                    "asset_id": a.get("identifier", ""), "download_url": blob})
    return out


def dandi_assets(dandiset: str, version: Optional[str]) -> List[dict]:
    try:
        return dandi_assets_api(dandiset, version or "draft")
    except Exception as e:  # API blocked or offline
        v = version or KNOWN_VERSIONS.get(dandiset)
        if not v:
            sys.exit(f"DANDI API unreachable ({e}) and no known published version for {dandiset}; pass --version")
        print(f"  DANDI API unreachable ({type(e).__name__}); using S3 metadata mirror for version {v}")
        return dandi_assets_s3(dandiset, v)


def download_dandi(dandiset: str, version: Optional[str], list_only: bool, sample: bool, all_assets: bool) -> None:
    assets = sorted(dandi_assets(dandiset, version), key=lambda a: a["size"])
    total = sum(a["size"] for a in assets) / 1e9
    print(f"dandiset {dandiset}: {len(assets)} assets, {total:.1f} GB")
    for a in assets:
        print(f"  {a['size'] / 1e9:8.2f} GB  {a['path']}")
    out = DATA / "dandi" / dandiset
    (out).mkdir(parents=True, exist_ok=True)
    # always keep the metadata next to the data
    v = version or KNOWN_VERSIONS.get(dandiset)
    if v:
        for name in ("dandiset.yaml", "assets.yaml"):
            try:
                _stream_download(f"{DANDI_S3}/dandisets/{dandiset}/{v}/{name}", out / name)
            except Exception as e:
                print(f"  could not fetch {name}: {e}")
    if list_only:
        return
    targets = assets[:1] if sample else (assets if all_assets else [])
    if not targets:
        print("nothing selected: pass --sample (smallest asset) or --all")
    for a in targets:
        _stream_download(a["download_url"], out / a["path"])


# ------------------------------------------------------------------------------------------------------
# PhysioNet Sleep-EDF
# ------------------------------------------------------------------------------------------------------

def _list_physionet_dir(url: str) -> List[str]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return [h for h in re.findall(r'href="([^"?/][^"]*)"', r.text) if not h.startswith("http")]


def download_sleep_edfx(sample: bool) -> None:
    out = DATA / "sleep-edfx"
    if sample:
        for name in SAMPLE_EDFX:
            _stream_download(PHYSIONET_BASE + "sleep-cassette/" + name, out / "sleep-cassette" / name)
        return
    for entry in _list_physionet_dir(PHYSIONET_BASE):
        if entry.endswith("/"):
            for name in _list_physionet_dir(PHYSIONET_BASE + entry):
                if not name.endswith("/"):
                    _stream_download(PHYSIONET_BASE + entry + name, out / entry / name)
        else:
            _stream_download(PHYSIONET_BASE + entry, out / entry)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dandi", "sleep-edfx"], required=True)
    ap.add_argument("--dandiset", default="000041")
    ap.add_argument("--version", default=None, help="published version id; default: API draft or known version")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.dataset == "dandi":
        download_dandi(args.dandiset, args.version, args.list, args.sample, args.all)
    else:
        download_sleep_edfx(args.sample)


if __name__ == "__main__":
    main()
