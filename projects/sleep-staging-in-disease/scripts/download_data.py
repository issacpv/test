#!/usr/bin/env python
"""Download the open parts of the data (Sleep-EDF Expanded) and wrap the NSRR client.

Usage
-----
    python scripts/download_data.py --dataset sleep-edfx [--sample]
    python scripts/download_data.py --dataset shhs|mesa|mros|cfs [--annotations-only]
    python scripts/download_data.py --build-manifest

Sleep-EDF is fetched from PhysioNet over plain HTTPS (no credentials). NSRR cohorts require
an approved Data Access Request and a token in the environment variable ``NSRR_TOKEN``; this
script shells out to the official ``nsrr`` Ruby gem (``gem install nsrr``) and never stores
the token on disk.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PHYSIONET_BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"
SAMPLE_RECORDS = ["SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf", "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"]

NSRR_PATHS = {
    "shhs": ["shhs/datasets", "shhs/polysomnography/annotations-events-profusion", "shhs/polysomnography/edfs"],
    "mesa": ["mesa/datasets", "mesa/polysomnography/annotations-events-profusion", "mesa/polysomnography/edfs"],
    "mros": ["mros/datasets", "mros/polysomnography/annotations-events-profusion", "mros/polysomnography/edfs"],
    "cfs": ["cfs/datasets", "cfs/polysomnography/annotations-events-profusion", "cfs/polysomnography/edfs"],
}


def _list_physionet_dir(url: str) -> List[str]:
    """Return the hrefs of a PhysioNet directory index page (files and sub-directories)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"?/][^"]*)"', r.text)
    return [h for h in hrefs if not h.startswith("http")]


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
        tmp.rename(dest)
    print(f"  {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1e6:.1f} MB)")


def download_sleep_edfx(sample: bool = False) -> None:
    """Fetch Sleep-EDF Expanded (sleep-cassette + sleep-telemetry) or a 2-subject sample."""
    out = DATA / "sleep-edfx"
    if sample:
        for name in SAMPLE_RECORDS:
            _stream_download(PHYSIONET_BASE + "sleep-cassette/" + name, out / "sleep-cassette" / name)
        return
    for entry in _list_physionet_dir(PHYSIONET_BASE):
        if entry.endswith("/"):
            sub = entry
            for name in _list_physionet_dir(PHYSIONET_BASE + sub):
                if not name.endswith("/"):
                    _stream_download(PHYSIONET_BASE + sub + name, out / sub / name)
        else:
            _stream_download(PHYSIONET_BASE + entry, out / entry)


def download_nsrr(dataset: str, annotations_only: bool = False) -> None:
    """Wrap the official ``nsrr`` client. Requires NSRR_TOKEN and an approved DUA."""
    token = os.environ.get("NSRR_TOKEN")
    if not token:
        sys.exit("NSRR_TOKEN is not set. Get it from https://sleepdata.org/token after DUA approval.")
    if shutil.which("nsrr") is None:
        sys.exit("The 'nsrr' client is not installed. Run: gem install nsrr")
    out = DATA / "nsrr"
    out.mkdir(parents=True, exist_ok=True)
    paths: Iterable[str] = NSRR_PATHS[dataset]
    if annotations_only:
        paths = [p for p in paths if "edfs" not in p]
    for p in paths:
        cmd = ["nsrr", "download", p, f"--token={token}"]
        print("  running:", " ".join(c if not c.startswith("--token") else "--token=***" for c in cmd))
        subprocess.run(cmd, cwd=out, check=True)


def build_manifest() -> None:
    """One row per recording with paths and XML-derived epoch counts (no clinical variables)."""
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from stagebias.hypnogram import parse_profusion_xml  # noqa: E402

    rows = []
    for cohort in NSRR_PATHS:
        xml_dir = DATA / "nsrr" / cohort / "polysomnography" / "annotations-events-profusion"
        for xml in sorted(xml_dir.rglob("*-profusion.xml")) if xml_dir.exists() else []:
            rec_id = xml.name.replace("-profusion.xml", "")
            edf = list((DATA / "nsrr" / cohort / "polysomnography" / "edfs").rglob(rec_id + ".edf"))
            stages, _ = parse_profusion_xml(xml.read_text())
            rows.append({
                "cohort": cohort, "record_id": rec_id, "xml_path": str(xml.relative_to(ROOT)),
                "edf_path": str(edf[0].relative_to(ROOT)) if edf else "",
                "n_epochs": int(len(stages)), "n3_minutes": float((stages == 3).sum() * 0.5),
            })
    sc = DATA / "sleep-edfx"
    for psg in sorted(sc.rglob("*-PSG.edf")) if sc.exists() else []:
        rows.append({"cohort": "sleep-edfx", "record_id": psg.name[:7], "xml_path": "",
                     "edf_path": str(psg.relative_to(ROOT)), "n_epochs": -1, "n3_minutes": float("nan")})
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "manifests" / "recordings.csv", index=False)
    print(f"wrote {len(df)} rows to data/manifests/recordings.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["sleep-edfx", "shhs", "mesa", "mros", "cfs"])
    ap.add_argument("--sample", action="store_true", help="Sleep-EDF: fetch two subjects only")
    ap.add_argument("--annotations-only", action="store_true", help="NSRR: skip EDFs")
    ap.add_argument("--build-manifest", action="store_true")
    args = ap.parse_args()
    if args.build_manifest:
        build_manifest()
        return
    if args.dataset == "sleep-edfx":
        download_sleep_edfx(sample=args.sample)
    elif args.dataset:
        download_nsrr(args.dataset, annotations_only=args.annotations_only)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
