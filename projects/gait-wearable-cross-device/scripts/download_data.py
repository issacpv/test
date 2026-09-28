#!/usr/bin/env python
"""Downloader for GaitShift-PD.

Open datasets are fetched directly (PhysioNet file server, UCI static
archive). Kaggle and Synapse data need their CLIs and accepted terms; PPMI
is a manual, application-gated download (instructions printed).

    python scripts/download_data.py --sample
    python scripts/download_data.py --all-open
    python scripts/download_data.py --kaggle | --mpower | --ppmi
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PHYSIONET_FILES = "https://physionet.org/files"
PHYSIONET_OPEN = ["gaitpdb/1.0.0", "gaitndd/1.0.0", "ltmm/1.0.0", "pads-parkinsons-disease-smartwatch/1.0.0"]
DAPHNET_ZIP = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
GAITPDB_SAMPLE = ["demographics.txt", "GaCo01_01.txt", "GaPt03_01.txt"]


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {url}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("fetched", dest.relative_to(ROOT))
    return True


def physionet_mirror(project: str) -> None:
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-P", str(DATA / "physionet"), f"{PHYSIONET_FILES}/{project}/"]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        sys.exit("wget not found; install wget or use the Python fetcher with --sample")


def cmd_sample() -> None:
    for rel in GAITPDB_SAMPLE:
        fetch(f"{PHYSIONET_FILES}/gaitpdb/1.0.0/{rel}", DATA / "physionet" / "gaitpdb" / "1.0.0" / rel)
    fetch(f"{PHYSIONET_FILES}/gaitndd/1.0.0/subject-description.txt", DATA / "physionet" / "gaitndd" / "1.0.0" / "subject-description.txt")


def cmd_all_open() -> None:
    for proj in PHYSIONET_OPEN:
        physionet_mirror(proj)
    zpath = DATA / "uci" / "daphnet.zip"
    if fetch(DAPHNET_ZIP, zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA / "uci" / "daphnet")
        print("Daphnet extracted to", DATA / "uci" / "daphnet")


def cmd_kaggle() -> None:
    dest = DATA / "kaggle" / "tlvmc-fog"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "tlvmc-parkinsons-freezing-gait-prediction", "-p", str(dest)], check=True)
    except FileNotFoundError:
        sys.exit("kaggle CLI not found: pip install kaggle; token in ~/.kaggle/kaggle.json")
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)


def cmd_mpower() -> None:
    print("mPower (Synapse project syn4993293): register at synapse.org, complete the qualified-researcher steps,")
    print("then in Python:")
    print("  import synapseclient; syn = synapseclient.login()")
    print("  walking = syn.tableQuery('SELECT * FROM syn5511449')   # walking activity table (verify id in the portal)")
    print("  files = syn.downloadTableColumns(walking, ['accel_walking_outbound.json.items'])")


def cmd_ppmi() -> None:
    print("PPMI: apply at https://www.ppmi-info.org/access-data-specimens/download-data (DUA, free).")
    print("After approval: https://ida.loni.usc.edu -> PPMI -> Download -> Study Data ->")
    print("  'Verily Study Watch' derived measures + MDS-UPDRS Part III + Participant Status + Demographics.")
    d = DATA / "ppmi"
    print("Place CSVs in", d, "(present)" if d.exists() and any(d.iterdir()) else "(missing)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--all-open", action="store_true")
    ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--mpower", action="store_true")
    ap.add_argument("--ppmi", action="store_true")
    args = ap.parse_args()
    ran = False
    for flag, fn in (("sample", cmd_sample), ("all_open", cmd_all_open), ("kaggle", cmd_kaggle), ("mpower", cmd_mpower), ("ppmi", cmd_ppmi)):
        if getattr(args, flag):
            fn()
            ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
