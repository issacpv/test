#!/usr/bin/env python
"""Downloader for the cardiac twin project.

Open data (PTB-XL, PTB-XL+) are fetched directly from PhysioNet's file
server. Credentialed data (MIMIC-IV-ECG / -ECHO, EchoNext) require
``PHYSIONET_USERNAME`` / ``PHYSIONET_PASSWORD`` in the environment and are
mirrored with wget (metadata-only by default). EchoNet-Dynamic requires a
manual DUA and is only checked.

Examples
--------
    python scripts/download_data.py --ptbxl --sample
    python scripts/download_data.py --ptbxl
    python scripts/download_data.py --mimic-ecg --metadata-only
    python scripts/download_data.py --check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PHYSIONET_FILES = "https://physionet.org/files"

PTBXL = "ptb-xl/1.0.3"
PTBXL_PLUS = "ptb-xl-plus/1.0.1"
MIMIC_ECG = "mimic-iv-ecg/1.0"
MIMIC_ECHO = "mimic-iv-echo/0.1"

PTBXL_SAMPLE_FILES = [
    "ptbxl_database.csv",
    "scp_statements.csv",
    "RECORDS",
    *[f"records100/00000/{i:05d}_lr.{ext}" for i in range(1, 6) for ext in ("hea", "dat")],
]


def fetch(url: str, dest: Path, auth: tuple[str, str] | None = None, chunk: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` (skips if present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, auth=auth) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                f.write(part)
    print("fetched", dest.relative_to(ROOT))


def wget_mirror(project: str, auth: tuple[str, str] | None, include: list[str] | None = None) -> None:
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-P", str(DATA)]
    if auth:
        cmd += ["--user", auth[0], "--password", auth[1]]
    if include:
        for pat in include:
            cmd += ["-A", pat]
    cmd.append(f"{PHYSIONET_FILES}/{project}/")
    shown = " ".join(cmd)
    if auth:
        shown = shown.replace(auth[1], "****")
    print(shown)
    subprocess.run(cmd, check=False)


def physionet_auth() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USERNAME")
    pw = os.environ.get("PHYSIONET_PASSWORD")
    if not user or not pw:
        sys.exit("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (credentialed access; see data/README.md).")
    return user, pw


def cmd_ptbxl(sample: bool) -> None:
    if sample:
        for rel in PTBXL_SAMPLE_FILES:
            fetch(f"{PHYSIONET_FILES}/{PTBXL}/{rel}", DATA / PTBXL / rel)
        return
    wget_mirror(PTBXL, None)
    wget_mirror(PTBXL_PLUS, None)


def cmd_mimic(project: str, metadata_only: bool, include_list: Path | None) -> None:
    auth = physionet_auth()
    meta = {
        MIMIC_ECG: ["record_list.csv", "machine_measurements.csv", "waveform_note_links.csv"],
        MIMIC_ECHO: ["echo-record-list.csv", "echo-study-list.csv"],
    }[project]
    if metadata_only:
        for rel in meta:
            try:
                fetch(f"{PHYSIONET_FILES}/{project}/{rel}", DATA / project / rel, auth=auth)
            except requests.HTTPError as exc:
                print(f"skip {rel}: {exc}")
        return
    if include_list is not None:
        paths = [p.strip() for p in include_list.read_text().splitlines() if p.strip()]
        for rel in paths:
            fetch(f"{PHYSIONET_FILES}/{project}/{rel}", DATA / project / rel, auth=auth)
        return
    wget_mirror(project, auth)


def cmd_check() -> None:
    import pandas as pd

    fl = DATA / "echonet-dynamic" / "FileList.csv"
    if fl.exists():
        df = pd.read_csv(fl)
        print(f"EchoNet-Dynamic: {len(df)} videos; EF mean {df['EF'].mean():.1f}; splits {df['Split'].value_counts().to_dict()}")
    else:
        print("EchoNet-Dynamic missing: request at https://echonet.github.io/dynamic/ (free registration + DUA)")
    db = DATA / PTBXL / "ptbxl_database.csv"
    print("PTB-XL metadata:", "present" if db.exists() else "missing (run --ptbxl --sample)")
    for proj in (MIMIC_ECG, MIMIC_ECHO):
        print(proj, "present" if (DATA / proj).exists() else "missing (credentialed)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ptbxl", action="store_true")
    ap.add_argument("--sample", action="store_true", help="with --ptbxl: metadata + 5 records only")
    ap.add_argument("--mimic-ecg", action="store_true")
    ap.add_argument("--mimic-echo", action="store_true")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--include-list", type=Path, default=None, help="text file of relative paths to fetch")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ran = False
    if args.ptbxl:
        cmd_ptbxl(args.sample)
        ran = True
    if args.mimic_ecg:
        cmd_mimic(MIMIC_ECG, args.metadata_only, args.include_list)
        ran = True
    if args.mimic_echo:
        cmd_mimic(MIMIC_ECHO, args.metadata_only, args.include_list)
        ran = True
    if args.check:
        cmd_check()
        ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
