#!/usr/bin/env python
"""Downloader for the respiratory mechanics project.

Open parts: MIMIC-IV demo and eICU-CRD demo (PhysioNet, no credentials) and
the Kaggle artificial-lung dataset (Kaggle API token). Credentialed parts
(MIMIC-IV, eICU-CRD, HiRID) read ``PHYSIONET_USERNAME`` / ``PHYSIONET_PASSWORD``.

    python scripts/download_data.py --demo
    python scripts/download_data.py --kaggle
    python scripts/download_data.py --mimic | --eicu | --hirid
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
FILES = "https://physionet.org/files"

DEMO_FILES = {
    "mimic-iv-demo/2.2": [
        "icu/chartevents.csv.gz", "icu/d_items.csv.gz", "icu/icustays.csv.gz", "icu/procedureevents.csv.gz",
        "hosp/admissions.csv.gz", "hosp/patients.csv.gz",
    ],
    "eicu-crd-demo/2.0.1": [
        "respiratoryCharting.csv.gz", "respiratoryCare.csv.gz", "patient.csv.gz", "apachePatientResult.csv.gz",
    ],
}
FULL_FILES = {
    "mimiciv/3.1": [
        "icu/chartevents.csv.gz", "icu/d_items.csv.gz", "icu/icustays.csv.gz", "icu/procedureevents.csv.gz",
        "hosp/admissions.csv.gz", "hosp/patients.csv.gz",
    ],
    "eicu-crd/2.0": [
        "respiratoryCharting.csv.gz", "respiratoryCare.csv.gz", "patient.csv.gz", "apachePatientResult.csv.gz",
    ],
    "hirid/1.1.1": [
        "hirid_variable_reference.csv",
        "reference_data.tar.gz",
        "raw_stage/observation_tables_parquet.tar.gz",
    ],
}


def fetch(url: str, dest: Path, auth: tuple[str, str] | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print("exists", dest.relative_to(ROOT))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300, auth=auth) as r:
        if r.status_code == 404:
            print("not found (check version/path on PhysioNet):", url)
            return
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("fetched", dest.relative_to(ROOT))


def auth_from_env() -> tuple[str, str]:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    if not u or not p:
        sys.exit("Set PHYSIONET_USERNAME / PHYSIONET_PASSWORD (credentialed PhysioNet access).")
    return u, p


def download_project(project: str, files: list[str], auth: tuple[str, str] | None) -> None:
    for rel in files:
        fetch(f"{FILES}/{project}/{rel}", DATA / project / rel, auth=auth)


def cmd_kaggle() -> None:
    dest = DATA / "kaggle-ventilator"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "ventilator-pressure-prediction", "-p", str(dest)], check=True)
    except FileNotFoundError:
        sys.exit("kaggle CLI not found: pip install kaggle and place ~/.kaggle/kaggle.json")
    for z in dest.glob("*.zip"):
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(dest)], check=False)
    print("Kaggle artificial-lung data in", dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="open MIMIC-IV demo + eICU demo tables")
    ap.add_argument("--kaggle", action="store_true", help="artificial-lung dataset via kaggle CLI")
    ap.add_argument("--mimic", action="store_true")
    ap.add_argument("--eicu", action="store_true")
    ap.add_argument("--hirid", action="store_true")
    args = ap.parse_args()
    ran = False
    if args.demo:
        for proj, files in DEMO_FILES.items():
            download_project(proj, files, None)
        ran = True
    if args.kaggle:
        cmd_kaggle()
        ran = True
    for flag, proj in (("mimic", "mimiciv/3.1"), ("eicu", "eicu-crd/2.0"), ("hirid", "hirid/1.1.1")):
        if getattr(args, flag):
            download_project(proj, FULL_FILES[proj], auth_from_env())
            ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
