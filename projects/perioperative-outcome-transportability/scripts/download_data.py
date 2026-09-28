#!/usr/bin/env python
"""Downloader for perioperative-outcome-transportability.

Open parts: MIMIC-IV demo (dry runs) and the VitalDB case index.
Credentialed parts (INSPIRE, MIMIC-IV) use wget with PHYSIONET_USERNAME /
PHYSIONET_PASSWORD from the environment.  MOVER must be downloaded manually
after signing the UCI data-use agreement (https://mover.ics.uci.edu/).

Examples
--------
python scripts/download_data.py --sample
python scripts/download_data.py --vitaldb-index
python scripts/download_data.py --inspire [--inspire-version 1.3]
python scripts/download_data.py --mimiciv
python scripts/download_data.py --check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PN = "https://physionet.org/files"

DEMO_TABLES = [
    "hosp/admissions", "hosp/patients", "hosp/transfers", "hosp/services", "hosp/procedures_icd",
    "hosp/diagnoses_icd", "hosp/labevents", "hosp/d_labitems", "icu/icustays",
]


def _wget(url: str, dest_dir: Path, credentialed: bool, recursive: bool = False) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-N", "-c", "-q", "--show-progress"]
    if recursive:
        cmd += ["-r", "-np", "-nH", "--cut-dirs=3", "-P", str(dest_dir)]
    else:
        cmd += ["-P", str(dest_dir)]
    if credentialed:
        user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
        if not user or not pwd:
            print("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD for credentialed resources.", file=sys.stderr)
            return 2
        cmd += ["--user", user, "--password", pwd]
    cmd.append(url)
    return subprocess.call(cmd)


def fetch_demo() -> None:
    for t in DEMO_TABLES:
        sub, name = t.split("/")
        rc = _wget(f"{PN}/mimic-iv-demo/2.2/{sub}/{name}.csv.gz", DATA / "mimic-iv-demo" / sub, credentialed=False)
        if rc != 0:
            print(f"failed: {t}", file=sys.stderr)
            return
    print("MIMIC-IV demo tables in", DATA / "mimic-iv-demo")


def fetch_mimiciv() -> None:
    for t in DEMO_TABLES:
        sub, name = t.split("/")
        rc = _wget(f"{PN}/mimiciv/3.1/{sub}/{name}.csv.gz", DATA / "mimiciv" / sub, credentialed=True)
        if rc != 0:
            print(f"failed: {t}", file=sys.stderr)
            return


def fetch_inspire(version: str) -> None:
    rc = _wget(f"{PN}/inspire/{version}/", DATA / "inspire", credentialed=True, recursive=True)
    if rc != 0:
        print("INSPIRE download failed; check version at https://physionet.org/content/inspire/", file=sys.stderr)


def fetch_vitaldb_index() -> None:
    import pandas as pd

    dest = DATA / "vitaldb"
    dest.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv("https://api.vitaldb.net/cases")
    trks = pd.read_csv("https://api.vitaldb.net/trks")
    cases.to_csv(dest / "cases.csv", index=False)
    trks.to_csv(dest / "trks.csv", index=False)
    print(f"{len(cases)} cases, {len(trks)} tracks; departments:")
    if "department" in cases:
        print(cases["department"].value_counts().to_string())


def check() -> None:
    parts = {
        "inspire operations": list((DATA / "inspire").glob("**/operations*")) if (DATA / "inspire").exists() else [],
        "mover surgery info": list((DATA / "mover").glob("**/*urgery*")) if (DATA / "mover").exists() else [],
        "mimiciv transfers": [p for p in [DATA / "mimiciv" / "hosp" / "transfers.csv.gz"] if p.exists()],
        "mimic-iv-demo transfers": [p for p in [DATA / "mimic-iv-demo" / "hosp" / "transfers.csv.gz"] if p.exists()],
        "vitaldb cases": [p for p in [DATA / "vitaldb" / "cases.csv"] if p.exists()],
    }
    for k, v in parts.items():
        print(f"{'OK ' if v else '-- '} {k}: {v[0] if v else 'missing'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="MIMIC-IV demo v2.2 (open)")
    ap.add_argument("--mimiciv", action="store_true")
    ap.add_argument("--inspire", action="store_true")
    ap.add_argument("--inspire-version", default="1.3")
    ap.add_argument("--vitaldb-index", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.sample:
        fetch_demo()
    if a.mimiciv:
        fetch_mimiciv()
    if a.inspire:
        fetch_inspire(a.inspire_version)
    if a.vitaldb_index:
        fetch_vitaldb_index()
    if a.check or not any([a.sample, a.mimiciv, a.inspire, a.vitaldb_index]):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
