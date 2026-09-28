#!/usr/bin/env python3
"""Download MIMIC-IV / eICU-CRD tables used by ``delirium_dyn``.

* ``--sample``: the open MIMIC-IV demo (v2.2) — no credentials.
* ``--db mimiciv|eicu``: credentialed full download (``PHYSIONET_USER`` / ``PHYSIONET_PASS`` from the
  environment, passed to ``wget`` only); ``--tables-only`` restricts to the tables this project reads.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --db mimiciv --tables-only
    python scripts/download_data.py --db eicu --tables-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PHYSIONET_FILES = "https://physionet.org/files"
DEMO = "mimic-iv-demo/2.2"
DBS = {"mimiciv": "mimiciv/3.1", "eicu": "eicu-crd/2.0"}
TABLES = {
    "mimiciv": [
        "icu/d_items.csv.gz", "icu/chartevents.csv.gz", "icu/inputevents.csv.gz", "icu/procedureevents.csv.gz",
        "icu/icustays.csv.gz", "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/d_labitems.csv.gz",
        "hosp/labevents.csv.gz",
    ],
    "eicu": [
        "nurseCharting.csv.gz", "infusionDrug.csv.gz", "vitalPeriodic.csv.gz", "lab.csv.gz", "patient.csv.gz",
        "hospital.csv.gz", "apachePatientResult.csv.gz",
    ],
}


def _wget() -> str:
    exe = shutil.which("wget")
    if exe is None:
        sys.exit("wget not found; install it or use the manual commands in data/README.md")
    return exe


def _credentials() -> tuple[str, str]:
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit("PHYSIONET_USER / PHYSIONET_PASS not set (see data/README.md).")
    return user, pw


def _run(cmd: list[str]) -> None:
    shown = [c if not c.startswith("--password=") else "--password=***" for c in cmd]
    print("+", " ".join(shown), flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit("download failed")


def download_sample(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _run([_wget(), "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-q", "--show-progress", f"{PHYSIONET_FILES}/{DEMO}/", "-P", str(out)])


def download_db(db: str, out: Path, tables_only: bool) -> None:
    user, pw = _credentials()
    rel = DBS[db]
    base = [_wget(), "-N", "-c", "-q", "--show-progress", f"--user={user}", f"--password={pw}"]
    if tables_only:
        for t in TABLES[db]:
            dest = out / rel / t
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run(base + ["-O", str(dest), f"{PHYSIONET_FILES}/{rel}/{t}"])
    else:
        _run(base + ["-r", "-np", "-nH", "--cut-dirs=1", f"{PHYSIONET_FILES}/{rel}/", "-P", str(out)])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--db", choices=sorted(DBS))
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args(argv)
    if a.sample:
        download_sample(a.out)
    else:
        download_db(a.db, a.out, a.tables_only)


if __name__ == "__main__":
    main()
