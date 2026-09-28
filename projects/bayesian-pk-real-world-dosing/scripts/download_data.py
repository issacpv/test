#!/usr/bin/env python3
"""Download MIMIC-IV tables used by ``bayes_pk``.

* ``--sample``: the open-access MIMIC-IV demo (v2.2, 100 patients) — no credentials needed.
* ``--full``: MIMIC-IV v3.1 ``hosp`` + ``icu`` modules (PhysioNet credentialed). Credentials are read
  from ``PHYSIONET_USER`` / ``PHYSIONET_PASS``; they are passed to ``wget`` and never written to disk.
* ``--tables-only``: with ``--full``, fetch only the tables this project reads.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --full --tables-only
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
FULL = "mimiciv/3.1"

PROJECT_TABLES = [
    "hosp/d_labitems.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/emar.csv.gz",
    "hosp/emar_detail.csv.gz",
    "hosp/prescriptions.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/omr.csv.gz",
    "icu/d_items.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/procedureevents.csv.gz",
    "icu/outputevents.csv.gz",
]


def _wget() -> str:
    exe = shutil.which("wget")
    if exe is None:
        sys.exit("wget not found; install it or download manually (see data/README.md).")
    return exe


def _credentials() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USER")
    pw = os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit(
            "PHYSIONET_USER / PHYSIONET_PASS are not set. Complete CITI training and sign the MIMIC-IV "
            "DUA at https://physionet.org/content/mimiciv/3.1/ then export both variables."
        )
    return user, pw


def _run(cmd: list[str]) -> None:
    print("+", " ".join(c if "--password" not in c else "--password=***" for c in cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(f"command failed with exit code {res.returncode}")


def download_sample(out: Path) -> None:
    """Fetch the open MIMIC-IV demo (hosp + icu) recursively."""
    out.mkdir(parents=True, exist_ok=True)
    _run([_wget(), "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-q", "--show-progress",
          f"{PHYSIONET_FILES}/{DEMO}/", "-P", str(out)])
    print(f"demo written under {out / DEMO}")


def download_full(out: Path, tables_only: bool) -> None:
    user, pw = _credentials()
    out.mkdir(parents=True, exist_ok=True)
    base = [_wget(), "-N", "-c", "-q", "--show-progress", f"--user={user}", f"--password={pw}"]
    if tables_only:
        for rel in PROJECT_TABLES:
            dest = out / FULL / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run(base + ["-O", str(dest), f"{PHYSIONET_FILES}/{FULL}/{rel}"])
    else:
        _run(base + ["-r", "-np", "-nH", "--cut-dirs=1", f"{PHYSIONET_FILES}/{FULL}/", "-P", str(out)])
    print(f"MIMIC-IV written under {out / FULL}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="open MIMIC-IV demo (no credentials)")
    g.add_argument("--full", action="store_true", help="MIMIC-IV v3.1 (credentialed)")
    p.add_argument("--tables-only", action="store_true", help="with --full: only project tables")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args(argv)
    if a.sample:
        download_sample(a.out)
    else:
        download_full(a.out, a.tables_only)


if __name__ == "__main__":
    main()
