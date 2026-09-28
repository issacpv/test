#!/usr/bin/env python3
"""Download the raw ICU databases used by icu_transport.

PhysioNet resources (MIMIC-IV, eICU-CRD, HiRID) are credentialed: the script reads
``PHYSIONET_USER`` and ``PHYSIONET_PASS`` from the environment and never stores them.
AmsterdamUMCdb is distributed by Amsterdam Medical Data Science after signing an
end-user licence; pass the emailed link with ``--aumcdb-url``.

Examples
--------
    python scripts/download_data.py --db mimiciv --sample
    python scripts/download_data.py --db eicu --out data/raw
    python scripts/download_data.py --db hirid
    python scripts/download_data.py --db aumcdb --aumcdb-url "https://.../AmsterdamUMCdb-v1.0.2.zip"

``--sample`` fetches only the small dictionary / demographic tables so the ontology can be
verified without pulling tens of gigabytes.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PHYSIONET = "https://physionet.org/files"

# Files fetched by --sample (small, enough for verify_ontology + stay tables).
SAMPLE_FILES = {
    "mimiciv": [
        "mimiciv/3.1/hosp/d_labitems.csv.gz",
        "mimiciv/3.1/icu/d_items.csv.gz",
        "mimiciv/3.1/hosp/patients.csv.gz",
        "mimiciv/3.1/hosp/admissions.csv.gz",
        "mimiciv/3.1/icu/icustays.csv.gz",
    ],
    "eicu": [
        "eicu-crd/2.0/patient.csv.gz",
        "eicu-crd/2.0/hospital.csv.gz",
    ],
    "hirid": [
        "hirid/1.1.1/reference_data.tar.gz",
    ],
}

FULL_DIRS = {
    "mimiciv": ["mimiciv/3.1/hosp/", "mimiciv/3.1/icu/"],
    "eicu": ["eicu-crd/2.0/"],
    "hirid": ["hirid/1.1.1/raw_stage/", "hirid/1.1.1/reference_data.tar.gz"],
}


def _credentials() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USER")
    pw = os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit(
            "PHYSIONET_USER / PHYSIONET_PASS not set. Complete CITI training and sign the DUA at "
            "https://physionet.org, then `export PHYSIONET_USER=... PHYSIONET_PASS=...`."
        )
    return user, pw


def _wget(url: str, out: Path, user: str, pw: str, recursive: bool) -> None:
    if shutil.which("wget") is None:
        sys.exit("wget is required (apt install wget / brew install wget).")
    cmd = ["wget", "-N", "-c", "--user", user, "--password", pw, "-P", str(out)]
    if recursive:
        cmd += ["-r", "-np", "-nH", "--cut-dirs=1"]  # drop 'files/' from the local path
    else:
        cmd += ["-x", "-nH", "--cut-dirs=1"]
    cmd.append(url)
    # never echo the password
    print("  wget", url)
    subprocess.run(cmd, check=True)


def download_physionet(db: str, out: Path, sample: bool) -> None:
    user, pw = _credentials()
    out.mkdir(parents=True, exist_ok=True)
    if sample:
        for rel in SAMPLE_FILES[db]:
            _wget(f"{PHYSIONET}/{rel}", out, user, pw, recursive=False)
    else:
        for rel in FULL_DIRS[db]:
            _wget(f"{PHYSIONET}/{rel}", out, user, pw, recursive=rel.endswith("/"))
    if db == "hirid":
        _extract_hirid(out / "hirid" / "1.1.1")


def _extract_hirid(root: Path) -> None:
    import tarfile

    for name in ["reference_data.tar.gz", "raw_stage/observation_tables_csv.tar.gz",
                 "raw_stage/pharma_records_csv.tar.gz"]:
        tar = root / name
        if tar.exists():
            print("  extracting", tar)
            with tarfile.open(tar) as tf:
                tf.extractall(tar.parent)


def download_aumcdb(url: str | None, out: Path) -> None:
    if not url:
        sys.exit(
            "AmsterdamUMCdb requires an end-user licence: register at "
            "https://amsterdammedicaldatascience.nl/amsterdamumcdb/ and pass the emailed link "
            "with --aumcdb-url."
        )
    dest = out / "aumcdb" / "1.0.2"
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget") is None:
        sys.exit("wget is required.")
    print("  wget (licensed link)")
    subprocess.run(["wget", "-c", "-P", str(dest), url], check=True)
    print("Extract the archive into", dest, "and gzip the large CSVs (DuckDB reads .csv.gz).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, choices=["mimiciv", "eicu", "hirid", "aumcdb"])
    ap.add_argument("--out", default="data/raw", help="output root (default data/raw)")
    ap.add_argument("--sample", action="store_true", help="only dictionaries and demographic tables")
    ap.add_argument("--aumcdb-url", default=None, help="licensed download link for AmsterdamUMCdb")
    args = ap.parse_args()

    out = Path(args.out)
    if args.db == "aumcdb":
        download_aumcdb(args.aumcdb_url, out)
    else:
        download_physionet(args.db, out, args.sample)
    print("Done. Set ICU_<DB>_ROOT environment variables as described in data/README.md.")


if __name__ == "__main__":
    main()
