#!/usr/bin/env python3
"""Data access for faers_ehr.

    --source openfda        2x2 tables (drug x outcome) through the openFDA API (open; key optional)
    --source openfda-bulk   quarterly bulk JSON partitions listed by https://api.fda.gov/download.json
    --source mimiciv        MIMIC-IV v3.1 tables used by the project (credentialed; env credentials)
    --source mimic-iv-ecg   MIMIC-IV-ECG machine measurements (open on PhysioNet)

``--sample`` keeps everything small: 3 ingredients x 2 outcomes for openFDA, one bulk partition,
or only the dictionary tables for MIMIC-IV.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PHYSIONET = "https://physionet.org/files"
MIMICIV_TABLES = [
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/prescriptions.csv.gz", "hosp/emar.csv.gz",
    "hosp/emar_detail.csv.gz", "hosp/labevents.csv.gz", "hosp/d_labitems.csv.gz", "hosp/diagnoses_icd.csv.gz",
    "icu/icustays.csv.gz", "icu/inputevents.csv.gz",
]
MIMICIV_SAMPLE = ["hosp/d_labitems.csv.gz", "hosp/patients.csv.gz"]
ECG_FILES = ["machine_measurements.csv", "record_list.csv"]

SAMPLE_INGREDIENTS = ["amiodarone", "trimethoprim", "vancomycin"]


def openfda_two_by_two(ingredients: list[str], outcomes: list[str] | None, out: Path) -> None:
    from faers_ehr.openfda_client import OUTCOME_PT, OpenFDAClient

    client = OpenFDAClient()  # reads OPENFDA_API_KEY
    outcomes = outcomes or list(OUTCOME_PT)
    out.mkdir(parents=True, exist_ok=True)
    df = client.drug_event_table(ingredients, outcomes, suspect_only=True)
    df["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = out / "two_by_two.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} drug x outcome rows)")


def openfda_bulk(quarters: list[str] | None, out: Path, sample: bool) -> None:
    idx = requests.get("https://api.fda.gov/download.json", timeout=60).json()
    parts = idx["results"]["drug"]["event"]["partitions"]
    files = [p["file"] for p in parts]
    if quarters:
        files = [f for f in files if any(q in f for q in quarters)]
    if sample:
        files = files[-1:]
    out.mkdir(parents=True, exist_ok=True)
    for url in files:
        dest = out / url.rsplit("/", 1)[-1]
        if dest.exists():
            continue
        print("  downloading", url)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    print(f"{len(files)} partitions in {out}")


def _physionet_credentials() -> tuple[str, str]:
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit("PHYSIONET_USER / PHYSIONET_PASS not set. Credentialed access is required for MIMIC-IV.")
    return user, pw


def mimiciv(out: Path, sample: bool) -> None:
    user, pw = _physionet_credentials()
    if shutil.which("wget") is None:
        sys.exit("wget is required.")
    for rel in (MIMICIV_SAMPLE if sample else MIMICIV_TABLES):
        url = f"{PHYSIONET}/mimiciv/3.1/{rel}"
        dest_dir = out / "mimiciv" / "3.1" / rel.rsplit("/", 1)[0]
        dest_dir.mkdir(parents=True, exist_ok=True)
        print("  wget", url)
        subprocess.run(["wget", "-N", "-c", "--user", user, "--password", pw, "-P", str(dest_dir), url], check=True)


def mimic_iv_ecg(out: Path) -> None:
    dest = out / "mimic-iv-ecg" / "1.0"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ECG_FILES:
        url = f"{PHYSIONET}/mimic-iv-ecg/1.0/{f}"
        print("  downloading", url)
        with requests.get(url, stream=True, timeout=600) as r:
            if r.status_code == 401:
                # some mirrors require a login even for open data; fall back to wget with credentials
                user, pw = _physionet_credentials()
                subprocess.run(["wget", "-N", "-c", "--user", user, "--password", pw, "-P", str(dest), url], check=True)
                continue
            r.raise_for_status()
            with open(dest / f, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["openfda", "openfda-bulk", "mimiciv", "mimic-iv-ecg"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--ingredients", default=None, help="text file, one ingredient (generic name) per line")
    ap.add_argument("--outcomes", nargs="*", default=None, help="outcome names from OUTCOME_PT")
    ap.add_argument("--quarters", nargs="*", default=None, help="e.g. 2023q1 2023q2")
    args = ap.parse_args()
    out = Path(args.out)
    if args.source == "openfda":
        if args.sample:
            ingredients, outcomes = SAMPLE_INGREDIENTS, ["hyperkalaemia", "qt_prolongation"]
        else:
            if not args.ingredients:
                sys.exit("--ingredients file required (or --sample)")
            ingredients = [ln.strip() for ln in open(args.ingredients) if ln.strip() and not ln.startswith("#")]
            outcomes = args.outcomes
        openfda_two_by_two(ingredients, outcomes, out / "openfda")
    elif args.source == "openfda-bulk":
        openfda_bulk(args.quarters, out / "openfda" / "bulk", args.sample)
    elif args.source == "mimiciv":
        mimiciv(out, args.sample)
    else:
        mimic_iv_ecg(out)


if __name__ == "__main__":
    main()
