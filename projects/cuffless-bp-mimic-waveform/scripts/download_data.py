#!/usr/bin/env python3
"""Index / download helpers for the cuffless-BP benchmark.

Open parts (no credentials):
    --db mimic3wdb   MIMIC-III Waveform Database Matched Subset: RECORDS index + record headers
    --db mimic4wdb   MIMIC-IV Waveform Database v0.1.0: RECORDS index + record headers
    --db vitaldb     VitalDB case list (+ a few cached cases with --sample)

Credentialed part (reads PHYSIONET_USER / PHYSIONET_PASS from the environment, never stores them):
    --db mimiciv-inputevents   MIMIC-IV v3.1 icu/inputevents + icu/icustays for bolus linkage

``--sample`` limits the index to the first N records that contain ECG + PLETH + ABP and, for
VitalDB, caches 60 s of three cases. Waveforms themselves are streamed at analysis time by
``cuffless_bp.wfdb_loader``; nothing here downloads terabytes.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

PHYSIONET_FILES = "https://physionet.org/files"
DBS = {
    "mimic3wdb": "mimic3wdb-matched/1.0",
    "mimic4wdb": "mimic4wdb/0.1.0",
}


def fetch_records(db_dir: str) -> list[str]:
    """RECORDS index of a PhysioNet database (open access)."""
    url = f"{PHYSIONET_FILES}/{db_dir}/RECORDS"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return [ln.strip() for ln in r.text.splitlines() if ln.strip()]


def index_records(db: str, out: Path, limit: int | None) -> None:
    """Write data/<db>/index.csv with the signals available in each record header."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cuffless_bp.wfdb_loader import describe_record  # noqa: E402

    db_dir = DBS[db]
    recs = fetch_records(db_dir)
    print(f"{db}: {len(recs)} records listed")
    out.mkdir(parents=True, exist_ok=True)
    rows, n_ok = [], 0
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record", "pn_dir", "subject_id", "has_ecg", "has_pleth", "has_abp", "fs", "n_samples", "base_datetime"])
        for rec in recs:
            if rec.endswith("n"):  # numerics records
                continue
            try:
                d = describe_record(rec, db_dir)
            except Exception as e:  # network / missing header
                print("  skip", rec, type(e).__name__)
                continue
            w.writerow([d["record"], d["pn_dir"], d["subject_id"], d["has_ecg"], d["has_pleth"], d["has_abp"],
                        d["fs"], d["n_samples"], d["base_datetime"]])
            rows.append(d)
            if d["has_ecg"] and d["has_pleth"] and d["has_abp"]:
                n_ok += 1
            if limit and n_ok >= limit:
                break
    print(f"  indexed {len(rows)} records, {n_ok} with ECG+PLETH+ABP -> {out / 'index.csv'}")


def vitaldb_cases(out: Path, sample: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://api.vitaldb.net/cases", timeout=120)
    r.raise_for_status()
    (out / "cases.csv").write_text(r.text)
    print("  wrote", out / "cases.csv")
    r = requests.get("https://api.vitaldb.net/trks", timeout=120)
    r.raise_for_status()
    (out / "trks.csv").write_text(r.text)
    print("  wrote", out / "trks.csv (track list per case; look for SNUADC/ART, SNUADC/PLETH, SNUADC/ECG_II)")
    if sample:
        try:
            import numpy as np
            import vitaldb
        except ImportError:
            sys.exit("pip install vitaldb to cache sample cases")
        tracks = ["SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"]
        cached = 0
        for caseid in vitaldb.find_cases(tracks)[:50]:
            arr = vitaldb.load_case(caseid, tracks, 1 / 125)  # resample to 125 Hz
            if arr is None or len(arr) < 125 * 60:
                continue
            np.savez_compressed(out / f"case_{caseid}.npz", x=arr[: 125 * 60].astype("float32"), fs=125, tracks=tracks)
            cached += 1
            if cached >= 3:
                break
        print(f"  cached {cached} sample cases (60 s each)")


def download_inputevents(out: Path) -> None:
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit("PHYSIONET_USER / PHYSIONET_PASS not set (credentialed MIMIC-IV access is required).")
    if shutil.which("wget") is None:
        sys.exit("wget is required.")
    out.mkdir(parents=True, exist_ok=True)
    for f in ["icu/inputevents.csv.gz", "icu/icustays.csv.gz", "icu/d_items.csv.gz", "hosp/patients.csv.gz"]:
        url = f"{PHYSIONET_FILES}/mimiciv/3.1/{f}"
        print("  wget", url)
        subprocess.run(["wget", "-N", "-c", "--user", user, "--password", pw, "-P", str(out), url], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, choices=["mimic3wdb", "mimic4wdb", "vitaldb", "mimiciv-inputevents"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after N records with ECG+PLETH+ABP")
    args = ap.parse_args()
    out = Path(args.out)
    if args.db in DBS:
        index_records(args.db, out / args.db, args.limit or (10 if args.sample else None))
    elif args.db == "vitaldb":
        vitaldb_cases(out / "vitaldb", args.sample)
    else:
        download_inputevents(out / "mimiciv")


if __name__ == "__main__":
    main()
