#!/usr/bin/env python3
"""OASIS-3 downloader (NITRC-IR XNAT REST API; DUA + credentials) focused on FLAIR sessions, PUP
Centiloid tables and FreeSurfer stats, plus an ADNI instructions/verification stub.

Credentials: OASIS_USER / OASIS_PASSWORD (aliases NITRC_USER / NITRC_PASSWORD); prompted if absent.

Examples
--------
    python scripts/download_data.py oasis-list --out data/oasis3/tables --sample 50
    python scripts/download_data.py oasis-flair --ids data/oasis3/tables/flair_sessions.csv --out data/oasis3/mr --sample 3
    python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv --out data/oasis3/freesurfer --sample 3
    python scripts/download_data.py adni --root data/adni
"""
from __future__ import annotations

import argparse
import csv
import getpass
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

XNAT_BASE = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"


def _credentials() -> tuple[str, str]:
    user = os.environ.get("OASIS_USER") or os.environ.get("NITRC_USER")
    pwd = os.environ.get("OASIS_PASSWORD") or os.environ.get("NITRC_PASSWORD")
    if not user:
        user = input("NITRC username: ").strip()
    if not pwd:
        pwd = getpass.getpass("NITRC password: ")
    return user, pwd


def _session() -> requests.Session:
    user, pwd = _credentials()
    s = requests.Session()
    r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pwd), timeout=60)
    r.raise_for_status()
    s.cookies.set("JSESSIONID", r.text.strip())
    return s


def _get_csv(s: requests.Session, url: str, params: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    params = dict(params or {})
    params.setdefault("format", "csv")
    r = s.get(url, params=params, timeout=300)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def _write_csv(rows: Iterable[Dict[str, str]], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def oasis_list(out: Path, sample: Optional[int]) -> None:
    """Write mr_sessions.csv, flair_sessions.csv (sessions with a FLAIR scan), pup.csv, freesurfer_ids.csv."""
    s = _session()
    mr = _get_csv(s, f"{XNAT_BASE}/data/projects/{PROJECT}/experiments",
                  {"xsiType": "xnat:mrSessionData", "columns": "ID,label,subject_label,scanner"})
    if sample:
        mr = mr[:sample]
    _write_csv(mr, out / "mr_sessions.csv")
    flair_rows = []
    for row in mr:
        scans = _get_csv(s, f"{XNAT_BASE}/data/experiments/{row['ID']}/scans", {"columns": "ID,type,series_description"})
        types = [sc.get("type", "") or sc.get("series_description", "") for sc in scans]
        fl = [sc["ID"] for sc, t in zip(scans, types) if "flair" in t.lower()]
        t1 = [sc["ID"] for sc, t in zip(scans, types) if "t1" in t.lower() or "mprage" in t.lower()]
        if fl:
            flair_rows.append({**row, "flair_scan_ids": ";".join(fl), "t1_scan_ids": ";".join(t1)})
    n = _write_csv(flair_rows, out / "flair_sessions.csv")
    print(f"MR sessions: {len(mr)}; with FLAIR: {n}")
    pup = _get_csv(s, f"{XNAT_BASE}/data/projects/{PROJECT}/experiments", {"xsiType": "xnat:petSessionData", "columns": "ID,label,subject_label"})
    print(f"PET sessions: {_write_csv(pup, out / 'pet_sessions.csv')} (Centiloid values are in the PUP spreadsheet on the project page)")
    fs = _get_csv(s, f"{XNAT_BASE}/data/projects/{PROJECT}/experiments", {"xsiType": "fs:fsData", "columns": "ID,label,subject_label"})
    print(f"FreeSurfer runs: {_write_csv(fs, out / 'freesurfer_ids.csv')}")


def _download_scan_zip(s: requests.Session, exp_label: str, scan_ids: List[str], dest: Path) -> None:
    for sid in scan_ids:
        url = f"{XNAT_BASE}/data/experiments/{exp_label}/scans/{sid}/resources/NIFTI/files"
        r = s.get(url, params={"format": "zip"}, timeout=1800)
        if r.status_code != 200:
            print(f"  scan {sid}: HTTP {r.status_code}")
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for member in z.namelist():
                if member.endswith((".nii.gz", ".json")):
                    target = dest / f"scan{sid}" / Path(member).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(member))
        print(f"  ok scan {sid}")


def oasis_flair(ids_csv: Path, out: Path, sample: Optional[int], with_t1: bool) -> None:
    s = _session()
    with ids_csv.open() as f:
        rows = list(csv.DictReader(f))
    if sample:
        rows = rows[:sample]
    for row in rows:
        label = row.get("label") or row.get("MR ID")
        print(label)
        ids = row.get("flair_scan_ids", "").split(";")
        if with_t1:
            ids += row.get("t1_scan_ids", "").split(";")
        _download_scan_zip(s, label, [i for i in ids if i], out / label)


def oasis_freesurfer(ids_csv: Path, out: Path, sample: Optional[int]) -> None:
    s = _session()
    with ids_csv.open() as f:
        rows = list(csv.DictReader(f))
    labels = [r.get("label") or r.get("FS ID") for r in rows]
    for lab in (labels[:sample] if sample else labels):
        dest = out / lab / "stats"
        if (dest / "aseg.stats").exists():
            continue
        r = s.get(f"{XNAT_BASE}/data/experiments/{lab}/resources/DATA/files", params={"format": "zip", "file": "stats/*"}, timeout=1200)
        if r.status_code != 200:
            print(f"failed {lab}: HTTP {r.status_code}")
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for member in z.namelist():
                if member.endswith(".stats"):
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / Path(member).name).write_bytes(z.read(member))
        print(f"ok {lab}")


def adni_check(root: Path) -> None:
    tables = root / "tables"
    print("ADNI is application-only: https://adni.loni.usc.edu/data-samples/access-data/")
    for name in ("ADNIMERGE.csv", "UCBERKELEY_AMY_6MM.csv"):
        print(f"  [{'x' if (tables / name).exists() else ' '}] {name}")
    wmh = list(tables.glob("*WMH*.csv")) if tables.exists() else []
    print(f"  [{'x' if wmh else ' '}] UC Davis WMH tables: {[p.name for p in wmh] or 'none found (search Study Data for UCD + WMH)'}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("oasis-list")
    a.add_argument("--out", type=Path, default=Path("data/oasis3/tables"))
    a.add_argument("--sample", type=int, default=None)
    b = sub.add_parser("oasis-flair")
    b.add_argument("--ids", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/oasis3/mr"))
    b.add_argument("--sample", type=int, default=None)
    b.add_argument("--with-t1", action="store_true")
    c = sub.add_parser("oasis-freesurfer")
    c.add_argument("--ids", type=Path, required=True)
    c.add_argument("--out", type=Path, default=Path("data/oasis3/freesurfer"))
    c.add_argument("--sample", type=int, default=None)
    d = sub.add_parser("adni")
    d.add_argument("--root", type=Path, default=Path("data/adni"))
    args = p.parse_args(argv)
    if args.cmd == "oasis-list":
        oasis_list(args.out, args.sample)
    elif args.cmd == "oasis-flair":
        oasis_flair(args.ids, args.out, args.sample, args.with_t1)
    elif args.cmd == "oasis-freesurfer":
        oasis_freesurfer(args.ids, args.out, args.sample)
    else:
        adni_check(args.root)


if __name__ == "__main__":
    main()
