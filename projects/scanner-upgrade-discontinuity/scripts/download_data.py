#!/usr/bin/env python3
"""Downloader for the DUA-gated OASIS-3 tables/FreeSurfer outputs (NITRC-IR XNAT REST API)
and an instructions/verification stub for ADNI (LONI IDA, application only).

OASIS-3 credentials are read from OASIS_USER / OASIS_PASSWORD (aliases NITRC_USER /
NITRC_PASSWORD); if absent you are prompted interactively.

Examples
--------
List MR sessions (with scanner model) and FreeSurfer runs into CSVs (no imaging download):
    python scripts/download_data.py oasis-list --out data/oasis3/tables --sample 50

Download FreeSurfer stats files for the first 5 runs listed in a CSV:
    python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv \
        --out data/oasis3/freesurfer --sample 5

Check the expected ADNI tables and print instructions:
    python scripts/download_data.py adni --root data/adni

The REST calls mirror the official scripts at https://github.com/NrgXnat/oasis-scripts.
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
    """Open an XNAT session (JSESSIONID) so that credentials are sent once."""
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


# ---------------------------------------------------------------------------
# OASIS-3 listing
# ---------------------------------------------------------------------------
def oasis_list(out: Path, sample: Optional[int]) -> None:
    """Write subjects.csv, mr_sessions.csv (with scanner column when exposed) and freesurfer_ids.csv."""
    s = _session()
    subjects = _get_csv(s, f"{XNAT_BASE}/data/projects/{PROJECT}/subjects", {"columns": "ID,label"})
    if sample:
        subjects = subjects[:sample]
    n = _write_csv(subjects, out / "subjects.csv")
    print(f"subjects: {n}")

    # MR sessions: request scanner and date-offset columns; XNAT ignores unknown columns.
    mr = _get_csv(
        s,
        f"{XNAT_BASE}/data/projects/{PROJECT}/experiments",
        {"xsiType": "xnat:mrSessionData", "columns": "ID,label,subject_label,scanner,scanner/manufacturer,scanner/model"},
    )
    if sample:
        keep = {r["label"] for r in subjects}
        mr = [r for r in mr if r.get("subject_label") in keep]
    n = _write_csv(mr, out / "mr_sessions.csv")
    print(f"MR sessions: {n} (columns: {list(mr[0].keys()) if mr else 'none'})")

    fs = _get_csv(
        s,
        f"{XNAT_BASE}/data/projects/{PROJECT}/experiments",
        {"xsiType": "fs:fsData", "columns": "ID,label,subject_label"},
    )
    if sample:
        fs = [r for r in fs if r.get("subject_label") in keep]
    n = _write_csv(fs, out / "freesurfer_ids.csv")
    print(f"FreeSurfer runs: {n}")


def oasis_freesurfer(ids_csv: Path, out: Path, sample: Optional[int]) -> None:
    """Download the stats/ folder of each FreeSurfer run (aseg.stats, ?h.aparc.stats)."""
    s = _session()
    with ids_csv.open() as f:
        rows = list(csv.DictReader(f))
    labels = [r.get("label") or r.get("FS ID") or r.get("ID") for r in rows]
    labels = [x for x in labels if x]
    if sample:
        labels = labels[:sample]
    for lab in labels:
        dest = out / lab
        if (dest / "stats" / "aseg.stats").exists():
            print(f"skip {lab}")
            continue
        url = f"{XNAT_BASE}/data/experiments/{lab}/resources/DATA/files"
        r = s.get(url, params={"format": "zip", "file": "stats/*"}, timeout=1200)
        if r.status_code != 200:
            print(f"failed {lab}: HTTP {r.status_code}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for member in z.namelist():
                if "/stats/" in member and member.endswith(".stats"):
                    target = dest / "stats" / Path(member).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(member))
        print(f"ok {lab}")


# ---------------------------------------------------------------------------
# ADNI stub
# ---------------------------------------------------------------------------
ADNI_EXPECTED = ["ADNIMERGE.csv", "MRILIST.csv", "MRIMETA.csv", "MRI3META.csv"]
ADNI_FS_PREFIXES = ["UCSFFSL", "UCSFFSX"]


def adni_check(root: Path) -> None:
    tables = root / "tables"
    print("ADNI is application-only (https://adni.loni.usc.edu/data-samples/access-data/).")
    print("Download the following into", tables)
    for name in ADNI_EXPECTED:
        print(f"  [{'x' if (tables / name).exists() else ' '}] {name}")
    fs = [p.name for p in tables.glob("UCSFFS*.csv")] if tables.exists() else []
    print(f"  [{'x' if fs else ' '}] UCSFFSL*/UCSFFSX* FreeSurfer tables: {fs or 'none found'}")
    if os.environ.get("IDA_USER"):
        print("IDA_USER is set, but the IDA has no public bulk API; use the IDA download manager.")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("oasis-list", help="list OASIS-3 subjects / MR sessions / FreeSurfer runs")
    a.add_argument("--out", type=Path, default=Path("data/oasis3/tables"))
    a.add_argument("--sample", type=int, default=None)

    b = sub.add_parser("oasis-freesurfer", help="download FreeSurfer stats files")
    b.add_argument("--ids", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/oasis3/freesurfer"))
    b.add_argument("--sample", type=int, default=None)

    c = sub.add_parser("adni", help="check ADNI tables and print instructions")
    c.add_argument("--root", type=Path, default=Path("data/adni"))

    args = p.parse_args(argv)
    if args.cmd == "oasis-list":
        oasis_list(args.out, args.sample)
    elif args.cmd == "oasis-freesurfer":
        oasis_freesurfer(args.ids, args.out, args.sample)
    elif args.cmd == "adni":
        adni_check(args.root)


if __name__ == "__main__":
    main()
