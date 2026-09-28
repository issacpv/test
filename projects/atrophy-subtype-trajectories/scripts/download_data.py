#!/usr/bin/env python3
"""OASIS-3 downloader via the NITRC-IR XNAT REST API + ADNI instructions/stub.

Credentials: OASIS_USER / OASIS_PASSWORD (aliases NITRC_USER / NITRC_PASSWORD); prompted if unset.
Endpoints mirror https://github.com/NrgXnat/oasis-scripts.

Examples
--------
  # inventory of subjects / MR sessions / FreeSurfer assessors / PUP assessors (CSV files)
  python scripts/download_data.py oasis-list --out data/oasis3/tables --sample 25

  # FreeSurfer stats (enough for the atrophy model)
  python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv \
      --out data/oasis3/freesurfer --sample 5

  # T1 + FLAIR for WMH segmentation
  python scripts/download_data.py oasis-mr --ids data/oasis3/tables/mr_ids.csv --out data/oasis3/mr \
      --scan-type T1w,FLAIR --sample 3

  # ADNI: print instructions and check expected tables
  python scripts/download_data.py adni --root data/adni
"""
from __future__ import annotations

import argparse
import csv
import getpass
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

XNAT_BASE = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"


def _credentials() -> tuple[str, str]:
    user = os.environ.get("OASIS_USER") or os.environ.get("NITRC_USER") or input("NITRC username: ").strip()
    pwd = os.environ.get("OASIS_PASSWORD") or os.environ.get("NITRC_PASSWORD") or getpass.getpass("NITRC password: ")
    return user, pwd


def xnat_session() -> requests.Session:
    user, pwd = _credentials()
    s = requests.Session()
    r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pwd), timeout=60)
    if r.status_code != 200:
        sys.exit(f"XNAT login failed ({r.status_code}); check credentials and DUA approval for {PROJECT}")
    s.cookies.set("JSESSIONID", r.text.strip())
    return s


def get_json(s: requests.Session, url: str) -> list[dict]:
    r = s.get(url, params={"format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json().get("ResultSet", {}).get("Result", [])


def download_zip(s: requests.Session, url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    z = out_dir / "download.zip"
    with s.get(url, params={"format": "zip"}, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with open(z, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out_dir)
    z.unlink()


def _ids(path: Path, cols: Iterable[str]) -> list[str]:
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        col = next((c for c in cols if c in (rd.fieldnames or [])), None)
        if col is None:
            fh.seek(0)
            return [r[0].strip() for r in csv.reader(fh) if r and r[0].strip()]
        return [r[col].strip() for r in rd if r.get(col, "").strip()]


def cmd_list(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = xnat_session()
    subjects = get_json(s, f"{XNAT_BASE}/data/projects/{PROJECT}/subjects")
    if a.sample:
        subjects = subjects[: a.sample]
    mr, fs, pet, pup = [], [], [], []
    for sub in subjects:
        label = sub.get("label") or sub["ID"]
        for e in get_json(s, f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{label}/experiments"):
            el = e.get("label", "")
            if "_MR_" in el:
                mr.append({"subject": label, "experiment_id": el})
                for x in get_json(s, f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{label}/experiments/{el}/assessors"):
                    if "Freesurfer" in x.get("label", ""):
                        fs.append({"subject": label, "experiment_id": el, "freesurfer_id": x["label"]})
            elif any(t in el for t in ("_PIB_", "_AV45_")):
                pet.append({"subject": label, "experiment_id": el})
                for x in get_json(s, f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{label}/experiments/{el}/assessors"):
                    if "PUP" in x.get("label", ""):
                        pup.append({"subject": label, "experiment_id": el, "pup_id": x["label"]})
    for name, rows in (("mr_ids.csv", mr), ("freesurfer_ids.csv", fs), ("pet_ids.csv", pet), ("pup_ids.csv", pup)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {out / name}: {len(rows)} rows")
    print(f"{len(subjects)} subjects, {len(mr)} MR sessions, {len(fs)} FreeSurfer, {len(pup)} PUP")


def _bulk(a: argparse.Namespace, kind: str) -> None:
    ids = _ids(Path(a.ids), {"mr": ["experiment_id", "MR ID"], "freesurfer": ["freesurfer_id", "FS ID"],
                             "pup": ["pup_id", "PUP ID"]}[kind])
    if a.sample:
        ids = ids[: a.sample]
    s = xnat_session()
    for i, oid in enumerate(ids, 1):
        subj = oid.split("_")[0]
        base = f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{subj}/experiments"
        if kind == "mr":
            url = f"{base}/{oid}/scans/{a.scan_type}/files"
        elif kind == "freesurfer":
            exp = oid.replace("_Freesurfer53_", "_MR_").replace("_Freesurfer_", "_MR_")
            url = f"{base}/{exp}/assessors/{oid}/files"
        else:
            url = f"{base}/{oid.replace('_PUPTIMECOURSE', '')}/assessors/{oid}/files"
        dest = Path(a.out) / oid
        if dest.exists() and any(dest.iterdir()):
            print(f"[{i}/{len(ids)}] {oid}: exists")
            continue
        print(f"[{i}/{len(ids)}] {oid}")
        try:
            download_zip(s, url, dest)
        except requests.HTTPError as exc:
            print(f"   failed: {exc}")


def cmd_adni(a: argparse.Namespace) -> None:
    print("ADNI requires an approved application (https://adni.loni.usc.edu/data-samples/access-data/).\n"
          "Download from https://ida.loni.usc.edu -> ADNI -> Study Data: ADNIMERGE.csv, UCBERKELEY_AMY_6MM*.csv,\n"
          "UCSFFSX*.csv (FreeSurfer ROIs), UCD_WMH*.csv (WMH volumes). Put them in <root>/tables/.")
    tables = Path(a.root) / "tables"
    expected = ["ADNIMERGE", "UCBERKELEY_AMY", "UCSFFS", "UCD_WMH"]
    missing = [e for e in expected if not list(tables.glob(f"{e}*.csv"))]
    print("missing:" if missing else "all expected tables present:", missing or tables)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("oasis-list"); s.add_argument("--out", default="data/oasis3/tables")
    s.add_argument("--sample", type=int, default=0); s.set_defaults(func=cmd_list)
    for name, kind in (("oasis-mr", "mr"), ("oasis-freesurfer", "freesurfer"), ("oasis-pup", "pup")):
        s = sub.add_parser(name)
        s.add_argument("--ids", required=True); s.add_argument("--out", required=True)
        s.add_argument("--sample", type=int, default=0)
        if kind == "mr":
            s.add_argument("--scan-type", default="ALL", help="e.g. ALL or T1w,FLAIR")
        s.set_defaults(func=lambda a, k=kind: _bulk(a, k))
    s = sub.add_parser("adni"); s.add_argument("--root", default="data/adni"); s.set_defaults(func=cmd_adni)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
