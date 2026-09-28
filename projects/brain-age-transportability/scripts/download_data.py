#!/usr/bin/env python3
"""Download FreeSurfer stats tables for the three cohorts.

Usage
-----
    python scripts/download_data.py hcp-ya   [--sample N] [--subjects CSV]
    python scripts/download_data.py oasis3   [--sample N]
    python scripts/download_data.py hcp-lifespan          # prints NDA steps, checks layout

Credentials (never stored in the repo):
    HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY   (ConnectomeDB -> Amazon S3 Access)
    OASIS_USER / OASIS_PASS                             (XNAT Central after OASIS DUA)
    NDA_USERNAME / NDA_PASSWORD                         (used by nda-tools, not here)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HCP_BUCKET = "hcp-openaccess"
HCP_PREFIX = "HCP_1200"
STATS_FILES = ["aseg.stats", "lh.aparc.stats", "rh.aparc.stats", "wmparc.stats"]

XNAT = "https://central.xnat.org"
OASIS_PROJECT = "OASIS3"


# --------------------------------------------------------------------------- HCP-YA
def _hcp_subject_ids(csv_path: Optional[Path]) -> List[str]:
    """Subject ids from the ConnectomeDB export (column 'Subject')."""
    if csv_path is None or not csv_path.exists():
        print(f"[hcp-ya] subject list not found ({csv_path}); using a public sample id list", file=sys.stderr)
        # First subjects of the S1200 release; used only for --sample smoke tests.
        return ["100206", "100307", "100408", "100610", "101006", "101107", "101309", "101410", "101915", "102008"]
    import csv
    with csv_path.open() as fh:
        return [row["Subject"] for row in csv.DictReader(fh) if row.get("Subject")]


def download_hcp_ya(sample: Optional[int], subjects_csv: Optional[Path]) -> None:
    try:
        import boto3
    except ImportError:
        sys.exit("pip install boto3 (see requirements.txt)")
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not (key and secret):
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID and HCP_AWS_SECRET_ACCESS_KEY (ConnectomeDB -> Amazon S3 Access)")
    s3 = boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)
    subjects = _hcp_subject_ids(subjects_csv or DATA / "hcp_ya" / "subjects.csv")
    if sample:
        subjects = subjects[:sample]
    out_root = DATA / "hcp_ya" / "stats"
    n_ok = 0
    for sid in subjects:
        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in STATS_FILES:
            s3_key = f"{HCP_PREFIX}/{sid}/T1w/{sid}/stats/{fname}"
            target = out_dir / fname
            if target.exists():
                continue
            try:
                s3.download_file(HCP_BUCKET, s3_key, str(target))
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"[hcp-ya] {sid}/{fname}: {e}", file=sys.stderr)
        n_ok += 1
        print(f"[hcp-ya] {sid} done ({n_ok}/{len(subjects)})")


# --------------------------------------------------------------------------- OASIS-3
def _xnat_get(session: requests.Session, path: str, **params) -> dict:
    r = session.get(f"{XNAT}{path}", params={"format": "json", **params}, timeout=120)
    r.raise_for_status()
    return r.json()


def _oasis_list_subjects(session: requests.Session) -> List[str]:
    res = _xnat_get(session, f"/data/archive/projects/{OASIS_PROJECT}/subjects")
    return sorted(row["label"] for row in res["ResultSet"]["Result"])


def _oasis_freesurfer_assessors(session: requests.Session, subject: str) -> List[dict]:
    """FreeSurfer assessors are exposed as experiments of type fs:fsData."""
    res = _xnat_get(session, f"/data/archive/projects/{OASIS_PROJECT}/subjects/{subject}/experiments",
                    xsiType="fs:fsData")
    return res["ResultSet"]["Result"]


def _oasis_download_fs_stats(session: requests.Session, subject: str, fs_label: str, out_root: Path) -> None:
    """Download the assessor's file bundle and keep only stats/*.stats."""
    url = f"{XNAT}/data/archive/projects/{OASIS_PROJECT}/subjects/{subject}/experiments/{fs_label}/files"
    r = session.get(url, params={"format": "zip"}, timeout=600, stream=True)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    with zipfile.ZipFile(buf) as zf:
        for member in zf.namelist():
            if "/stats/" in member and member.endswith(".stats"):
                target = out_root / fs_label / "stats" / Path(member).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))


def download_oasis3(sample: Optional[int]) -> None:
    user, pw = os.environ.get("OASIS_USER"), os.environ.get("OASIS_PASS")
    if not (user and pw):
        sys.exit("Set OASIS_USER / OASIS_PASS (XNAT Central credentials, after the OASIS-3 DUA)")
    session = requests.Session()
    session.auth = (user, pw)
    # establishing a JSESSION avoids re-authenticating on every request
    r = session.post(f"{XNAT}/data/JSESSION", timeout=60)
    r.raise_for_status()
    subjects = _oasis_list_subjects(session)
    if sample:
        subjects = subjects[:sample]
    out_root = DATA / "oasis3" / "freesurfer"
    for i, subj in enumerate(subjects, 1):
        try:
            assessors = _oasis_freesurfer_assessors(session, subj)
        except requests.HTTPError as e:
            print(f"[oasis3] {subj}: {e}", file=sys.stderr)
            continue
        for a in assessors:
            label = a.get("label", "")
            if "Freesurfer" not in label:
                continue
            if (out_root / label / "stats" / "aseg.stats").exists():
                continue
            try:
                _oasis_download_fs_stats(session, subj, label, out_root)
                print(f"[oasis3] {label}")
            except (requests.HTTPError, zipfile.BadZipFile) as e:
                print(f"[oasis3] {label}: {e}", file=sys.stderr)
        print(f"[oasis3] {subj} done ({i}/{len(subjects)})")
    print("Remember to export MR Sessions and ADRC Clinical Data CSVs from the XNAT UI (data/README.md, section 3).")


# --------------------------------------------------------------------------- Lifespan
def hcp_lifespan_instructions() -> None:
    print(
        "HCP-Aging / HCP-Development are distributed only through the NIMH Data Archive.\n"
        "  1. NDA account + institutional Data Use Certification (permission group 'Connectomes Related to Human Disease').\n"
        "  2. Build a data package with collections 2847 (HCP-A) and 2846 (HCP-D), including ndar_subject01 and\n"
        "     the FreeSurfer/structural-preprocessed outputs.\n"
        "  3. pip install nda-tools && downloadcmd -dp <PACKAGE_ID> -d data/hcp_lifespan -u $NDA_USERNAME -p $NDA_PASSWORD\n"
    )
    root = DATA / "hcp_lifespan"
    found = list(root.rglob("aseg.stats")) if root.exists() else []
    print(f"Found {len(found)} aseg.stats files under {root}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cohort", choices=["hcp-ya", "oasis3", "hcp-lifespan"])
    p.add_argument("--sample", type=int, default=None, help="only the first N subjects")
    p.add_argument("--subjects", type=Path, default=None, help="CSV with a 'Subject' column (hcp-ya)")
    args = p.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    if args.cohort == "hcp-ya":
        download_hcp_ya(args.sample, args.subjects)
    elif args.cohort == "oasis3":
        download_oasis3(args.sample)
    else:
        hcp_lifespan_instructions()


if __name__ == "__main__":
    main()
