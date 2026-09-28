#!/usr/bin/env python
"""Download HCP S1200 MEG resting-state data from the HCP open-access S3 bucket.

Credentials: HCP-issued AWS keys from ConnectomeDB, read from the environment
(``HCP_AWS_ACCESS_KEY_ID`` / ``HCP_AWS_SECRET_ACCESS_KEY``; falls back to the standard
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``). Nothing is stored in the repository.

Examples
--------
List subjects with MEG data::

    python scripts/download_data.py --list-subjects

Smoke test (one subject, one resting run, preprocessed + raw ECG-bearing files)::

    python scripts/download_data.py --sample

All resting runs + anatomy for a subject list::

    python scripts/download_data.py --subjects-file data/hcp_meg_subjects.txt --rest --anatomy

Restricted data (zygosity) cannot be downloaded by script; see data/README.md.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "hcp-openaccess"
PREFIX = "HCP_1200"
REST_SCANS = ("3-Restin", "4-Restin", "5-Restin")
LOG = logging.getLogger("download_data")


def _credentials() -> tuple[str, str]:
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise SystemExit("Set HCP_AWS_ACCESS_KEY_ID and HCP_AWS_SECRET_ACCESS_KEY (from ConnectomeDB) first.")
    return key, secret


def _client():
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install boto3 (or use the aws CLI commands in data/README.md)") from exc
    key, secret = _credentials()
    return boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)


def iter_keys(s3, prefix: str, delimiter: str = "") -> Iterator[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    for page in paginator.paginate(**kwargs):
        if delimiter:
            for cp in page.get("CommonPrefixes", []):
                yield {"Prefix": cp["Prefix"]}
        for obj in page.get("Contents", []):
            yield obj


def list_subjects(s3) -> List[str]:
    subjects = []
    for cp in iter_keys(s3, f"{PREFIX}/", delimiter="/"):
        if "Prefix" not in cp:
            continue
        sub = cp["Prefix"].split("/")[1]
        if not sub.isdigit():
            continue
        subjects.append(sub)
    return subjects


def has_meg(s3, subject: str) -> bool:
    for _ in iter_keys(s3, f"{PREFIX}/{subject}/MEG/", delimiter="/"):
        return True
    return False


def sync_prefix(s3, prefix: str, out_dir: Path, max_files: Optional[int] = None) -> int:
    n = 0
    for obj in iter_keys(s3, prefix):
        key = obj.get("Key")
        if not key or key.endswith("/"):
            continue
        rel = key[len(PREFIX) + 1:]
        target = out_dir / rel
        if target.exists() and target.stat().st_size == obj.get("Size", -1):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("s3://%s/%s -> %s", BUCKET, key, target)
        s3.download_file(BUCKET, key, str(target))
        n += 1
        if max_files is not None and n >= max_files:
            break
    return n


def download_subject(s3, subject: str, out_dir: Path, rest: bool, raw: bool, anatomy: bool,
                     scans: tuple = REST_SCANS, max_files: Optional[int] = None) -> int:
    n = 0
    if rest:
        n += sync_prefix(s3, f"{PREFIX}/{subject}/MEG/Restin/rmegpreproc/", out_dir, max_files)
    if raw:
        for scan in scans:
            n += sync_prefix(s3, f"{PREFIX}/{subject}/unprocessed/MEG/{scan}/4D/", out_dir, max_files)
    if anatomy:
        n += sync_prefix(s3, f"{PREFIX}/{subject}/MEG/anatomy/", out_dir, max_files)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "hcp")
    ap.add_argument("--list-subjects", action="store_true", help="write data/hcp_meg_subjects.txt")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--subjects-file", type=Path, default=None)
    ap.add_argument("--rest", action="store_true", help="HCP-preprocessed resting sensor data (rmegpreproc)")
    ap.add_argument("--raw", action="store_true", help="unprocessed 4D resting files (contain ECG/EOG, head shape)")
    ap.add_argument("--anatomy", action="store_true", help="MEG anatomy (head/source models)")
    ap.add_argument("--sample", action="store_true", help="one subject, run 3 only, a handful of files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s3 = _client()
    subj_file = ROOT / "data" / "hcp_meg_subjects.txt"

    if args.list_subjects:
        subs = [s for s in list_subjects(s3) if has_meg(s3, s)]
        subj_file.parent.mkdir(parents=True, exist_ok=True)
        subj_file.write_text("\n".join(subs) + "\n")
        LOG.info("%d subjects with MEG -> %s", len(subs), subj_file)
        return 0

    if args.sample:
        subs = args.subjects or (subj_file.read_text().split() if subj_file.exists() else ["100307"])
        n = download_subject(s3, subs[0], args.out, rest=True, raw=True, anatomy=False, scans=("3-Restin",), max_files=5)
        LOG.info("sample: %d files for subject %s", n, subs[0])
        return 0

    subjects = list(args.subjects or [])
    if args.subjects_file:
        subjects += args.subjects_file.read_text().split()
    if not subjects:
        ap.error("give --subjects, --subjects-file, --list-subjects or --sample")
    if not (args.rest or args.raw or args.anatomy):
        ap.error("choose at least one of --rest --raw --anatomy")
    total = 0
    for sub in subjects:
        try:
            total += download_subject(s3, sub, args.out, args.rest, args.raw, args.anatomy)
        except Exception as exc:  # noqa: BLE001
            LOG.error("subject %s failed: %s", sub, exc)
    LOG.info("downloaded %d files", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
