#!/usr/bin/env python3
"""Downloaders for the open parts (OpenNeuro, reference curves) and credentialed parts
(HCP-YA via S3, HCP-A/D via NDA, OASIS-3 via NITRC XNAT) of the lifespan-normative-models project.

Credentials (environment variables):
  HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY   ConnectomeDB "Amazon S3 access" keys
  NDA_USERNAME / NDA_PASSWORD                         NIMH Data Archive
  OASIS_USER / OASIS_PASSWORD                         NITRC (aliases NITRC_USER / NITRC_PASSWORD)

Examples
--------
  python scripts/download_data.py openneuro --datasets ds004169 ds000030 --sample 2
  python scripts/download_data.py references
  python scripts/download_data.py hcp-ya --what freesurfer --sample 5
  python scripts/download_data.py hcp-lifespan --package-id 123456 --out data/hcp_lifespan
  python scripts/download_data.py oasis3 --what freesurfer --ids data/oasis3/tables/freesurfer_ids.csv --sample 3
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

OPENNEURO_HTTP = "https://s3.amazonaws.com/openneuro.org"
OPENNEURO_DEFAULT = ["ds004169", "ds000030", "ds000221", "ds003097"]
HCP_BUCKET = "hcp-openaccess"
HCP_PREFIX = "HCP_1200"
XNAT_BASE = "https://www.nitrc.org/ir"
OASIS_PROJECT = "OASIS3"
REFERENCES = {
    "Lifespan": "https://github.com/brainchart/Lifespan.git",
    "braincharts": "https://github.com/predictive-clinical-neuroscience/braincharts.git",
}


# ---------------------------------------------------------------------------
# OpenNeuro (public S3 over HTTPS; optional aws / openneuro-py for bulk)
# ---------------------------------------------------------------------------
def _http_get(url: str, dest: Path) -> bool:
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print(f"  {url}: HTTP {r.status_code}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def cmd_openneuro(a: argparse.Namespace) -> None:
    root = Path(a.out)
    for ds in a.datasets:
        d = root / ds
        print(f"[{ds}] metadata")
        ok = _http_get(f"{OPENNEURO_HTTP}/{ds}/participants.tsv", d / "participants.tsv")
        _http_get(f"{OPENNEURO_HTTP}/{ds}/dataset_description.json", d / "dataset_description.json")
        if not ok:
            continue
        with open(d / "participants.tsv") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        print(f"  {len(rows)} participants; columns: {list(rows[0].keys()) if rows else []}")
        if a.sample:
            subs = [r["participant_id"] for r in rows[: a.sample]]
            if shutil.which("aws"):
                for s in subs:
                    subprocess.run(["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{ds}/{s}",
                                    str(d / s)], check=False)
            elif shutil.which("openneuro-py"):
                subprocess.run(["openneuro-py", "download", f"--dataset={ds}", f"--target-dir={d}",
                                *[f"--include={s}/*" for s in subs]], check=False)
            else:
                print("  install awscli or openneuro-py for imaging downloads (participants.tsv fetched)")


# ---------------------------------------------------------------------------
# reference curves
# ---------------------------------------------------------------------------
def cmd_references(a: argparse.Namespace) -> None:
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    for name, url in REFERENCES.items():
        dest = root / name
        if dest.exists():
            print(f"{dest} exists")
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=False)


# ---------------------------------------------------------------------------
# HCP-YA via S3
# ---------------------------------------------------------------------------
def _hcp_client():
    try:
        import boto3
    except ImportError:
        sys.exit("pip install boto3")
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY (from ConnectomeDB 'Amazon S3 Access')")
    return boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)


def cmd_hcp_ya(a: argparse.Namespace) -> None:
    s3 = _hcp_client()
    prefix = f"{a.release}/"
    subjects = []
    token = None
    while True:
        kw = {"Bucket": HCP_BUCKET, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        subjects += [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
        token = resp.get("NextContinuationToken")
        if not token:
            break
    if a.subjects:
        subjects = [s for s in subjects if s in set(a.subjects)]
    if a.sample:
        subjects = subjects[: a.sample]
    print(f"{len(subjects)} subjects under s3://{HCP_BUCKET}/{prefix}")
    root = Path(a.out) / a.release
    for i, s in enumerate(subjects, 1):
        if a.what == "freesurfer":
            keys = [f"{prefix}{s}/T1w/{s}/stats/{f}" for f in ("aseg.stats", "lh.aparc.stats", "rh.aparc.stats")]
        else:
            keys = [f"{prefix}{s}/T1w/Diffusion/{f}" for f in ("bvals", "bvecs", "data.nii.gz", "nodif_brain_mask.nii.gz")]
        for k in keys:
            dest = root / k[len(prefix):]
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(HCP_BUCKET, k, str(dest))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(subjects)}] {k}: {exc}")
        print(f"[{i}/{len(subjects)}] {s} done")


# ---------------------------------------------------------------------------
# HCP-A / HCP-D via NDA (nda-tools)
# ---------------------------------------------------------------------------
def cmd_hcp_lifespan(a: argparse.Namespace) -> None:
    user = os.environ.get("NDA_USERNAME")
    pwd = os.environ.get("NDA_PASSWORD")
    if not shutil.which("downloadcmd"):
        print("Install nda-tools: pip install nda-tools  (provides `downloadcmd`).")
    if not (user and pwd):
        print("Set NDA_USERNAME / NDA_PASSWORD. You need an NDA account and an approved Data Use\n"
              "Certification for the HCP Lifespan collections (HCP-D 2846, HCP-A 2847), and a Data Package\n"
              "created in the NDA web interface (its numeric id is --package-id).")
    if not (a.package_id and user and pwd and shutil.which("downloadcmd")):
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cmd = ["downloadcmd", "-dp", str(a.package_id), "-u", user, "-p", pwd, "-d", a.out]
    if a.file_regex:
        cmd += ["--file-regex", a.file_regex]
    print("running:", " ".join(c if c != pwd else "****" for c in cmd))
    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# OASIS-3 via NITRC XNAT
# ---------------------------------------------------------------------------
def _xnat_session() -> requests.Session:
    user = os.environ.get("OASIS_USER") or os.environ.get("NITRC_USER") or input("NITRC username: ").strip()
    pwd = os.environ.get("OASIS_PASSWORD") or os.environ.get("NITRC_PASSWORD") or getpass.getpass("NITRC password: ")
    s = requests.Session()
    r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pwd), timeout=60)
    if r.status_code != 200:
        sys.exit(f"XNAT login failed ({r.status_code})")
    s.cookies.set("JSESSIONID", r.text.strip())
    return s


def _xnat_zip(s: requests.Session, url: str, out_dir: Path) -> None:
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


def cmd_oasis3(a: argparse.Namespace) -> None:
    with open(a.ids, newline="") as fh:
        rd = csv.DictReader(fh)
        col = next((c for c in ("freesurfer_id", "FS ID", "experiment_id", "MR ID") if c in (rd.fieldnames or [])), None)
        ids = [r[col].strip() for r in rd] if col else []
    if not ids:
        with open(a.ids) as fh:
            ids = [l.strip() for l in fh if l.strip()]
    if a.sample:
        ids = ids[: a.sample]
    s = _xnat_session()
    for i, oid in enumerate(ids, 1):
        subj = oid.split("_")[0]
        base = f"{XNAT_BASE}/data/projects/{OASIS_PROJECT}/subjects/{subj}/experiments"
        if a.what == "freesurfer":
            exp = oid.replace("_Freesurfer53_", "_MR_").replace("_Freesurfer_", "_MR_")
            url = f"{base}/{exp}/assessors/{oid}/files"
        else:  # dwi scans of an MR session
            url = f"{base}/{oid}/scans/dwi/files"
        dest = Path(a.out) / a.what / oid
        if dest.exists() and any(dest.iterdir()):
            print(f"[{i}/{len(ids)}] {oid}: exists")
            continue
        print(f"[{i}/{len(ids)}] {oid}")
        try:
            _xnat_zip(s, url, dest)
        except requests.HTTPError as exc:
            print(f"   failed: {exc}")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("openneuro")
    s.add_argument("--datasets", nargs="+", default=OPENNEURO_DEFAULT)
    s.add_argument("--out", default="data/openneuro")
    s.add_argument("--sample", type=int, default=0, help="download the first N subjects (needs aws or openneuro-py)")
    s.set_defaults(func=cmd_openneuro)

    s = sub.add_parser("references")
    s.add_argument("--out", default="data/references")
    s.set_defaults(func=cmd_references)

    s = sub.add_parser("hcp-ya")
    s.add_argument("--what", choices=["freesurfer", "diffusion"], default="freesurfer")
    s.add_argument("--release", default=HCP_PREFIX, help="HCP_1200 or HCP_Retest")
    s.add_argument("--subjects", nargs="*")
    s.add_argument("--sample", type=int, default=0)
    s.add_argument("--out", default="data/hcp_ya")
    s.set_defaults(func=cmd_hcp_ya)

    s = sub.add_parser("hcp-lifespan")
    s.add_argument("--package-id", type=int)
    s.add_argument("--out", default="data/hcp_lifespan")
    s.add_argument("--file-regex", default=None, help="e.g. '.*(stats|Diffusion).*' to limit files")
    s.set_defaults(func=cmd_hcp_lifespan)

    s = sub.add_parser("oasis3")
    s.add_argument("--what", choices=["freesurfer", "dwi"], default="freesurfer")
    s.add_argument("--ids", required=True)
    s.add_argument("--out", default="data/oasis3")
    s.add_argument("--sample", type=int, default=0)
    s.set_defaults(func=cmd_oasis3)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
