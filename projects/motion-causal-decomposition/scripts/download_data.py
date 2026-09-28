#!/usr/bin/env python3
"""Downloader for ABIDE (open, anonymous HTTPS), HCP movement files (registration + S3 keys),
and an instructions stub for ABCD (NDA application).

Examples
--------
ABIDE phenotypic file + first 20 CC200 ROI series, two denoising strategies:
    python scripts/download_data.py abide --out data/abide --sample 20

HCP movement regressors / RMS files for 3 subjects (needs boto3 and HCP S3 keys in env):
    python scripts/download_data.py hcp --out data/hcp --sample 3

ABCD instructions and table check:
    python scripts/download_data.py abcd --root data/abcd
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

ABIDE_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_PHENO = f"{ABIDE_BASE}/Phenotypic_V1_0b_preprocessed1.csv"


# ---------------------------------------------------------------------------
# ABIDE
# ---------------------------------------------------------------------------
def abide(out: Path, sample: Optional[int], pipeline: str, atlas: str, strategies: List[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pheno_path = out / "Phenotypic_V1_0b_preprocessed1.csv"
    if not pheno_path.exists():
        r = requests.get(ABIDE_PHENO, timeout=120)
        r.raise_for_status()
        pheno_path.write_bytes(r.content)
        print(f"phenotypic file -> {pheno_path}")
    rows = list(csv.DictReader(io.StringIO(pheno_path.read_text())))
    # keep subjects with usable functional data and a mean FD value
    rows = [r for r in rows if r.get("FILE_ID") and r["FILE_ID"] != "no_filename" and r.get("func_mean_fd")]
    print(f"{len(rows)} subjects with preprocessed data; columns include func_mean_fd, DX_GROUP, SITE_ID, AGE_AT_SCAN")
    if sample:
        rows = rows[:sample]
    for strat in strategies:
        d = out / pipeline / strat / f"rois_{atlas}"
        d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            fid = r["FILE_ID"]
            dest = d / f"{fid}_rois_{atlas}.1D"
            if dest.exists():
                continue
            url = f"{ABIDE_BASE}/Outputs/{pipeline}/{strat}/rois_{atlas}/{fid}_rois_{atlas}.1D"
            resp = requests.get(url, timeout=300)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                print(f"ok {strat} {fid}")
            else:
                print(f"missing {strat} {fid} (HTTP {resp.status_code})")


# ---------------------------------------------------------------------------
# HCP (S3, credentials from ConnectomeDB)
# ---------------------------------------------------------------------------
HCP_RUNS = ("rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL")
HCP_FILES = ("Movement_Regressors.txt", "Movement_RelativeRMS_mean.txt", "Movement_RelativeRMS.txt")


def hcp(out: Path, sample: Optional[int], subjects: Optional[List[str]]) -> None:
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not (key and secret):
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY (enable S3 access in ConnectomeDB).")
    try:
        import boto3
    except ImportError:
        sys.exit("pip install boto3")
    s3 = boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)
    bucket = "hcp-openaccess"
    if not subjects:
        subjects = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="HCP_1200/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                subjects.append(cp["Prefix"].split("/")[1])
            if sample and len(subjects) >= sample:
                break
        subjects = subjects[:sample] if sample else subjects
    for sub in subjects:
        for run in HCP_RUNS:
            for fname in HCP_FILES:
                keyname = f"HCP_1200/{sub}/MNINonLinear/Results/{run}/{fname}"
                dest = out / sub / run / fname
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    s3.download_file(bucket, keyname, str(dest))
                    print(f"ok {sub} {run} {fname}")
                except Exception as e:  # noqa: BLE001
                    print(f"missing {sub} {run} {fname}: {type(e).__name__}")
        eddy = f"HCP_1200/{sub}/T1w/Diffusion/eddylogs/eddy_unwarped_images.eddy_movement_rms"
        dest = out / sub / "Diffusion" / "eddy_movement_rms"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, eddy, str(dest))
                print(f"ok {sub} eddy movement rms")
            except Exception as e:  # noqa: BLE001
                print(f"missing {sub} eddy: {type(e).__name__}")


# ---------------------------------------------------------------------------
# ABCD stub
# ---------------------------------------------------------------------------
ABCD_TABLES = ["mri_y_qc_motion.csv", "mri_y_adm_info.csv", "nc_y_nihtb.csv", "mh_p_cbcl.csv", "abcd_y_lt.csv"]


def abcd(root: Path) -> None:
    print("ABCD requires an NDA Data Use Certification: https://nda.nih.gov/abcd")
    tables = root / "tables"
    for t in ABCD_TABLES:
        print(f"  [{'x' if (tables / t).exists() else ' '}] {t}")
    print("Connectomes: ABCD-BIDS / DCAN derivatives from the NDA collection; place under", root / "connectomes")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("abide")
    a.add_argument("--out", type=Path, default=Path("data/abide"))
    a.add_argument("--sample", type=int, default=None)
    a.add_argument("--pipeline", default="cpac")
    a.add_argument("--atlas", default="cc200")
    a.add_argument("--strategies", nargs="+", default=["filt_noglobal", "filt_global"])
    h = sub.add_parser("hcp")
    h.add_argument("--out", type=Path, default=Path("data/hcp"))
    h.add_argument("--sample", type=int, default=None)
    h.add_argument("--subjects", nargs="*", default=None)
    c = sub.add_parser("abcd")
    c.add_argument("--root", type=Path, default=Path("data/abcd"))
    args = p.parse_args(argv)
    if args.cmd == "abide":
        abide(args.out, args.sample, args.pipeline, args.atlas, args.strategies)
    elif args.cmd == "hcp":
        hcp(args.out, args.sample, args.subjects)
    elif args.cmd == "abcd":
        abcd(args.root)


if __name__ == "__main__":
    main()
