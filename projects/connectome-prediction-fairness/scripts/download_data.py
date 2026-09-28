#!/usr/bin/env python3
"""Download the open parts of the data used by ``cpm_fair``.

Sources
-------
1. HCP S1200 (open access after free ConnectomeDB registration).
   Imaging lives in the AWS bucket ``s3://hcp-openaccess`` and requires the
   AWS credentials issued by ConnectomeDB (Account -> "Amazon S3 access").
   Set ``HCP_AWS_ACCESS_KEY_ID`` / ``HCP_AWS_SECRET_ACCESS_KEY`` in the
   environment. The behavioural CSVs (unrestricted & restricted) must be
   downloaded manually from ConnectomeDB and placed in ``data/hcp/``.
2. AOMIC-PIOP1 / PIOP2 (OpenNeuro ds002785 / ds002790): fully open; used for
   the sex / education (SES proxy) replication with fMRIPrep confounds.
   Downloaded from the public ``s3://openneuro.org`` bucket (no credentials).
3. ABCD (NDA, optional): credentialed; only instructions are printed.

Examples
--------
    python scripts/download_data.py --sample               # tiny smoke-test download
    python scripts/download_data.py --hcp --subjects data/hcp/subjects.txt
    python scripts/download_data.py --aomic ds002785 --n-subjects 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HCP_BUCKET = "hcp-openaccess"
HCP_PREFIX = "HCP_1200"
HCP_REST_RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
# MSMAll-registered, ICA-FIX cleaned dense time series (~ 1 GB / run) - the
# input to parcellation with wb_command. Set --hcp-file to change.
HCP_REST_FILE = "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
HCP_MOTION_FILE = "Movement_RelativeRMS_mean.txt"

OPENNEURO_BUCKET = "openneuro.org"


def _boto3(anonymous: bool):
    try:
        import boto3  # type: ignore
        from botocore import UNSIGNED  # type: ignore
        from botocore.config import Config  # type: ignore
    except ImportError:
        sys.exit("boto3 is required: pip install boto3")
    if anonymous:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        sys.exit(
            "HCP credentials missing. Register at https://db.humanconnectome.org, accept the "
            "Open Access Data Use Terms, create AWS keys under your profile and export "
            "HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY."
        )
    return boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)


def s3_list(client, bucket: str, prefix: str, max_keys: Optional[int] = None) -> List[str]:
    """List keys under a prefix with pagination."""
    keys: List[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if max_keys and len(keys) >= max_keys:
                return keys
    return keys


def s3_get(client, bucket: str, key: str, dest: Path, overwrite: bool = False) -> bool:
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return True


# --------------------------------------------------------------------------- #
# HCP
# --------------------------------------------------------------------------- #
def download_hcp(subjects: Iterable[str], runs: List[str], rest_file: str, out: Path, dry_run: bool = False) -> None:
    client = _boto3(anonymous=False)
    for sid in subjects:
        sid = str(sid).strip()
        if not sid:
            continue
        for run in runs:
            base = f"{HCP_PREFIX}/{sid}/MNINonLinear/Results/{run}"
            for fname in (rest_file.format(run=run), HCP_MOTION_FILE):
                key = f"{base}/{fname}"
                dest = out / sid / run / fname
                if dry_run:
                    print(f"[dry-run] s3://{HCP_BUCKET}/{key} -> {dest}")
                    continue
                try:
                    if s3_get(client, HCP_BUCKET, key, dest):
                        print(f"downloaded {key}")
                    else:
                        print(f"exists     {key}")
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED     {key}: {exc}")


# --------------------------------------------------------------------------- #
# OpenNeuro (AOMIC)
# --------------------------------------------------------------------------- #
def download_aomic(dataset: str, n_subjects: Optional[int], out: Path, include_bold: bool = False) -> None:
    """Fetch participants.tsv, fMRIPrep confounds and (optionally) preprocessed rest BOLD."""
    client = _boto3(anonymous=True)
    prefix = f"{dataset}/"
    for fname in ("participants.tsv", "participants.json", "dataset_description.json", "README"):
        try:
            s3_get(client, OPENNEURO_BUCKET, prefix + fname, out / dataset / fname)
            print(f"downloaded {prefix + fname}")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {fname}: {exc}")
    keys = s3_list(client, OPENNEURO_BUCKET, prefix + "derivatives/fmriprep/")
    if not keys:
        print("No fMRIPrep derivatives found under derivatives/fmriprep/ - check the dataset on openneuro.org")
        return
    subjects = sorted({k.split("/")[3] for k in keys if k.split("/")[3].startswith("sub-")})
    if n_subjects:
        subjects = subjects[:n_subjects]
    wanted = []
    for k in keys:
        parts = k.split("/")
        if len(parts) < 4 or parts[3] not in subjects:
            continue
        if "task-restingstate" not in k:
            continue
        if k.endswith("desc-confounds_regressors.tsv") or k.endswith("desc-confounds_timeseries.tsv"):
            wanted.append(k)
        elif include_bold and k.endswith("space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"):
            wanted.append(k)
    for k in wanted:
        dest = out / k
        try:
            if s3_get(client, OPENNEURO_BUCKET, k, dest):
                print(f"downloaded {k}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {k}: {exc}")
    print(f"{len(wanted)} files for {len(subjects)} subjects")


def print_abcd_instructions() -> None:
    print(
        "\nABCD (optional cross-dataset replication):\n"
        "  1. Obtain an NDA account and an approved ABCD Data Use Certification (DUC).\n"
        "  2. Use the NDA download manager / nda-tools (`downloadcmd`) to fetch the ABCD\n"
        "     rsfMRI derivatives (ABCD-BIDS, DCAN/fMRIPrep) and the demographic tables\n"
        "     (abcd_p_demo, race/ethnicity, household income, parental education).\n"
        "  3. Never copy ABCD data outside the approved environment; do not commit any file.\n"
        "  Environment variables expected by later scripts: NDA_USERNAME, NDA_PASSWORD.\n"
    )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="tiny smoke test: 3 HCP subjects (1 run) + 2 AOMIC subjects")
    ap.add_argument("--hcp", action="store_true", help="download HCP rest runs for --subjects")
    ap.add_argument("--subjects", type=Path, help="text file with one HCP subject ID per line")
    ap.add_argument("--runs", nargs="+", default=HCP_REST_RUNS)
    ap.add_argument("--hcp-file", default=HCP_REST_FILE, help="per-run file name template ({run})")
    ap.add_argument("--aomic", default=None, help="OpenNeuro dataset id, e.g. ds002785 or ds002790")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--include-bold", action="store_true", help="also fetch preprocessed rest BOLD (large)")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.sample:
        print("Sample mode: AOMIC ds002785 confounds for 2 subjects (anonymous S3)")
        download_aomic("ds002785", 2, args.out)
        print("Sample mode: HCP listing for 3 subjects (needs credentials; dry-run shown)")
        download_hcp(["100307", "100408", "101107"], ["rfMRI_REST1_LR"], args.hcp_file, args.out / "hcp", dry_run=True)
        print_abcd_instructions()
        return
    if args.hcp:
        if not args.subjects:
            sys.exit("--subjects file required with --hcp")
        subjects = [s.strip() for s in args.subjects.read_text().splitlines() if s.strip()]
        download_hcp(subjects, args.runs, args.hcp_file, args.out / "hcp", dry_run=args.dry_run)
    if args.aomic:
        download_aomic(args.aomic, args.n_subjects, args.out, include_bold=args.include_bold)
    if not (args.hcp or args.aomic):
        ap.print_help()
        print_abcd_instructions()


if __name__ == "__main__":
    main()
