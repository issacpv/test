#!/usr/bin/env python3
"""Downloader for ``conn_h2``.

Open parts
----------
* HCP S1200 imaging from ``s3://hcp-openaccess`` (ConnectomeDB-issued AWS
  keys in ``HCP_AWS_ACCESS_KEY_ID`` / ``HCP_AWS_SECRET_ACCESS_KEY``):
  rest fMRI dense time series (for gradients / dynamic FC) and the
  preprocessed diffusion data (for structural connectomes).
* Allen Human Brain Atlas microarray via ``abagen`` (public, ~4 GB, fetched
  automatically on first call).
* Parcellation files (Schaefer / Glasser) from their public GitHub releases.

Credentialed parts (instructions only)
--------------------------------------
* HCP *restricted* CSV (Family_ID, ZygosityGT, Mother_ID, Father_ID,
  Age_in_Yrs): download manually from ConnectomeDB after approval and place
  at ``data/hcp/restricted.csv`` (path configurable via ``HCP_RESTRICTED_CSV``).
* ABCD (NDA) for replication.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --hcp-rest --subjects data/hcp/twins.txt
    python scripts/download_data.py --hcp-diffusion --subjects data/hcp/twins.txt
    python scripts/download_data.py --abagen --atlas data/parcellations/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HCP_BUCKET = "hcp-openaccess"
HCP_PREFIX = "HCP_1200"
REST_RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
REST_FILE = "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
DIFFUSION_FILES = ["bvals", "bvecs", "data.nii.gz", "nodif_brain_mask.nii.gz", "grad_dev.nii.gz"]

SCHAEFER_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations"
)
PARCELLATION_FILES = {
    "schaefer400_fslr32k_dlabel": f"{SCHAEFER_BASE}/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii",
    "schaefer400_mni2mm": f"{SCHAEFER_BASE}/MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    "schaefer400_info": f"{SCHAEFER_BASE}/MNI/Schaefer2018_400Parcels_7Networks_order.txt",
}


def _hcp_client():
    try:
        import boto3  # type: ignore
    except ImportError:
        sys.exit("pip install boto3")
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY (ConnectomeDB -> Amazon S3 access).")
    return boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)


def _get(client, key: str, dest: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] s3://{HCP_BUCKET}/{key} -> {dest}")
        return
    if dest.exists():
        print(f"exists     {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(HCP_BUCKET, key, str(dest))
        print(f"downloaded {key}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED     {key}: {exc}")


def download_hcp_rest(subjects: Iterable[str], out: Path, runs: List[str] = REST_RUNS, dry_run: bool = False) -> None:
    client = None if dry_run else _hcp_client()
    for sid in subjects:
        for run in runs:
            key = f"{HCP_PREFIX}/{sid}/MNINonLinear/Results/{run}/{REST_FILE.format(run=run)}"
            _get(client, key, out / sid / run / REST_FILE.format(run=run), dry_run)
            mkey = f"{HCP_PREFIX}/{sid}/MNINonLinear/Results/{run}/Movement_RelativeRMS_mean.txt"
            _get(client, mkey, out / sid / run / "Movement_RelativeRMS_mean.txt", dry_run)


def download_hcp_diffusion(subjects: Iterable[str], out: Path, dry_run: bool = False) -> None:
    """Preprocessed diffusion (~ 4.5 GB / subject) + T1w for tractography."""
    client = None if dry_run else _hcp_client()
    for sid in subjects:
        for f in DIFFUSION_FILES:
            _get(client, f"{HCP_PREFIX}/{sid}/T1w/Diffusion/{f}", out / sid / "T1w" / "Diffusion" / f, dry_run)
        for f in ("T1w_acpc_dc_restore_brain.nii.gz", "aparc+aseg.nii.gz"):
            _get(client, f"{HCP_PREFIX}/{sid}/T1w/{f}", out / sid / "T1w" / f, dry_run)


def download_parcellations(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, url in PARCELLATION_FILES.items():
        dest = out / url.rsplit("/", 1)[-1]
        if dest.exists():
            print(f"exists     {dest}")
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"downloaded {dest.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED     {url}: {exc}")


def fetch_abagen(atlas: Path, out: Path) -> None:
    try:
        import abagen  # type: ignore
    except ImportError:
        sys.exit("pip install abagen")
    out.mkdir(parents=True, exist_ok=True)
    print("Fetching AHBA microarray (first run downloads ~4 GB to ~/abagen-data)...")
    expr = abagen.get_expression_data(str(atlas), lr_mirror="bidirectional", missing="interpolate", norm_matched=True)
    dest = out / f"expression_{atlas.stem.split('.')[0]}.csv"
    expr.to_csv(dest)
    print(f"saved {dest} with shape {expr.shape}")


def print_restricted_instructions() -> None:
    print(
        "\nHCP restricted data (required for twin pairs):\n"
        "  1. https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage\n"
        "  2. After approval: ConnectomeDB -> WU-Minn HCP Data 1200 Subjects -> 'Restricted' CSV export.\n"
        "  3. Save as data/hcp/restricted.csv (or set HCP_RESTRICTED_CSV). Columns used: Subject, Family_ID,\n"
        "     Mother_ID, Father_ID, ZygosityGT, ZygositySR, Age_in_Yrs.\n"
        "  4. Never commit or share this file.\n"
        "ABCD (replication): NDA DUC required; use nda-tools `downloadcmd` for ABCD-BIDS rsfMRI derivatives\n"
        "  and the family/twin tables (acspsw03 / genetic relatedness)."
    )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="parcellation files + dry-run listing for 2 subjects")
    ap.add_argument("--hcp-rest", action="store_true")
    ap.add_argument("--hcp-diffusion", action="store_true")
    ap.add_argument("--subjects", type=Path, help="text file with one HCP subject ID per line")
    ap.add_argument("--parcellations", action="store_true")
    ap.add_argument("--abagen", action="store_true")
    ap.add_argument("--atlas", type=Path, default=DATA / "parcellations" / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.sample:
        download_parcellations(args.out / "parcellations")
        download_hcp_rest(["100307", "100408"], args.out / "hcp", runs=["rfMRI_REST1_LR"], dry_run=True)
        print_restricted_instructions()
        return
    subjects: List[str] = []
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.read_text().splitlines() if s.strip()]
    if args.parcellations:
        download_parcellations(args.out / "parcellations")
    if args.hcp_rest:
        if not subjects:
            sys.exit("--subjects required")
        download_hcp_rest(subjects, args.out / "hcp", dry_run=args.dry_run)
    if args.hcp_diffusion:
        if not subjects:
            sys.exit("--subjects required")
        download_hcp_diffusion(subjects, args.out / "hcp", dry_run=args.dry_run)
    if args.abagen:
        fetch_abagen(args.atlas, args.out / "ahba")
    if not any((args.parcellations, args.hcp_rest, args.hcp_diffusion, args.abagen)):
        ap.print_help()
        print_restricted_instructions()


if __name__ == "__main__":
    main()
