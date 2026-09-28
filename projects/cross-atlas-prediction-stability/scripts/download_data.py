#!/usr/bin/env python3
"""Fetch atlases (nilearn), AOMIC fMRIPrep derivatives from OpenNeuro (anonymous S3), and HCP
dense resting-state series (registration + S3 keys).

Examples
--------
    python scripts/download_data.py atlases --out data/atlases
    python scripts/download_data.py aomic --dataset ds002785 --out data/aomic --sample 5
    python scripts/download_data.py hcp --out data/hcp --sample 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

S3_HTTP = "https://s3.amazonaws.com/openneuro.org"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ---------------------------------------------------------------------------
# atlases via nilearn
# ---------------------------------------------------------------------------
def atlases(out: Path) -> None:
    try:
        from nilearn import datasets
    except ImportError:
        sys.exit("pip install nilearn")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (100, 200, 400, 600, 800, 1000):
        for yeo in (7, 17):
            a = datasets.fetch_atlas_schaefer_2018(n_rois=n, yeo_networks=yeo, resolution_mm=2, data_dir=str(out))
            rows.append({"name": f"schaefer{n}_{yeo}", "family": "schaefer", "resolution": n, "space": "MNI", "path": a["maps"]})
    a = datasets.fetch_atlas_aal(data_dir=str(out))
    rows.append({"name": "aal", "family": "anatomical", "resolution": len(a["labels"]), "space": "MNI", "path": a["maps"]})
    a = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=str(out))
    rows.append({"name": "harvard_oxford_cort", "family": "anatomical", "resolution": len(a["labels"]) - 1, "space": "MNI", "path": a["filename"]})
    a = datasets.fetch_atlas_craddock_2012(data_dir=str(out))
    rows.append({"name": "craddock_scorr_mean", "family": "clustering", "resolution": "multi", "space": "MNI", "path": a["scorr_mean"]})
    for dim in (64, 128, 256, 512):
        a = datasets.fetch_atlas_difumo(dimension=dim, resolution_mm=2, data_dir=str(out))
        rows.append({"name": f"difumo{dim}", "family": "dictionary", "resolution": dim, "space": "MNI", "path": a["maps"]})
    a = datasets.fetch_atlas_yeo_2011(data_dir=str(out))
    rows.append({"name": "yeo7", "family": "networks", "resolution": 7, "space": "MNI", "path": a["thick_7"]})
    rows.append({"name": "yeo17", "family": "networks", "resolution": 17, "space": "MNI", "path": a["thick_17"]})
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} atlases -> {out / 'manifest.csv'}")
    print("Glasser (BALSA), Gordon and Brainnetome require registration; see data/README.md.")


# ---------------------------------------------------------------------------
# AOMIC via anonymous S3 listing
# ---------------------------------------------------------------------------
def s3_list(prefix: str, max_keys: int = 1000) -> List[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if token:
            params["continuation-token"] = token
        r = requests.get(S3_HTTP, params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [c.find(S3_NS + "Key").text for c in root.findall(S3_NS + "Contents")]
        trunc = root.find(S3_NS + "IsTruncated")
        if trunc is None or trunc.text != "true":
            break
        token = root.find(S3_NS + "NextContinuationToken").text
    return keys


def _download(key: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{S3_HTTP}/{key}", stream=True, timeout=1200) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"ok {key}")


def aomic(dataset: str, out: Path, sample: Optional[int], task: str, space: str) -> None:
    root = out / dataset
    _download(f"{dataset}/participants.tsv", root / "participants.tsv")
    keys = s3_list(f"{dataset}/derivatives/fmriprep/sub-")
    wanted = [k for k in keys if f"task-{task}" in k and (
        (f"space-{space}" in k and k.endswith("desc-preproc_bold.nii.gz")) or k.endswith("desc-confounds_regressors.tsv") or k.endswith("desc-confounds_timeseries.tsv"))]
    subjects = sorted({k.split("/")[3] for k in wanted})
    if not subjects:
        print(f"no files for task-{task} in {dataset}; available tasks:", sorted({s.split("task-")[1].split("_")[0] for s in keys if "task-" in s}))
        return
    if sample:
        subjects = subjects[:sample]
    for k in wanted:
        if k.split("/")[3] in subjects:
            _download(k, out / k)


# ---------------------------------------------------------------------------
# HCP dense series via S3
# ---------------------------------------------------------------------------
HCP_RUNS = ("rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL")


def hcp(out: Path, sample: Optional[int]) -> None:
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not (key and secret):
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY.")
    try:
        import boto3
    except ImportError:
        sys.exit("pip install boto3")
    s3 = boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)
    subjects: List[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket="hcp-openaccess", Prefix="HCP_1200/", Delimiter="/"):
        subjects += [cp["Prefix"].split("/")[1] for cp in page.get("CommonPrefixes", [])]
        if sample and len(subjects) >= sample:
            break
    for sub in subjects[:sample] if sample else subjects:
        for run in HCP_RUNS:
            fname = f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
            dest = out / sub / run / fname
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file("hcp-openaccess", f"HCP_1200/{sub}/MNINonLinear/Results/{run}/{fname}", str(dest))
                print(f"ok {sub} {run}")
            except Exception as e:  # noqa: BLE001
                print(f"missing {sub} {run}: {type(e).__name__}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("atlases")
    a.add_argument("--out", type=Path, default=Path("data/atlases"))
    b = sub.add_parser("aomic")
    b.add_argument("--dataset", default="ds002785")
    b.add_argument("--out", type=Path, default=Path("data/aomic"))
    b.add_argument("--sample", type=int, default=None)
    b.add_argument("--task", default="restingstate")
    b.add_argument("--space", default="MNI152NLin2009cAsym")
    c = sub.add_parser("hcp")
    c.add_argument("--out", type=Path, default=Path("data/hcp"))
    c.add_argument("--sample", type=int, default=None)
    args = p.parse_args(argv)
    if args.cmd == "atlases":
        atlases(args.out)
    elif args.cmd == "aomic":
        aomic(args.dataset, args.out, args.sample, args.task, args.space)
    else:
        hcp(args.out, args.sample)


if __name__ == "__main__":
    main()
