#!/usr/bin/env python
"""Download helpers for the four Allen Brain Observatory datasets on public S3 (no credentials).

Usage
-----
    python scripts/download_data.py --dataset vc2p  [--sample | --analysis]
    python scripts/download_data.py --dataset vcnpx [--sample | --session 715093703]
    python scripts/download_data.py --dataset vbo   [--sample | --n 20]
    python scripts/download_data.py --dataset vbn   [--sample | --n 5]

Listings use ListObjectsV2 with continuation tokens; downloads are streamed and resumable.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

BUCKETS = {
    "vc2p": ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com", "visual-coding-2p/"),
    "vcnpx": ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com", "visual-coding-neuropixels/ecephys-cache/"),
    "vbo": ("https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com", "visual-behavior-ophys/"),
    "vbn": ("https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com", "visual-behavior-neuropixels/"),
}
LOCAL = {"vc2p": "visual-coding-2p", "vcnpx": "visual-coding-neuropixels", "vbo": "visual-behavior-ophys",
         "vbn": "visual-behavior-neuropixels"}


def s3_list(bucket: str, prefix: str, max_items: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall(f"{S3NS}Contents"):
            out.append({"key": c.find(f"{S3NS}Key").text, "size": int(c.find(f"{S3NS}Size").text)})
            if max_items and len(out) >= max_items:
                return out
        trunc = root.find(f"{S3NS}IsTruncated")
        if trunc is None or trunc.text != "true":
            return out
        token = root.find(f"{S3NS}NextContinuationToken").text


def _stream(url: str, dest: Path, chunk: int = 1 << 22) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  exists: {dest.relative_to(ROOT)}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Range": f"bytes={tmp.stat().st_size}-"} if tmp.exists() else {}
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        if r.status_code == 416:
            tmp.rename(dest)
            return
        r.raise_for_status()
        with open(tmp, "ab" if headers else "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
    tmp.rename(dest)
    print(f"  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_keys(dataset: str, keys: List[str]) -> None:
    bucket, prefix = BUCKETS[dataset]
    for k in keys:
        rel = k[len(prefix):] if k.startswith(prefix) else k
        _stream(f"{bucket}/{k}", DATA / LOCAL[dataset] / rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(BUCKETS), required=True)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--analysis", action="store_true", help="vc2p: all ophys_experiment_analysis h5 files")
    ap.add_argument("--session", help="vcnpx: session id")
    ap.add_argument("--n", type=int, default=0, help="vbo/vbn: first N experiments/sessions")
    args = ap.parse_args()
    bucket, prefix = BUCKETS[args.dataset]
    if args.dataset == "vc2p":
        fetch_keys("vc2p", [prefix + n for n in ("cell_specimens.json", "experiment_containers.json", "manifest.json")])
        if args.sample or args.analysis:
            objs = s3_list(bucket, prefix + "ophys_experiment_analysis/", max_items=None if args.analysis else 5)
            fetch_keys("vc2p", [o["key"] for o in objs])
    elif args.dataset == "vcnpx":
        fetch_keys("vcnpx", [prefix + n for n in ("sessions.csv", "probes.csv", "channels.csv", "units.csv",
                                                  "brain_observatory_1.1_analysis_metrics.csv")])
        if args.session:
            fetch_keys("vcnpx", [f"{prefix}session_{args.session}/session_{args.session}.nwb"])
    elif args.dataset == "vbo":
        n = 3 if args.sample else args.n
        objs = s3_list(bucket, prefix + "behavior_ophys_experiments/", max_items=n or None)
        fetch_keys("vbo", [o["key"] for o in objs])
    elif args.dataset == "vbn":
        n = 1 if args.sample else args.n
        objs = [o for o in s3_list(bucket, prefix + "behavior_ecephys_sessions/") if "ecephys_session_" in o["key"]]
        fetch_keys("vbn", [o["key"] for o in objs[: n or None]])


if __name__ == "__main__":
    main()
