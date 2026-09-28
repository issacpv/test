#!/usr/bin/env python
"""Download Allen Visual Coding / Visual Behavior Neuropixels files from the public S3 buckets.

Usage
-----
    python scripts/download_data.py --cache                        # sessions/probes/channels/units CSVs
    python scripts/download_data.py --sample                       # CSVs + one session's analysis_metrics.csv
    python scripts/download_data.py --sessions 5                   # first N brain_observatory_1.1 session NWBs
    python scripts/download_data.py --session 715093703 [--lfp]    # one session NWB (+ probe LFP NWBs)
    python scripts/download_data.py --vbn-session 1043752325       # Visual Behavior Neuropixels session (+ LFP)

No credentials are needed. Large files are streamed and resumed if a partial file exists.
"""
from __future__ import annotations

import argparse
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VC_BUCKET = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com"
VC_PREFIX = "visual-coding-neuropixels/ecephys-cache"
VBN_BUCKET = "https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com"
VBN_PREFIX = "visual-behavior-neuropixels/behavior_ecephys_sessions"
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _stream(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  exists: {dest.relative_to(ROOT)}")
        return dest
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        if r.status_code == 416:  # already complete
            tmp.rename(dest)
            return dest
        r.raise_for_status()
        with open(tmp, mode) as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
    tmp.rename(dest)
    print(f"  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e9:.2f} GB)")
    return dest


def s3_keys(bucket: str, prefix: str) -> List[str]:
    """Paginated ListObjectsV2 keys under a prefix."""
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        keys += [c.find(f"{S3NS}Key").text for c in root.findall(f"{S3NS}Contents")]
        if (root.find(f"{S3NS}IsTruncated") is None) or root.find(f"{S3NS}IsTruncated").text != "true":
            return keys
        token = root.find(f"{S3NS}NextContinuationToken").text


def download_cache() -> pd.DataFrame:
    out = DATA / "allen" / "visual-coding" / "cache"
    for name in ("sessions.csv", "probes.csv", "channels.csv", "units.csv"):
        _stream(f"{VC_BUCKET}/{VC_PREFIX}/{name}", out / name)
    return pd.read_csv(out / "sessions.csv")


def download_session(session_id: int, lfp: bool = False) -> None:
    out = DATA / "allen" / "visual-coding" / f"session_{session_id}"
    _stream(f"{VC_BUCKET}/{VC_PREFIX}/session_{session_id}/session_{session_id}.nwb", out / f"session_{session_id}.nwb")
    _stream(f"{VC_BUCKET}/{VC_PREFIX}/session_{session_id}/session_{session_id}_analysis_metrics.csv",
            out / f"session_{session_id}_analysis_metrics.csv")
    if lfp:
        for key in s3_keys(VC_BUCKET, f"{VC_PREFIX}/session_{session_id}/probe_"):
            _stream(f"{VC_BUCKET}/{key}", out / Path(key).name)


def download_vbn_session(session_id: int) -> None:
    out = DATA / "allen" / "visual-behavior" / "behavior_ecephys_sessions" / str(session_id)
    for key in s3_keys(VBN_BUCKET, f"{VBN_PREFIX}/{session_id}/"):
        _stream(f"{VBN_BUCKET}/{key}", out / Path(key).name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--sessions", type=int, default=0, help="download the first N brain_observatory_1.1 sessions")
    ap.add_argument("--session", type=int)
    ap.add_argument("--lfp", action="store_true", help="with --session: also fetch probe LFP NWBs")
    ap.add_argument("--vbn-session", type=int)
    args = ap.parse_args()
    if args.cache or args.sample or args.sessions:
        sessions = download_cache()
        print(f"{len(sessions)} sessions; types: {sessions['session_type'].value_counts().to_dict()}")
    if args.sample:
        sid = int(pd.read_csv(DATA / "allen" / "visual-coding" / "cache" / "sessions.csv")["id"].iloc[0])
        out = DATA / "allen" / "visual-coding" / f"session_{sid}"
        _stream(f"{VC_BUCKET}/{VC_PREFIX}/session_{sid}/session_{sid}_analysis_metrics.csv",
                out / f"session_{sid}_analysis_metrics.csv")
    if args.sessions:
        sessions = pd.read_csv(DATA / "allen" / "visual-coding" / "cache" / "sessions.csv")
        for sid in sessions.loc[sessions.session_type == "brain_observatory_1.1", "id"].head(args.sessions):
            download_session(int(sid), lfp=args.lfp)
    if args.session:
        download_session(args.session, lfp=args.lfp)
    if args.vbn_session:
        download_vbn_session(args.vbn_session)
    if not any([args.cache, args.sample, args.sessions, args.session, args.vbn_session]):
        ap.print_help()


if __name__ == "__main__":
    main()
