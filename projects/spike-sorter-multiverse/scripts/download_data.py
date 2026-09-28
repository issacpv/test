#!/usr/bin/env python
"""Manifest, cache and windowed raw-data extraction for the Allen Visual Coding Neuropixels raw AP band.

Usage
-----
    python scripts/download_data.py --list                       # paginated S3 listing -> manifests/raw_probes.csv
    python scripts/download_data.py --cache [--units]            # ecephys-cache CSVs
    python scripts/download_data.py --sample                     # 2-s byte-range slice of one probe (+ geometry)
    python scripts/download_data.py --session 715093703 --probe 810755797 --t-start 600 --t-end 3000

The bucket is public (no credentials). Windows are fetched with HTTP Range requests so that only the needed
bytes of the ~197 GB ``spike_band.dat`` are transferred. DANDI:000034 and IBL raw data: see data/README.md.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
from sortverse.raw_io import (FS_AP, N_CHANNELS, RAW_BUCKET, byte_range_for_window,  # noqa: E402
                              neuropixels1_geometry, raw_url)

CACHE_PREFIX = "visual-coding-neuropixels/ecephys-cache/"
RAW_PREFIX = "visual-coding-neuropixels/raw-data/"
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def s3_list(prefix: str, delimiter: Optional[str] = None, max_keys: int = 1000) -> List[Dict[str, object]]:
    """Paginated ListObjectsV2 over HTTP (continuation tokens). Returns dicts with key and size."""
    out: List[Dict[str, object]] = []
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = requests.get(RAW_BUCKET + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall(f"{S3NS}Contents"):
            out.append({"key": c.find(f"{S3NS}Key").text, "size": int(c.find(f"{S3NS}Size").text)})
        for p in root.findall(f"{S3NS}CommonPrefixes"):
            out.append({"key": p.find(f"{S3NS}Prefix").text, "size": -1})
        trunc = root.find(f"{S3NS}IsTruncated")
        if trunc is None or trunc.text != "true":
            break
        token = root.find(f"{S3NS}NextContinuationToken").text
    return out


def build_manifest() -> None:
    import pandas as pd

    objs = s3_list(RAW_PREFIX)
    rows = []
    for o in objs:
        parts = str(o["key"]).split("/")
        if len(parts) != 5 or not parts[4]:
            continue
        rows.append({"session_id": parts[2], "probe_id": parts[3], "file": parts[4], "size_bytes": o["size"]})
    df = pd.DataFrame(rows)
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "manifests" / "raw_probes.csv", index=False)
    spike = df[df.file == "spike_band.dat"]
    print(f"{len(spike)} probes with spike_band.dat across {spike.session_id.nunique()} sessions; "
          f"total {spike.size_bytes.sum() / 1e12:.2f} TB; median {spike.size_bytes.median() / 1e9:.0f} GB/probe")
    if len(spike):
        s = spike.iloc[0]
        est_h = s.size_bytes / (N_CHANNELS * 2 * FS_AP) / 3600
        print(f"example: session {s.session_id} probe {s.probe_id} -> {est_h:.2f} h of AP data")


def _stream(url: str, dest: Path, headers: Optional[dict] = None, chunk: int = 1 << 22) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers or {}, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
        tmp.rename(dest)
    print(f"  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_cache(units: bool = False) -> None:
    names = ["sessions.csv", "probes.csv", "channels.csv"] + (["units.csv"] if units else [])
    for n in names:
        _stream(f"{RAW_BUCKET}/{CACHE_PREFIX}{n}", DATA / "allen" / "cache" / n)


def extract_window(session: str, probe: str, t_start: float, t_end: float) -> Path:
    """Byte-range download of [t_start, t_end) seconds of spike_band.dat plus the small sync files."""
    out = DATA / "allen" / "raw" / session / probe
    b0, b1 = byte_range_for_window(t_start, t_end)
    dest = out / f"spike_band_t{t_start:g}-{t_end:g}s.dat"
    print(f"requesting bytes {b0}-{b1} ({(b1 - b0 + 1) / 1e9:.2f} GB) of {raw_url(session, probe)}")
    _stream(raw_url(session, probe), dest, headers={"Range": f"bytes={b0}-{b1}"})
    for small in ("channel_states.npy", "event_timestamps.npy"):
        try:
            _stream(raw_url(session, probe, small), out / small)
        except Exception as e:  # not fatal
            print(f"  could not fetch {small}: {e}")
    x, y = neuropixels1_geometry()
    np.savetxt(out / "geometry_np1.csv", np.column_stack([np.arange(N_CHANNELS), x, y]), delimiter=",",
               header="channel,x_um,y_um", comments="", fmt=["%d", "%.1f", "%.1f"])
    n_samples = (b1 - b0 + 1) // (N_CHANNELS * 2)
    print(f"  wrote {n_samples} samples x {N_CHANNELS} channels ({n_samples / FS_AP:.2f} s)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--units", action="store_true", help="with --cache: also fetch units.csv")
    ap.add_argument("--sample", action="store_true", help="2-s slice of session 715093703 probe 810755797")
    ap.add_argument("--session")
    ap.add_argument("--probe")
    ap.add_argument("--t-start", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=2.0)
    args = ap.parse_args()
    if args.list:
        build_manifest()
    if args.cache:
        download_cache(units=args.units)
    if args.sample:
        extract_window("715093703", "810755797", 0.0, 2.0)
    if args.session and args.probe:
        extract_window(args.session, args.probe, args.t_start, args.t_end)
    if not any([args.list, args.cache, args.sample, args.session]):
        ap.print_help()


if __name__ == "__main__":
    main()
