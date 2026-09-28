#!/usr/bin/env python3
"""List, stream and cache Neuropixels sessions from DANDI (Allen Visual Coding),
AllenSDK S3 (Visual Behavior) and IBL ONE (Brain-wide Map).

Default mode streams a single NWB over HTTP and caches only the derived tables
(units + quality metrics, spike times, stimulus/trial intervals) as .npz.

Examples
--------
    python scripts/download_data.py --dataset visual_coding --list
    python scripts/download_data.py --dataset visual_coding --sample
    python scripts/download_data.py --dataset visual_coding --session 715093703
    python scripts/download_data.py --dataset visual_coding --download      # full dandi download (large)
    python scripts/download_data.py --dataset visual_behavior --sample
    python scripts/download_data.py --dataset ibl --sample
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

DANDISETS = {"visual_coding": ["000021", "000022"]}


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# DANDI (Allen Visual Coding)
# --------------------------------------------------------------------------- #
def list_dandi_assets(dandiset_id: str, page_size: int = 200) -> List[dict]:
    """Enumerate all assets of a dandiset with pagination and resolve S3 URLs."""
    from dandi.dandiapi import DandiAPIClient  # lazy

    rows = []
    with DandiAPIClient() as client:
        ds = client.get_dandiset(dandiset_id, "draft")
        # get_assets() is a paginated generator; iterate lazily
        for asset in ds.get_assets():
            try:
                url = asset.get_content_url(follow_redirects=1, strip_query=True)
            except Exception:  # pragma: no cover - network
                url = ""
            rows.append(dict(dandiset=dandiset_id, path=asset.path, size=asset.size,
                             asset_id=asset.identifier, s3_url=url))
    return rows


def write_manifest(rows: List[dict], name: str) -> Path:
    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    out = man / name
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    _log(f"wrote {out} ({len(rows)} rows)")
    return out


def _session_assets(rows: List[dict]) -> List[dict]:
    """Keep session-level NWBs (exclude per-probe LFP files)."""
    return [r for r in rows if r["path"].endswith(".nwb") and "probe" not in r["path"].lower()]


def stream_and_cache_nwb(url: str, out: Path) -> None:
    from npx_drift.loaders import cache_session, load_session_from_nwb, open_nwb_streaming

    _log(f"streaming {url}")
    io, nwb = open_nwb_streaming(url)
    try:
        sess = load_session_from_nwb(nwb, session_id=out.stem)
        cache_session(sess, out)
    finally:
        io.close()
    _log(f"cached {out}")


def do_visual_coding(args: argparse.Namespace) -> None:
    rows_all: List[dict] = []
    for ds_id in DANDISETS["visual_coding"]:
        rows = list_dandi_assets(ds_id)
        write_manifest(rows, f"dandi_{ds_id}_assets.csv")
        rows_all += rows
    if args.list:
        return
    sessions = _session_assets(rows_all)
    if not sessions:
        sys.exit("no session NWBs found; check `dandi ls -r DANDI:000021`")
    if args.download:
        for ds_id in DANDISETS["visual_coding"]:
            subprocess.check_call(["dandi", "download", f"DANDI:{ds_id}", "-o", str(DATA / "nwb")])
        return
    chosen = sessions[:1] if args.sample else sessions
    if args.session:
        chosen = [r for r in sessions if args.session in r["path"]]
        if not chosen:
            sys.exit(f"session {args.session} not found among {len(sessions)} session NWBs")
    outdir = DATA / "cache" / "visual_coding"
    outdir.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        stem = Path(r["path"]).stem.replace("/", "_")
        stream_and_cache_nwb(r["s3_url"], outdir / f"{stem}.npz")


# --------------------------------------------------------------------------- #
# AllenSDK Visual Behavior Neuropixels
# --------------------------------------------------------------------------- #
def do_visual_behavior(args: argparse.Namespace) -> None:
    try:
        from allensdk.brain_observatory.behavior.behavior_project_cache import (
            VisualBehaviorNeuropixelsProjectCache,
        )
    except ImportError:
        sys.exit("pip install allensdk (use a dedicated environment; it pins many packages)")
    from npx_drift.loaders import cache_session, load_session_from_allensdk

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=str(DATA / "visual_behavior_neuropixels_cache"))
    table = cache.get_ecephys_session_table()
    write_manifest([dict(ecephys_session_id=i, **{k: str(v) for k, v in row.items()})
                    for i, row in table.iterrows()], "visual_behavior_sessions.csv")
    if args.list:
        return
    ids = list(table.index[:1]) if args.sample else list(table.index)
    if args.session:
        ids = [int(args.session)]
    outdir = DATA / "cache" / "visual_behavior"
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        session = cache.get_ecephys_session(ecephys_session_id=sid)
        sess = load_session_from_allensdk(session, session_id=str(sid))
        cache_session(sess, outdir / f"{sid}.npz")
        _log(f"cached session {sid}")


# --------------------------------------------------------------------------- #
# IBL Brain-wide Map
# --------------------------------------------------------------------------- #
def do_ibl(args: argparse.Namespace) -> None:
    try:
        from one.api import ONE
        from brainbox.io.one import SpikeSortingLoader
    except ImportError:
        sys.exit("pip install ONE-api ibllib")
    from npx_drift.loaders import cache_session, load_session_from_ibl

    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
    eids = one.search(project="brainwide", task_protocol="ephys")
    write_manifest([dict(eid=e) for e in eids], "ibl_bwm_sessions.csv")
    if args.list:
        return
    chosen = eids[:1] if args.sample else eids
    if args.session:
        chosen = [args.session]
    outdir = DATA / "cache" / "ibl"
    outdir.mkdir(parents=True, exist_ok=True)
    for eid in chosen:
        trials = one.load_object(eid, "trials")
        pids, _ = one.eid2pid(eid)
        for pid in pids:
            ssl = SpikeSortingLoader(pid=pid, one=one)
            spikes, clusters, channels = ssl.load_spike_sorting()
            clusters = ssl.merge_clusters(spikes, clusters, channels)
            sess = load_session_from_ibl(spikes, clusters, trials, session_id=f"{eid}_{pid}")
            cache_session(sess, outdir / f"{eid}_{pid}.npz")
            _log(f"cached {eid} / {pid}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["visual_coding", "visual_behavior", "ibl"], required=True)
    p.add_argument("--list", action="store_true", help="write manifests only")
    p.add_argument("--sample", action="store_true", help="stream/cache a single session")
    p.add_argument("--session", help="session id / eid to fetch")
    p.add_argument("--download", action="store_true", help="full local download (visual_coding only)")
    args = p.parse_args(argv)
    {"visual_coding": do_visual_coding, "visual_behavior": do_visual_behavior, "ibl": do_ibl}[args.dataset](args)


if __name__ == "__main__":
    main()
