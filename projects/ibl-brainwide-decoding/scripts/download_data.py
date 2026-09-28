#!/usr/bin/env python3
"""Download open data for ibl-brainwide-decoding.

Sub-commands
------------
ibl     IBL Brain-wide Map via the public ONE/Alyx server (needs ONE-api, ibllib, brainwidemap).
allen   Allen Visual Coding Neuropixels via allensdk EcephysProjectCache (needs allensdk).
dandi   Any dandiset (000409 = IBL BWM NWB, 000021 = Allen Visual Coding) via the dandi CLI.

Examples
--------
python scripts/download_data.py ibl --sample            # session table + 3 insertions
python scripts/download_data.py ibl --labs mainenlab churchlandlab --max-insertions 20
python scripts/download_data.py allen --sample          # session table + 1 session NWB
python scripts/download_data.py dandi --dandiset 000021 --sample

All heavy libraries are imported lazily so `--help` works without them. Nothing here requires
credentials; the public IBL Alyx uses user `intbrainlab` / password `international`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_PASSWORD = "international"


# ----------------------------------------------------------------------------------------- IBL
def _one(cache_dir: Path):
    try:
        from one.api import ONE
    except ImportError as e:  # pragma: no cover
        sys.exit(f"ONE-api not installed: pip install ONE-api ibllib ({e})")
    return ONE(base_url=ONE_BASE_URL, password=ONE_PASSWORD, silent=True, cache_dir=str(cache_dir))


def cmd_ibl(args: argparse.Namespace) -> None:
    out = DATA / "ibl"
    out.mkdir(parents=True, exist_ok=True)
    one = _one(out / "one_cache")
    try:
        from brainwidemap import bwm_query
        df = bwm_query(one)
    except ImportError:
        print("brainwidemap not installed (pip install git+https://github.com/int-brain-lab/paper-brain-wide-map.git);"
              " falling back to Alyx search for BWM-tagged insertions.")
        import pandas as pd
        ins = one.alyx.rest("insertions", "list", django="session__projects__name__icontains,brainwide")
        rows = []
        for i in ins:
            s = i.get("session_info", {})
            rows.append({"pid": i["id"], "eid": i["session"], "probe_name": i["name"],
                         "subject": s.get("subject"), "lab": s.get("lab"), "date": s.get("start_time")})
        df = pd.DataFrame(rows)
    df.to_csv(out / "bwm_sessions.csv", index=False)
    print(f"BWM insertions: {len(df)}; labs: {df['lab'].nunique() if 'lab' in df else '?'}")
    if "lab" in df:
        print(df["lab"].value_counts().to_string())

    if args.labs:
        df = df[df["lab"].isin(args.labs)]
    n = 3 if args.sample else (args.max_insertions or len(df))
    df = df.head(n)

    from brainbox.io.one import SpikeSortingLoader
    from iblatlas.atlas import AllenAtlas
    ba = AllenAtlas()
    for _, row in df.iterrows():
        pid, eid = row["pid"], row["eid"]
        print(f"  downloading {pid} ({row.get('lab')}, {row.get('subject')})")
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        _ = sl.merge_clusters(spikes, clusters, channels)
        _ = one.load_object(eid, "trials")
    print(f"done: {len(df)} insertions cached under {out / 'one_cache'}")


# --------------------------------------------------------------------------------------- Allen
def cmd_allen(args: argparse.Namespace) -> None:
    out = DATA / "allen" / "ecephys_cache"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError as e:  # pragma: no cover
        sys.exit(f"allensdk not installed: pip install allensdk ({e})")
    cache = EcephysProjectCache.from_warehouse(manifest=str(out / "manifest.json"))
    sessions = cache.get_session_table()
    sessions.to_csv(DATA / "allen" / "sessions.csv")
    print(f"Allen Visual Coding sessions: {len(sessions)}")
    if args.session_type:
        sessions = sessions[sessions["session_type"] == args.session_type]
    ids = list(sessions.index[: (1 if args.sample else (args.max_sessions or len(sessions)))])
    for sid in ids:
        print(f"  downloading session {sid} ...")
        s = cache.get_session_data(sid)
        print(f"    units={len(s.units)} regions={s.units['ecephys_structure_acronym'].nunique()}")


# --------------------------------------------------------------------------------------- DANDI
def cmd_dandi(args: argparse.Namespace) -> None:
    if shutil.which("dandi") is None:
        sys.exit("dandi CLI not found: pip install dandi")
    out = DATA / "dandi"
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://dandiarchive.org/dandiset/{args.dandiset}"
    if args.sample:
        ls = subprocess.run(["dandi", "ls", "-r", url, "-f", "json"], capture_output=True, text=True, check=False)
        print(ls.stdout[:2000])
        print("Sample mode lists assets only; pick one path and run:\n"
              f"  dandi download --output-dir {out} 'https://api.dandiarchive.org/api/dandisets/{args.dandiset}/versions/draft/assets/?path=<asset_path>'")
        return
    subprocess.run(["dandi", "download", "--output-dir", str(out), "--existing", "skip", url], check=False)


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ibl")
    a.add_argument("--sample", action="store_true")
    a.add_argument("--labs", nargs="*")
    a.add_argument("--max-insertions", type=int, default=None)
    a.set_defaults(func=cmd_ibl)
    b = sub.add_parser("allen")
    b.add_argument("--sample", action="store_true")
    b.add_argument("--session-type", default=None, help="brain_observatory_1.1 or functional_connectivity")
    b.add_argument("--max-sessions", type=int, default=None)
    b.set_defaults(func=cmd_allen)
    c = sub.add_parser("dandi")
    c.add_argument("--dandiset", default="000409")
    c.add_argument("--sample", action="store_true")
    c.set_defaults(func=cmd_dandi)
    args = p.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
