#!/usr/bin/env python
"""Download the open VitalDB tracks used by pulse_contour and stage the credentialed MIMIC sources.

VitalDB: REST API (https://api.vitaldb.net/cases, /trks, /{tid}) with --sample and track filtering.
PhysioNet (credentialed): MIMIC-III Waveform Matched Subset, MIMIC-III clinical, MIMIC-IV Waveform,
MIMIC-IV-ECHO, MIMIC-IV icu. Credentials from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Examples
--------
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --sample
python scripts/download_data.py --dataset mimic3wdb-matched --out data/mimic3wdb-matched/1.0 --sample
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

VITALDB_API = "https://api.vitaldb.net"
WANTED_TRACKS = ["SNUADC/ART", "CardioQ/SV", "CardioQ/CO", "CardioQ/FTc", "EV1000/SV", "EV1000/CO", "EV1000/SVV",
                 "Vigileo/SV", "Vigileo/CO", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/HR"]
REFERENCE_TRACKS = {"CardioQ/SV", "EV1000/SV", "Vigileo/SV"}

PHYSIONET = "https://physionet.org/files"
CREDENTIALED = {
    "mimic3wdb-matched": {"base": f"{PHYSIONET}/mimic3wdb-matched/1.0/",
                          "sample": ["RECORDS", "RECORDS-waveforms", "RECORDS-numerics"]},
    "mimiciii": {"base": f"{PHYSIONET}/mimiciii/1.4/",
                 "files": ["NOTEEVENTS.csv.gz", "D_ITEMS.csv.gz", "CHARTEVENTS.csv.gz", "ICUSTAYS.csv.gz"],
                 "sample": ["D_ITEMS.csv.gz", "ICUSTAYS.csv.gz"]},
    "mimic4wdb": {"base": f"{PHYSIONET}/mimic4wdb/0.1.0/", "sample": ["RECORDS", "RECORDS-waveforms"]},
    "mimic-iv-echo": {"base": f"{PHYSIONET}/mimic-iv-echo/0.1/", "sample": ["echo-record-list.csv", "echo-study-list.csv"]},
    "mimiciv-icu": {"base": f"{PHYSIONET}/mimiciv/3.1/icu/", "files": ["d_items.csv.gz", "icustays.csv.gz", "chartevents.csv.gz"],
                    "sample": ["d_items.csv.gz", "icustays.csv.gz"]},
}


def _need_requests() -> None:
    if requests is None:
        sys.exit("pip install requests")


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None = None) -> bool:
    _need_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
        if r.status_code in (401, 403):
            print(f"  [{r.status_code}] {url}: credentials rejected or DUA not signed")
            return False
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(dest)
    print(f"  [ok] {dest}")
    return True


def physionet_auth() -> tuple[str, str] | None:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (u, p) if u and p else None


def download_vitaldb(out: Path, tracks: bool, sample: bool, max_cases: int | None) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    stream_download(f"{VITALDB_API}/cases", out / "cases.csv")
    stream_download(f"{VITALDB_API}/trks", out / "trks.csv")
    if not tracks:
        return
    trks = pd.read_csv(out / "trks.csv")
    has_ref = trks[trks["tname"].isin(REFERENCE_TRACKS)]["caseid"].unique()
    has_art = trks[trks["tname"] == "SNUADC/ART"]["caseid"].unique()
    eligible = sorted(set(has_ref) & set(has_art))
    print(f"  {len(eligible)} cases with an SV reference and an arterial waveform")
    if sample:
        eligible = eligible[:5]
    elif max_cases:
        eligible = eligible[:max_cases]
    for cid in eligible:
        sub = trks[(trks["caseid"] == cid) & (trks["tname"].isin(WANTED_TRACKS))]
        for _, row in sub.iterrows():
            dest = out / "tracks" / str(cid) / (row["tname"].replace("/", "_") + ".csv")
            stream_download(f"{VITALDB_API}/{row['tid']}", dest)


def download_physionet(name: str, out: Path, sample: bool) -> None:
    src = CREDENTIALED[name]
    auth = physionet_auth()
    if auth is None:
        print(f"{name} is credentialed. Complete CITI training + DUA at "
              f"{src['base'].replace('/files/', '/content/')} then export PHYSIONET_USERNAME/PASSWORD.")
        return
    files = src.get("sample", []) if sample or "files" not in src else src["files"]
    if not files:
        print("  nothing to fetch without --sample; mirror with wget as described in data/README.md")
    for rel in files:
        stream_download(src["base"] + rel, out / rel, auth)


def verify(root: Path) -> None:
    v = root / "vitaldb"
    n_tracks = len(list((v / "tracks").glob("*/*.csv"))) if (v / "tracks").exists() else 0
    print(f"vitaldb: cases.csv={'yes' if (v / 'cases.csv').exists() else 'NO'} trks.csv={'yes' if (v / 'trks.csv').exists() else 'NO'} track files={n_tracks}")
    for name in CREDENTIALED:
        d = root / {"mimic3wdb-matched": "mimic3wdb-matched/1.0", "mimiciii": "mimiciii/1.4", "mimic4wdb": "mimic4wdb/0.1.0",
                    "mimic-iv-echo": "mimic-iv-echo", "mimiciv-icu": "mimiciv/3.1/icu"}[name]
        n = len(list(d.rglob("*"))) if d.exists() else 0
        print(f"{name:20s} files={n}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["vitaldb"] + list(CREDENTIALED))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--tracks", action="store_true", help="vitaldb: also download per-case tracks")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
    elif args.dataset == "vitaldb":
        download_vitaldb(args.out, args.tracks, args.sample, args.max_cases)
    elif args.dataset:
        download_physionet(args.dataset, args.out, args.sample)
    else:
        ap.error("--dataset or --verify required")


if __name__ == "__main__":
    main()
