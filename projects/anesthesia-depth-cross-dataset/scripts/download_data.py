#!/usr/bin/env python3
"""Download VitalDB tracks (open REST API) and the Cambridge propofol set for doa_xfer.

Examples
--------
    python scripts/download_data.py --dataset vitaldb --sample        # tables + 3 + 3 cases
    python scripts/download_data.py --dataset vitaldb                  # all eligible cases
    CAMBRIDGE_PROPOFOL_URL=https://... python scripts/download_data.py --dataset cambridge
    python scripts/download_data.py --build-manifests

The VitalDB API needs no credentials.  Nothing secret is written to disk.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

API = "https://api.vitaldb.net"
TRACKS = ["BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/BIS", "BIS/SEF", "BIS/SR", "BIS/EMG", "BIS/SQI",
          "Orchestra/PPF20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_CE",
          "Primus/EXP_SEVO", "Primus/INSP_SEVO", "Primus/EXP_DES", "Primus/MAC",
          "Solar8000/HR", "Solar8000/ART_MBP"]


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "doa_xfer/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_tables() -> None:
    d = DATA / "vitaldb"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "trks"):
        dest = d / f"{name}.csv"
        if dest.exists():
            _log(f"{dest.relative_to(ROOT)} exists, skipping")
            continue
        dest.write_bytes(_get(f"{API}/{name}"))
        _log(f"saved {dest.relative_to(ROOT)}")


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def fetch_track(caseid: str, tname: str, tid: str) -> Optional[Path]:
    dest = DATA / "vitaldb" / "tracks" / str(caseid) / (tname.replace("/", "_") + ".csv.gz")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = _get(f"{API}/{tid}")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        _log(f"case {caseid} {tname}: HTTP {exc.code}")
        return None
    with gzip.open(dest, "wb") as fh:
        fh.write(raw)
    return dest


def download_vitaldb(sample: bool, n_sample: int = 3) -> None:
    from doa_xfer.vitaldb_io import select_cases

    fetch_tables()
    cases = _read_csv(DATA / "vitaldb" / "cases.csv")
    trks = _read_csv(DATA / "vitaldb" / "trks.csv")
    sel = select_cases(cases, trks)
    todo: List[dict] = []
    for agent in ("propofol", "sevoflurane"):
        rows = [r for r in sel if r["agent"] == agent]
        todo += rows[:n_sample] if sample else rows
    _log(f"{len(todo)} cases to fetch ({'sample' if sample else 'full'})")
    by_case: Dict[str, Dict[str, str]] = {}
    for t in trks:
        by_case.setdefault(str(t["caseid"]), {})[t["tname"]] = t["tid"]
    for r in todo:
        avail = by_case.get(str(r["caseid"]), {})
        for tname in TRACKS:
            tid = avail.get(tname)
            if tid:
                fetch_track(str(r["caseid"]), tname, tid)
        _log(f"case {r['caseid']} ({r['agent']}): done")


def download_cambridge(sample: bool) -> None:
    urls = os.environ.get("CAMBRIDGE_PROPOFOL_URL", "")
    if not urls:
        sys.exit("Set CAMBRIDGE_PROPOFOL_URL to the Apollo download URL(s) of the Chennu et al. (2016) "
                 "propofol sedation EEG dataset (comma-separated for several files).")
    dest = DATA / "cambridge_propofol"
    dest.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(u.strip() for u in urls.split(",") if u.strip()):
        name = url.rstrip("/").split("/")[-1] or f"file_{i}"
        if not os.path.splitext(name)[1]:
            name = f"{name}.zip"
        out = dest / name
        if out.exists():
            _log(f"{out.name} exists, skipping")
            continue
        out.write_bytes(_get(url, timeout=600))
        _log(f"saved {out.relative_to(ROOT)}")


def build_manifests() -> None:
    from doa_xfer.vitaldb_io import select_cases

    cases_p, trks_p = DATA / "vitaldb" / "cases.csv", DATA / "vitaldb" / "trks.csv"
    if not (cases_p.exists() and trks_p.exists()):
        sys.exit("Run --dataset vitaldb --sample first to fetch cases.csv and trks.csv.")
    sel = select_cases(_read_csv(cases_p), _read_csv(trks_p))
    out_dir = DATA / "vitaldb" / "case_lists"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["caseid", "subjectid", "age", "sex", "asa", "agent", "has_eeg", "n_tracks"]
    for agent in ("propofol", "sevoflurane"):
        rows = [r for r in sel if r["agent"] == agent]
        with open(out_dir / f"{agent}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        _log(f"{agent}: {len(rows)} cases")
    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    with open(man / "cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sel)
    _log(f"wrote manifests/cases.csv: {len(sel)} rows")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["vitaldb", "cambridge", "all"])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)
    if args.dataset in ("vitaldb", "all"):
        download_vitaldb(args.sample)
    if args.dataset in ("cambridge", "all"):
        download_cambridge(args.sample)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
