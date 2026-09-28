#!/usr/bin/env python
"""Stage PTB-XL (open) and MIMIC-IV-ECG demographics (credentialed) for ecg_probe.

Credentials read from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Examples
--------
python scripts/download_data.py --dataset ptbxl --out data/ptbxl --sample
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --tables
python scripts/download_data.py --dataset mimiciv-hosp --out data/mimiciv-hosp --tables
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

PHYSIONET = "https://physionet.org/files"
ROOTS = {
    "ptbxl": f"{PHYSIONET}/ptb-xl/1.0.3/",
    "mimic-iv-ecg": f"{PHYSIONET}/mimic-iv-ecg/1.0/",
    "mimiciv-hosp": f"{PHYSIONET}/mimiciv/3.1/",
}
META = {
    "ptbxl": ["ptbxl_database.csv", "scp_statements.csv"],
    "mimic-iv-ecg": ["record_list.csv", "machine_measurements.csv"],
    "mimiciv-hosp": ["hosp/admissions.csv.gz", "hosp/patients.csv.gz"],
}
SAMPLE = {
    "ptbxl": [f"records500/00000/{i:05d}_hr.{ext}" for i in range(1, 21) for ext in ("hea", "dat")],
    "mimic-iv-ecg": [f"files/p1000/p10000032/s40689238/40689238.{ext}" for ext in ("hea", "dat")],
    "mimiciv-hosp": [],
}
CREDENTIALED = {"mimic-iv-ecg", "mimiciv-hosp"}


def _auth(dataset: str) -> tuple[str, str] | None:
    if dataset not in CREDENTIALED:
        return None
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    if not u or not p:
        sys.exit("Set PHYSIONET_USERNAME / PHYSIONET_PASSWORD for credentialed data.")
    return u, p


def download(url: str, dest: Path, auth) -> bool:
    if requests is None:
        sys.exit("pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
        if r.status_code != 200:
            print(f"  [fail {r.status_code}] {url}")
            return False
        with open(dest, "wb") as f:
            for c in r.iter_content(1 << 20):
                if c:
                    f.write(c)
    print(f"  [ok] {dest}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(ROOTS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tables", action="store_true", help="metadata/tabular files")
    ap.add_argument("--sample", action="store_true", help="a few sample waveforms + metadata")
    args = ap.parse_args(argv)

    auth = _auth(args.dataset)
    files = list(META[args.dataset])
    if args.sample:
        files += SAMPLE[args.dataset]
    if not (args.tables or args.sample):
        files = META[args.dataset]  # default: metadata only
    for rel in files:
        download(ROOTS[args.dataset] + rel, args.out / Path(rel).name, auth)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
