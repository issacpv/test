#!/usr/bin/env python
"""Stage CGM datasets for cgm_shift.

Only AZT1D is openly and directly downloadable (Mendeley Data). OhioT1DM,
DiaTrend, OpenAPS Data Commons and T1DEXI require registration / DUA / data
request; this script documents them and fetches AZT1D where possible.

Examples
--------
python scripts/download_data.py --dataset info
python scripts/download_data.py --dataset azt1d --out data/azt1d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

# Mendeley Data article for AZT1D (files enumerated via the Mendeley Data API).
AZT1D_ARTICLE = "gk9m674wcx"
MENDELEY_API = f"https://data.mendeley.com/public-api/datasets/{AZT1D_ARTICLE}/files?folder_id=root&version=1"

ACCESS_INFO = {
    "ohiot1dm": "DUA form: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html",
    "diatrend": "Synapse registration; see doi:10.1038/s41597-023-02469-5",
    "openaps": "Data request: https://openaps.org/outcomes/ (OpenAPS Data Commons)",
    "t1dexi": "JAEB/Vivli registration: https://public.jaeb.org/",
    "azt1d": "Open (Mendeley Data): https://data.mendeley.com/datasets/gk9m674wcx/1",
}


def _need_requests() -> None:
    if requests is None:
        sys.exit("pip install requests")


def download(url: str, dest: Path) -> bool:
    _need_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, timeout=180) as r:
        if r.status_code != 200:
            print(f"  [fail {r.status_code}] {url}")
            return False
        with open(dest, "wb") as f:
            for c in r.iter_content(1 << 20):
                if c:
                    f.write(c)
    print(f"  [ok] {dest}")
    return True


def fetch_azt1d(out: Path) -> None:
    """Enumerate AZT1D files via the Mendeley Data public API and download them."""
    _need_requests()
    try:
        r = requests.get(MENDELEY_API, timeout=60)
    except requests.RequestException as e:  # pragma: no cover - network
        print(f"  [warn] Mendeley API unreachable ({e}); download manually from "
              "https://data.mendeley.com/datasets/gk9m674wcx/1")
        return
    if r.status_code != 200:
        print(f"  [warn] Mendeley API returned {r.status_code}; download manually.")
        return
    for f in r.json():
        name = f.get("filename") or f.get("name")
        link = (f.get("content_details") or {}).get("download_url") or f.get("download_url")
        if name and link:
            download(link, out / name)
        else:
            print(f"  [avail] {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["info", "azt1d"])
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    if args.dataset == "info":
        print("Dataset access:")
        for k, v in ACCESS_INFO.items():
            print(f"  {k}: {v}")
        return 0
    fetch_azt1d(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
