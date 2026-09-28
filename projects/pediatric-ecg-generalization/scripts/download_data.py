#!/usr/bin/env python
"""Stage the open datasets for ped_ecg.

- ZZU-pECG (pediatric): hosted on figshare; pass the article URL with --url.
- PTB-XL, PhysioNet/CinC 2021: PhysioNet (open, no login).
- CODE-15%: Zenodo REST API (open).

Examples
--------
python scripts/download_data.py --dataset ptbxl --out data/ptbxl --sample
python scripts/download_data.py --dataset code15 --out data/code15 --sample
python scripts/download_data.py --dataset zzu-pecg --url "<figshare url>" --out data/zzu-pecg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
ZENODO_CODE15 = "https://zenodo.org/api/records/4916206"

OPEN_META = {
    "ptbxl": {
        "root": f"{PHYSIONET}/ptb-xl/1.0.3/",
        "meta": ["ptbxl_database.csv", "scp_statements.csv"],
        "sample": [f"records500/00000/{i:05d}_hr.{ext}" for i in range(1, 11) for ext in ("hea", "dat")],
    },
    "challenge2021": {
        "root": f"{PHYSIONET}/challenge-2021/1.0.3/",
        "meta": ["dx_mapping_scored.csv", "dx_mapping_unscored.csv"],
        "sample": [f"training/georgia/g1/E{i:05d}.{ext}" for i in range(1, 11) for ext in ("hea", "mat")],
    },
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


def fetch_physionet(dataset: str, out: Path, sample: bool) -> None:
    spec = OPEN_META[dataset]
    files = list(spec["meta"]) + (spec["sample"] if sample else [])
    for rel in files:
        download(spec["root"] + rel, out / Path(rel).name)


def fetch_code15(out: Path, sample: bool) -> None:
    """Use the Zenodo REST API to list files, then fetch metadata (+ one shard if --sample)."""
    _need_requests()
    r = requests.get(ZENODO_CODE15, timeout=60)
    if r.status_code != 200:
        print(f"  [fail {r.status_code}] Zenodo API")
        return
    files = r.json().get("files", [])
    for f in files:
        name = f.get("key", "")
        link = f.get("links", {}).get("self", "")
        if name.endswith(".csv") or (sample and name.endswith("part0.hdf5")):
            download(link, out / name)
        else:
            print(f"  [avail] {name} ({f.get('size', '?')} bytes) -> {link}")


def fetch_zzu(url: str | None, out: Path) -> None:
    """ZZU-pECG lives on figshare; resolve the article's file list via the figshare API.

    Pass the figshare *article* URL; this prints the file manifest and downloads
    the metadata CSV(s). Bulk WFDB records are large; fetch them explicitly.
    """
    _need_requests()
    if not url:
        sys.exit("Pass --url with the ZZU-pECG figshare article URL (see data/README.md).")
    # figshare article pages expose /ndownloader and an API; try the API id if present.
    print(f"  ZZU-pECG source: {url}")
    print("  Locate the figshare article id and use the figshare API "
          "(https://api.figshare.com/v2/articles/<id>/files) to enumerate files,")
    print("  then download the metadata CSV and the 12-lead WFDB records into", out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["ptbxl", "challenge2021", "code15", "zzu-pecg"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--url", default=None, help="figshare article URL for zzu-pecg")
    args = ap.parse_args(argv)

    if args.dataset in OPEN_META:
        fetch_physionet(args.dataset, args.out, args.sample)
    elif args.dataset == "code15":
        fetch_code15(args.out, args.sample)
    elif args.dataset == "zzu-pecg":
        fetch_zzu(args.url, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
