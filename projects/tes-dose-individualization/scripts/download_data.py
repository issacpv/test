#!/usr/bin/env python
"""Registry search and small-sample downloader for the tES dose project.

Usage
-----
    python scripts/download_data.py --search             # build data/registry.csv from OpenNeuro
    python scripts/download_data.py --sample             # metadata + one subject anat per registry dataset
    python scripts/download_data.py --ixi                # print IXI download instructions

All data are open; no credentials are needed. The OpenNeuro GraphQL API is
paged with cursors (see ``tes_dose.openneuro``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tes_dose.openneuro import fetch_all_datasets, filter_tes_datasets, registry_frame  # noqa: E402

DATA = ROOT / "data"
REGISTRY = DATA / "registry.csv"
IXI_URLS = [
    "https://brain-development.org/ixi-dataset/",
    "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls",
]


def cmd_search(max_pages: int | None) -> None:
    nodes = fetch_all_datasets(max_pages=max_pages)
    hits = filter_tes_datasets(nodes)
    frame = registry_frame(hits)
    DATA.mkdir(exist_ok=True)
    frame.to_csv(REGISTRY, index=False)
    print(f"{len(nodes)} datasets scanned, {len(frame)} tES candidates -> {REGISTRY}")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(frame.head(30))


def cmd_sample() -> None:
    if not REGISTRY.exists():
        sys.exit("Run --search first (or create data/registry.csv by hand).")
    reg = pd.read_csv(REGISTRY)
    for _, row in reg.iterrows():
        target = DATA / "openneuro" / row["accession"]
        target.mkdir(parents=True, exist_ok=True)
        includes = ["participants.tsv", "dataset_description.json", "README", "sub-01/anat", "sub-01/ses-01/anat"]
        cmd = ["openneuro-py", "download", "--dataset", row["accession"], "--target-dir", str(target)]
        for inc in includes:
            cmd += ["--include", inc]
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            sys.exit("openneuro-py not found: pip install openneuro-py")


def cmd_ixi() -> None:
    print("IXI is ~4.5 GB and licensed CC BY-SA 3.0. Download manually:")
    for url in IXI_URLS:
        print("  ", url)
    print("Extract to data/ixi/IXI-T1/ and place IXI.xls in data/ixi/.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true", help="query OpenNeuro for tES datasets")
    ap.add_argument("--max-pages", type=int, default=None, help="limit GraphQL pages (debug)")
    ap.add_argument("--sample", action="store_true", help="download metadata + one subject anat per dataset")
    ap.add_argument("--ixi", action="store_true", help="print IXI instructions")
    args = ap.parse_args()
    if args.search:
        cmd_search(args.max_pages)
    if args.sample:
        cmd_sample()
    if args.ixi:
        cmd_ixi()
    if not (args.search or args.sample or args.ixi):
        ap.print_help()


if __name__ == "__main__":
    main()
