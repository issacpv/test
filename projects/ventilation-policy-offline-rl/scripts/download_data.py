#!/usr/bin/env python3
"""Download PhysioNet ICU databases for ventilation-policy-offline-rl.

--sample   downloads the OPEN demo datasets (MIMIC-IV Demo 2.2, eICU-CRD Demo 2.0.1) with no credentials.
--mimic / --eicu / --hirid   download the CREDENTIALED full datasets; requires environment variables
           PHYSIONET_USER and PHYSIONET_PASSWORD (never hard-code them) and an accepted DUA.

The credentialed download shells out to `wget -r -N -c -np` (PhysioNet's recommended method) so that it
is resumable; if wget is missing, a pure-Python recursive fallback is used for the demo sets only.

Examples
--------
python scripts/download_data.py --sample
PHYSIONET_USER=me PHYSIONET_PASSWORD=... python scripts/download_data.py --mimic --eicu
python scripts/download_data.py --mimic --only hosp/admissions.csv.gz icu/icustays.csv.gz
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PHYSIONET_FILES = "https://physionet.org/files"
DATASETS = {
    "mimic": "mimiciv/3.1",
    "eicu": "eicu-crd/2.0",
    "hirid": "hirid/1.1.1",
    "mimic-demo": "mimic-iv-demo/2.2",
    "eicu-demo": "eicu-crd-demo/2.0.1",
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: D401
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v and not v.startswith("?") and not v.startswith("/"):
                    self.links.append(v)


def _py_recursive(url: str, dest: Path, max_files: Optional[int] = None) -> int:
    """Minimal recursive fetch of a PhysioNet directory listing (open datasets only)."""
    n = 0
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", "ignore")
    p = _LinkParser()
    p.feed(html)
    for link in p.links:
        if link.startswith(("http", "../")):
            continue
        if link.endswith("/"):
            n += _py_recursive(url + link, dest / link, None if max_files is None else max_files - n)
        else:
            target = dest / link
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url + link, target)
            print(f"  [ok] {target.relative_to(ROOT)}")
            n += 1
        if max_files is not None and n >= max_files:
            break
    return n


def _wget(url: str, dest: Path, user: Optional[str] = None, password: Optional[str] = None,
          only: Optional[Iterable[str]] = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget") is None:
        sys.exit("wget not found; install it (apt-get install wget) or use the BigQuery route in data/README.md")
    base = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=0", "-P", str(dest)]
    if user:
        base += ["--user", user, "--password", password or ""]
    urls = [url + f for f in only] if only else [url]
    for u in urls:
        print(f"  wget {u}")
        subprocess.run(base + [u], check=False)


def download(name: str, sample: bool = False, only: Optional[Iterable[str]] = None) -> None:
    rel = DATASETS[name]
    url = f"{PHYSIONET_FILES}/{rel}/"
    dest = RAW / "physionet.org" / "files"
    credentialed = not name.endswith("demo")
    if credentialed:
        user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
        if not user or not pw:
            print(f"[skip] {name}: set PHYSIONET_USER and PHYSIONET_PASSWORD (credentialed access, see data/README.md)")
            return
        _wget(url, dest, user, pw, only)
    else:
        print(f"[open] {name} -> {dest / rel}")
        if shutil.which("wget"):
            _wget(url, dest, None, None, only)
        else:
            _py_recursive(url, dest / rel, max_files=40 if sample else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="open demo datasets only")
    p.add_argument("--mimic", action="store_true")
    p.add_argument("--eicu", action="store_true")
    p.add_argument("--hirid", action="store_true")
    p.add_argument("--only", nargs="*", help="relative file paths to fetch instead of the whole dataset")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.sample:
        download("mimic-demo", sample=True, only=args.only)
        download("eicu-demo", sample=True, only=args.only)
    if args.mimic:
        download("mimic", only=args.only)
    if args.eicu:
        download("eicu", only=args.only)
    if args.hirid:
        download("hirid", only=args.only)
    if not (args.sample or args.mimic or args.eicu or args.hirid):
        p.print_help()


if __name__ == "__main__":
    main()
