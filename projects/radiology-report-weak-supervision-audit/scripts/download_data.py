#!/usr/bin/env python
"""Download helpers for radiology-report-weak-supervision-audit.

* ``physionet`` – credentialed PhysioNet projects via HTTP basic auth (``PHYSIONET_USER``/``PHYSIONET_PASS``).
                  Walks the project's file index, filters by regex, and (``--sample``) fetches only the
                  small label/metadata tables. Requires an approved DUA for each project.
* ``nih``        – prints instructions for the open NIH ChestX-ray14 files and adjudicated labels.

Examples
--------
python scripts/download_data.py --source physionet --project mimic-cxr-jpg --version 2.1.0 --sample
python scripts/download_data.py --source physionet --project chest-imagenome --version 1.0.0 --include gold_dataset
python scripts/download_data.py --source physionet --project mimic-cxr-jpg --version 2.1.0 --study-list outputs/studies.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SAMPLE_PATTERNS = {
    "mimic-cxr-jpg": [r"mimic-cxr-2\.0\.0-(chexpert|negbio|metadata|split)\.csv\.gz$", r"IMAGE_FILENAMES$"],
    "mimic-cxr": [r"cxr-record-list\.csv\.gz$", r"cxr-study-list\.csv\.gz$"],
    "chest-imagenome": [r"gold_dataset/", r"silver_dataset/scene_tabular/.*attribute.*\.csv$"],
    "reflacx-xray-localization": [r"metadata_phase_\d\.csv$", r"main_data/.*/anomaly_location_ellipses\.csv$"],
    "ms-cxr": [r"\.csv$"],
    "vindr-cxr": [r"annotations/"],
    "mimiciv": [r"hosp/(patients|admissions|transfers)\.csv\.gz$", r"icu/icustays\.csv\.gz$"],
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v and not v.startswith(("?", "/", "#", "http")):
                    self.links.append(v)


def _session() -> "requests.Session":
    if requests is None:
        raise SystemExit("pip install requests")
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not (user and pw):
        raise SystemExit("Set PHYSIONET_USER and PHYSIONET_PASS (credentialed PhysioNet account with signed DUA).")
    s = requests.Session()
    s.auth = (user, pw)
    return s


def walk(sess: "requests.Session", url: str, rel: str = "") -> Iterator[str]:
    """Yield relative file paths below ``url`` by parsing PhysioNet's directory listings."""
    r = sess.get(url, timeout=120)
    if r.status_code in (401, 403):
        raise SystemExit(f"{r.status_code} for {url}: check credentials / DUA approval for this project")
    r.raise_for_status()
    p = _LinkParser()
    p.feed(r.text)
    for link in p.links:
        if link.endswith("/"):
            yield from walk(sess, url + link, rel + link)
        else:
            yield rel + link


def download_physionet(project: str, version: str, include: Iterable[str], sample: bool,
                       study_list: Path | None, max_files: int | None) -> None:
    sess = _session()
    base = f"{PHYSIONET}/{project}/{version}/"
    pats = [re.compile(p) for p in include]
    if sample:
        pats += [re.compile(p) for p in SAMPLE_PATTERNS.get(project, [r"\.csv(\.gz)?$", r"\.txt$"])]
    studies: set[str] | None = None
    if study_list:
        studies = {line.strip() for line in study_list.read_text().splitlines() if line.strip()}
    out = DATA_DIR / "physionet" / project / version
    n = 0
    for rel in walk(sess, base):
        if pats and not any(p.search(rel) for p in pats):
            if studies is None:
                continue
        if studies is not None:
            m = re.search(r"/(s\d+)(/|\.txt$)", rel)
            if not m or m.group(1) not in studies:
                if not (pats and any(p.search(rel) for p in pats)):
                    continue
        dest = out / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{project}/{version}] {rel}")
        with sess.get(base + rel, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        n += 1
        if max_files and n >= max_files:
            break
    print(f"downloaded {n} files into {out}")


def nih_instructions() -> None:
    print("NIH ChestX-ray14 (open): https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
          "  - Data_Entry_2017.csv (NLP labels), images_001..012.tar.gz, train_val_list.txt / test_list.txt\n"
          "  - radiologist-adjudicated labels: see Majkowska et al., 2020, Radiology (supplement / NIH Box)\n"
          "Place files under data/nih/. Box links are not stable enough to script; download in a browser or\n"
          "with the URLs listed in the Box folder's batch_download_zips.py.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["physionet", "nih"], required=True)
    p.add_argument("--project", default="mimic-cxr-jpg")
    p.add_argument("--version", default="2.1.0")
    p.add_argument("--include", nargs="*", default=[], help="regex filters on relative paths")
    p.add_argument("--sample", action="store_true", help="only label/metadata tables")
    p.add_argument("--study-list", type=Path, default=None, help="text file of study ids (s5041...) to fetch")
    p.add_argument("--max-files", type=int, default=None)
    a = p.parse_args(argv)
    if a.source == "physionet":
        download_physionet(a.project, a.version, a.include, a.sample, a.study_list, a.max_files)
    else:
        nih_instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
