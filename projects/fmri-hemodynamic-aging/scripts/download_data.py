#!/usr/bin/env python
"""Download helpers for fmri-hemodynamic-aging.

Three sources are handled:

* ``openneuro``  – open data through the OpenNeuro GraphQL API (no credentials).
                   ``--sample`` fetches a couple of subjects' rest + T1w files only.
* ``hcp-aging``  – wraps NDA ``downloadcmd``; requires ``NDA_USERNAME``/``NDA_PASSWORD``
                   and an NDA package id created in the NDA web interface.
* ``oasis3``     – XNAT Central REST download of selected MR sessions; requires
                   ``NITRC_USER``/``NITRC_PASS`` and an approved OASIS-3 DUA.

Examples
--------
python scripts/download_data.py --source openneuro --dataset ds000030 --sample
python scripts/download_data.py --source hcp-aging --package-id 1234567
python scripts/download_data.py --source oasis3 --session-list sessions.csv --scan-types bold,T1w,FLAIR
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover - requests is in requirements.txt
    requests = None  # type: ignore

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
XNAT_BASE = "https://central.xnat.org"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ----------------------------------------------------------------------------
# OpenNeuro (open; GraphQL API)
# ----------------------------------------------------------------------------
def _gql(query: str, variables: dict) -> dict:
    if requests is None:
        raise SystemExit("pip install requests")
    resp = requests.post(OPENNEURO_GRAPHQL, json={"query": query, "variables": variables}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


def latest_snapshot_tag(dataset: str) -> str:
    data = _gql(
        "query($id: ID!) { dataset(id: $id) { latestSnapshot { tag } } }",
        {"id": dataset},
    )
    return data["dataset"]["latestSnapshot"]["tag"]


def iter_snapshot_files(dataset: str, tag: str, tree: str | None = None, prefix: str = "") -> Iterator[dict]:
    """Recursively yield ``{filename, size, urls}`` for every file in a snapshot."""
    query = (
        "query($id: ID!, $tag: String!, $tree: String) {"
        " snapshot(datasetId: $id, tag: $tag) { files(tree: $tree) { id filename size directory urls } } }"
    )
    data = _gql(query, {"id": dataset, "tag": tag, "tree": tree})
    for f in data["snapshot"]["files"]:
        path = f"{prefix}{f['filename']}"
        if f["directory"]:
            yield from iter_snapshot_files(dataset, tag, tree=f["id"], prefix=path + "/")
        else:
            yield {"filename": path, "size": f["size"], "urls": f["urls"]}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def download_openneuro(dataset: str, include: Iterable[str], sample: bool, max_files: int | None) -> None:
    tag = latest_snapshot_tag(dataset)
    patterns = [re.compile(p) for p in include]
    out = DATA_DIR / "openneuro" / dataset
    n = 0
    seen_subjects: set[str] = set()
    for f in iter_snapshot_files(dataset, tag):
        name = f["filename"]
        if patterns and not any(p.search(name) for p in patterns):
            continue
        if sample:
            m = re.match(r"(sub-[^/]+)/", name)
            if m:
                seen_subjects.add(m.group(1))
                if len(seen_subjects) > 2:
                    break
            if not (name.endswith(("_bold.nii.gz", "_T1w.nii.gz", ".json", ".tsv")) or "/" not in name):
                continue
        dest = out / name
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print(f"[{dataset}@{tag}] {name} ({f['size'] / 1e6:.1f} MB)")
        download_file(f["urls"][0], dest)
        n += 1
        if max_files and n >= max_files:
            break
    print(f"downloaded {n} files to {out}")


# ----------------------------------------------------------------------------
# HCP-Aging via NDA downloadcmd (credentialed)
# ----------------------------------------------------------------------------
def download_hcp_aging(package_id: str, workers: int = 8) -> None:
    user, pw = os.environ.get("NDA_USERNAME"), os.environ.get("NDA_PASSWORD")
    if not (user and pw):
        raise SystemExit("Set NDA_USERNAME and NDA_PASSWORD (NDA account with an approved HCP-Aging DUC).")
    if shutil.which("downloadcmd") is None:
        raise SystemExit("pip install nda-tools  (provides the downloadcmd CLI)")
    out = DATA_DIR / "hcp_aging"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["downloadcmd", "-dp", package_id, "-d", str(out), "-u", user, "-p", pw, "-wt", str(workers)]
    print("running:", " ".join(c if c != pw else "****" for c in cmd))
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------
# OASIS-3 via XNAT Central REST (credentialed)
# ----------------------------------------------------------------------------
def download_oasis3(session_list: Path, scan_types: list[str], project: str = "OASIS3") -> None:
    user, pw = os.environ.get("NITRC_USER"), os.environ.get("NITRC_PASS")
    if not (user and pw):
        raise SystemExit("Set NITRC_USER and NITRC_PASS (NITRC account with an approved OASIS-3 DUA).")
    if requests is None:
        raise SystemExit("pip install requests")
    sess = requests.Session()
    sess.auth = (user, pw)
    out = DATA_DIR / "oasis3"
    with open(session_list) as fh:
        sessions = [row[0].strip() for row in csv.reader(fh) if row and not row[0].startswith("#")]
    for exp in sessions:  # e.g. OAS30001_MR_d0129
        subject = exp.split("_")[0]
        r = sess.get(f"{XNAT_BASE}/data/projects/{project}/subjects/{subject}/experiments/{exp}/scans",
                     params={"format": "json"}, timeout=120)
        r.raise_for_status()
        scans = r.json()["ResultSet"]["Result"]
        for scan in scans:
            stype = scan.get("type", "")
            if not any(t.lower() in stype.lower() for t in scan_types):
                continue
            url = f"{XNAT_BASE}/data/projects/{project}/subjects/{subject}/experiments/{exp}/scans/{scan['ID']}/files"
            files = sess.get(url, params={"format": "json"}, timeout=120).json()["ResultSet"]["Result"]
            for f in files:
                if not f["Name"].endswith((".nii.gz", ".json", ".bvec", ".bval")):
                    continue
                dest = out / exp / stype / f["Name"]
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[{exp}] {stype}/{f['Name']}")
                with sess.get(f"{XNAT_BASE}{f['URI']}", stream=True, timeout=600) as rr:
                    rr.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in rr.iter_content(1 << 20):
                            fh.write(chunk)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["openneuro", "hcp-aging", "oasis3"], required=True)
    p.add_argument("--dataset", default="ds000030", help="OpenNeuro accession")
    p.add_argument("--include", nargs="*", default=[], help="regex filters on file paths (openneuro)")
    p.add_argument("--sample", action="store_true", help="openneuro: 2 subjects, rest + T1w only")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--package-id", help="NDA package id (hcp-aging)")
    p.add_argument("--session-list", type=Path, help="CSV of OASIS-3 MR session labels (oasis3)")
    p.add_argument("--scan-types", default="bold,T1w,FLAIR", help="comma-separated XNAT scan-type substrings")
    a = p.parse_args(argv)

    if a.source == "openneuro":
        download_openneuro(a.dataset, a.include, a.sample, a.max_files)
    elif a.source == "hcp-aging":
        if not a.package_id:
            raise SystemExit("--package-id is required for hcp-aging")
        download_hcp_aging(a.package_id)
    else:
        if not a.session_list:
            raise SystemExit("--session-list is required for oasis3")
        download_oasis3(a.session_list, a.scan_types.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
