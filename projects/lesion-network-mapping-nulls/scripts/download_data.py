#!/usr/bin/env python
"""Download helpers for lesion-network-mapping-nulls.

* ``openneuro`` – ARC (ds004884) / SOOP (ds004889) via the OpenNeuro GraphQL API (open).
                  ``--sample`` fetches two subjects' anat + lesion masks and participants.tsv.
* ``atlas``      – ATLAS v2.0: prints the DUA-gated steps and checks ATLAS_PASSWORD is set.
* ``hcp``        – HCP-YA rfMRI for the functional connectome via the hcp-openaccess S3 bucket
                  (needs AWS credentials issued by ConnectomeDB and the awscli).

Examples
--------
python scripts/download_data.py --source openneuro --dataset ds004884 --sample
python scripts/download_data.py --source openneuro --dataset ds004884 --include participants lesion T1w
python scripts/download_data.py --source hcp --subjects 100307 100408 --runs REST1_LR
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_SUFFIXES = ("_T1w.nii.gz", "_T2w.nii.gz", "mask.nii.gz", "participants.tsv", "participants.json",
                   "dataset_description.json", ".tsv", ".json")


def _gql(query: str, variables: dict) -> dict:
    if requests is None:
        raise SystemExit("pip install requests")
    r = requests.post(OPENNEURO_GRAPHQL, json={"query": query, "variables": variables}, timeout=60)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def latest_snapshot_tag(dataset: str) -> str:
    d = _gql("query($id: ID!) { dataset(id: $id) { latestSnapshot { tag } } }", {"id": dataset})
    return d["dataset"]["latestSnapshot"]["tag"]


def iter_files(dataset: str, tag: str, tree: str | None = None, prefix: str = "") -> Iterator[dict]:
    q = ("query($id: ID!, $tag: String!, $tree: String) { snapshot(datasetId: $id, tag: $tag) {"
         " files(tree: $tree) { id filename size directory urls } } }")
    for f in _gql(q, {"id": dataset, "tag": tag, "tree": tree})["snapshot"]["files"]:
        path = prefix + f["filename"]
        if f["directory"]:
            yield from iter_files(dataset, tag, f["id"], path + "/")
        else:
            yield {"filename": path, "size": f["size"], "urls": f["urls"]}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def download_openneuro(dataset: str, include: Iterable[str], sample: bool, max_files: int | None) -> None:
    tag = latest_snapshot_tag(dataset)
    pats = [re.compile(p) for p in include]
    out = DATA_DIR / "openneuro" / dataset
    subjects: set[str] = set()
    n = 0
    for f in iter_files(dataset, tag):
        name = f["filename"]
        if pats and not any(p.search(name) for p in pats):
            continue
        if sample:
            m = re.match(r"(sub-[^/]+)/", name)
            if m:
                subjects.add(m.group(1))
                if len(subjects) > 2:
                    break
            if not name.endswith(SAMPLE_SUFFIXES):
                continue
        dest = out / name
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print(f"[{dataset}@{tag}] {name} ({f['size'] / 1e6:.1f} MB)")
        fetch(f["urls"][0], dest)
        n += 1
        if max_files and n >= max_files:
            break
    print(f"downloaded {n} files into {out}")


def atlas_instructions() -> None:
    pw = os.environ.get("ATLAS_PASSWORD")
    print("ATLAS v2.0 is DUA-gated (INDI). Steps:\n"
          "  1. Sign the DUA at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html\n"
          "  2. Use the e-mailed S3 path and password; decrypt with openssl as in data/README.md\n"
          f"  3. ATLAS_PASSWORD is {'set' if pw else 'NOT set'} in this environment.")
    if shutil.which("aws") is None:
        print("  note: awscli not found (pip install awscli)")


def download_hcp(subjects: list[str], runs: list[str]) -> None:
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        raise SystemExit("Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY issued by ConnectomeDB (HCP open access).")
    if shutil.which("aws") is None:
        raise SystemExit("pip install awscli")
    for s in subjects:
        for run in runs:
            key = f"HCP_1200/{s}/MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}_hp2000_clean.nii.gz"
            dest = DATA_DIR / "hcp" / s / f"rfMRI_{run}_hp2000_clean.nii.gz"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"aws s3 cp s3://hcp-openaccess/{key}")
            subprocess.run(["aws", "s3", "cp", f"s3://hcp-openaccess/{key}", str(dest)], check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["openneuro", "atlas", "hcp"], required=True)
    p.add_argument("--dataset", default="ds004884")
    p.add_argument("--include", nargs="*", default=[], help="regex filters on file paths")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--subjects", nargs="*", default=["100307"])
    p.add_argument("--runs", nargs="*", default=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"])
    a = p.parse_args(argv)
    if a.source == "openneuro":
        download_openneuro(a.dataset, a.include, a.sample, a.max_files)
    elif a.source == "atlas":
        atlas_instructions()
    else:
        download_hcp(a.subjects, a.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
