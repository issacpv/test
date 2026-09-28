#!/usr/bin/env python
"""Download helpers for cvr-normative-maps.

* ``openneuro`` – EuskalIBUR (ds003192) or any other OpenNeuro dataset via the GraphQL API (open).
                  ``--sample`` pulls one subject/one session: breath-hold echoes, physio, events, T1w.
* ``nki``        – NKI-Rockland enhanced imaging from the public ``fcp-indi`` S3 bucket (awscli,
                  ``--no-sign-request``); phenotypes require the NKI DUA and are not downloadable here.

Examples
--------
python scripts/download_data.py --source openneuro --dataset ds003192 --sample
python scripts/download_data.py --source nki --sample
python scripts/download_data.py --source nki --subjects sub-A00008326 --tasks BREATHHOLD rest
"""
from __future__ import annotations

import argparse
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
NKI_S3 = "s3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


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
    n = 0
    first_sub: str | None = None
    first_ses: str | None = None
    for f in iter_files(dataset, tag):
        name = f["filename"]
        if pats and not any(p.search(name) for p in pats):
            continue
        if sample:
            m = re.match(r"(sub-[^/]+)/(ses-[^/]+)?", name)
            if m:
                sub, ses = m.group(1), m.group(2)
                first_sub = first_sub or sub
                if sub != first_sub:
                    continue
                if ses:
                    first_ses = first_ses or ses
                    if ses != first_ses:
                        continue
                if not ("breathhold" in name.lower() or "physio" in name.lower() or "T1w" in name
                        or name.endswith((".json", ".tsv"))):
                    continue
            elif not name.endswith((".json", ".tsv", "README", "CHANGES")):
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


def _aws() -> str:
    exe = shutil.which("aws")
    if exe is None:
        raise SystemExit("pip install awscli   (public bucket; no credentials needed)")
    return exe


def list_nki_subjects() -> list[str]:
    aws = _aws()
    res = subprocess.run([aws, "s3", "ls", "--no-sign-request", NKI_S3 + "/"], capture_output=True, text=True, check=True)
    return [line.split()[-1].rstrip("/") for line in res.stdout.splitlines() if "sub-" in line]


def download_nki(subjects: list[str] | None, tasks: list[str], sample: bool) -> None:
    aws = _aws()
    if sample or not subjects:
        subjects = list_nki_subjects()[:1] if sample else list_nki_subjects()
    out = DATA_DIR / "nki"
    for sub in subjects:
        cmd = [aws, "s3", "sync", "--no-sign-request", f"{NKI_S3}/{sub}", str(out / sub),
               "--exclude", "*", "--include", "*/anat/*T1w*"]
        for t in tasks:
            cmd += ["--include", f"*task-{t}*"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    print("NOTE: phenotypes (age, sex, vitals, medical history) need the NKI-RS DUA; see data/README.md")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["openneuro", "nki"], required=True)
    p.add_argument("--dataset", default="ds003192")
    p.add_argument("--include", nargs="*", default=[])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=["BREATHHOLD", "rest"])
    a = p.parse_args(argv)
    if a.source == "openneuro":
        download_openneuro(a.dataset, a.include, a.sample, a.max_files)
    else:
        download_nki(a.subjects, a.tasks, a.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
