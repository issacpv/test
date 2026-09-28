#!/usr/bin/env python
"""Download / discover open laminar-fMRI datasets on OpenNeuro (GraphQL API; no credentials).

Examples
--------
python scripts/download_data.py --list
python scripts/download_data.py --search laminar layer VASO "cortical depth"
python scripts/download_data.py --dataset ds001547 --sample
python scripts/download_data.py --dataset ds003216 --include 'ses-01' anat --max-files 50
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CURATED = {
    "ds003216": "Kenshu whole-brain layer-fMRI VASO+BOLD, 7T, 1 participant x 6 sessions (Huber et al., 2023)",
    "ds001547": "Layer VASO in the visual system (V1), 7T (Huber)",
}


def _gql(query: str, variables: dict | None = None) -> dict:
    if requests is None:
        raise SystemExit("pip install requests")
    r = requests.post(OPENNEURO_GRAPHQL, json={"query": query, "variables": variables or {}}, timeout=60)
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


def search_datasets(keywords: Iterable[str], page_size: int = 100, max_pages: int = 40) -> list[tuple[str, str]]:
    """Scan the public dataset list and keep those whose name/README mention any keyword.

    OpenNeuro's GraphQL exposes a paginated ``datasets`` connection; full-text search is done client-side
    here so that the query stays valid across schema versions.
    """
    pats = [re.compile(k, re.I) for k in keywords]
    hits: list[tuple[str, str]] = []
    after = None
    for _ in range(max_pages):
        q = ("query($first: Int!, $after: String) { datasets(first: $first, after: $after, filterBy: {public: true}) {"
             " pageInfo { hasNextPage endCursor } edges { node { id latestSnapshot { tag description { Name } readme } } } } }")
        d = _gql(q, {"first": page_size, "after": after})["datasets"]
        for e in d["edges"]:
            node = e["node"]
            snap = node.get("latestSnapshot") or {}
            name = ((snap.get("description") or {}).get("Name") or "")
            readme = snap.get("readme") or ""
            text = f"{name}\n{readme}"
            if any(p.search(text) for p in pats):
                hits.append((node["id"], name))
        if not d["pageInfo"]["hasNextPage"]:
            break
        after = d["pageInfo"]["endCursor"]
    return hits


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def download(dataset: str, include: Iterable[str], sample: bool, max_files: int | None) -> None:
    tag = latest_snapshot_tag(dataset)
    pats = [re.compile(p) for p in include]
    out = DATA_DIR / "openneuro" / dataset
    n = 0
    first_sub = None
    for f in iter_files(dataset, tag):
        name = f["filename"]
        if pats and not any(p.search(name) for p in pats):
            continue
        if sample:
            m = re.match(r"(sub-[^/]+)/", name)
            if m:
                first_sub = first_sub or m.group(1)
                if m.group(1) != first_sub:
                    continue
                if not (name.endswith((".json", ".tsv")) or "/anat/" in name
                        or re.search(r"(bold|vaso)\.nii(\.gz)?$", name, re.I)):
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print curated laminar datasets")
    p.add_argument("--search", nargs="*", help="keywords to search OpenNeuro dataset names/READMEs")
    p.add_argument("--dataset", help="OpenNeuro accession to download")
    p.add_argument("--include", nargs="*", default=[], help="regex filters on file paths")
    p.add_argument("--sample", action="store_true", help="one subject: bold/vaso + anat + sidecars")
    p.add_argument("--max-files", type=int, default=None)
    a = p.parse_args(argv)
    if a.list:
        for k, v in CURATED.items():
            print(f"{k}: {v}")
    if a.search:
        for ds, name in search_datasets(a.search):
            print(f"{ds}\t{name}")
    if a.dataset:
        download(a.dataset, a.include, a.sample, a.max_files)
    if not (a.list or a.search or a.dataset):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
