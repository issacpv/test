#!/usr/bin/env python3
"""Build the audit corpus: code repositories linked to biomedical ML papers.

The "data" for this project are public GitHub repositories, not biosignals.
Three acquisition routes are implemented; all read credentials from the
environment and none executes downloaded code.

Examples
--------
    # 1. GitHub repository search for dataset markers (works without a token, slowly)
    python scripts/download_data.py --dataset chbmit --sample
    GITHUB_TOKEN=ghp_... python scripts/download_data.py --dataset ptbxl --max-pages 5

    # 2. GitHub *code* search (needs a token) for file-level markers
    GITHUB_TOKEN=ghp_... python scripts/download_data.py --dataset mimic --code-search

    # 3. Papers citing a dataset descriptor (Semantic Scholar), mined for GitHub links
    S2_API_KEY=... python scripts/download_data.py --citations ptbxl

    # 4. Shallow-clone everything listed in data/corpus/<dataset>_repos.csv
    python scripts/download_data.py --clone data/corpus/chbmit_repos.csv

Outputs land in ``data/corpus/`` (CSV manifests) and ``data/repos/`` (clones);
both are git-ignored.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakscan.corpus import (  # noqa: E402
    DATASET_DOIS,
    DATASET_QUERIES,
    clone_shallow,
    extract_repo_links,
    github_search,
    semantic_scholar_citations,
)

CORPUS = ROOT / "data" / "corpus"
REPOS = ROOT / "data" / "repos"


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def search_repos(dataset: str, max_pages: int, code_search: bool, sample: bool) -> Path:
    token = os.environ.get("GITHUB_TOKEN")
    if code_search and not token:
        sys.exit("code search needs GITHUB_TOKEN")
    q = DATASET_QUERIES[dataset]
    rows = []
    kind = "code" if code_search else "repositories"
    queries = q["code"] if code_search else q["repo"]
    for query in queries:
        full_q = query if code_search else f"{query} language:Python"
        _log(f"github {kind} search: {full_q}")
        n = 0
        for item in github_search(kind, full_q, token=token, max_pages=1 if sample else max_pages, per_page=10 if sample else 100):
            item["dataset"] = dataset
            rows.append(item)
            n += 1
            if sample and n >= 10:
                break
        _log(f"  {n} results")
    df = pd.DataFrame(rows).drop_duplicates(subset=["full_name"]) if rows else pd.DataFrame(columns=["full_name", "dataset"])
    CORPUS.mkdir(parents=True, exist_ok=True)
    out = CORPUS / f"{dataset}_repos.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["full_name"])
    df.to_csv(out, index=False)
    _log(f"wrote {out.relative_to(ROOT)} ({len(df)} repos)")
    return out


def citations(dataset: str, limit: int) -> Path:
    doi = DATASET_DOIS[dataset]
    _log(f"semantic scholar citations of DOI {doi}")
    papers = semantic_scholar_citations(doi, limit=limit)
    rows = []
    for p in papers:
        text = " ".join(str(p.get(k, "")) for k in ("abstract", "title"))
        pdf = (p.get("openAccessPdf") or {}).get("url", "")
        for slug in extract_repo_links(text):
            rows.append({"full_name": slug, "dataset": dataset, "paper_title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"), "doi": (p.get("externalIds") or {}).get("DOI"), "pdf": pdf})
        if not extract_repo_links(text):
            rows.append({"full_name": None, "dataset": dataset, "paper_title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"), "doi": (p.get("externalIds") or {}).get("DOI"), "pdf": pdf})
    CORPUS.mkdir(parents=True, exist_ok=True)
    out = CORPUS / f"{dataset}_citing_papers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    _log(f"{len(papers)} citing papers; {sum(1 for r in rows if r['full_name'])} GitHub links found in abstracts (PDF mining is a separate step, see data/README.md)")
    return out


def clone(csv_path: Path, limit: int | None) -> None:
    df = pd.read_csv(csv_path)
    slugs = [s for s in df["full_name"].dropna().unique().tolist()]
    if limit:
        slugs = slugs[:limit]
    ok = 0
    for s in slugs:
        dest = clone_shallow(s, REPOS)
        if dest is not None:
            ok += 1
        else:
            _log(f"clone failed: {s}")
    _log(f"cloned {ok}/{len(slugs)} into {REPOS.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_QUERIES), help="GitHub search for this dataset marker")
    ap.add_argument("--code-search", action="store_true", help="use the code-search endpoint (token required)")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--sample", action="store_true", help="fetch only ~10 results per query")
    ap.add_argument("--citations", choices=sorted(DATASET_DOIS), help="mine papers citing this dataset descriptor")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--clone", type=Path, help="shallow-clone repos listed in this CSV")
    args = ap.parse_args()
    if args.dataset:
        search_repos(args.dataset, args.max_pages, args.code_search, args.sample)
    if args.citations:
        citations(args.citations, args.limit)
    if args.clone:
        clone(args.clone, args.limit if args.sample else None)
    if not (args.dataset or args.citations or args.clone):
        ap.print_help()


if __name__ == "__main__":
    main()
