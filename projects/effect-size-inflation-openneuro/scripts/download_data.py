#!/usr/bin/env python
"""Download OpenNeuro metadata (and optional paper / NeuroVault links) for the effect-size audit.

Examples
--------
    python scripts/download_data.py --out data/openneuro --sample 50
    python scripts/download_data.py --out data/openneuro
    python scripts/download_data.py --out data/openneuro --fallback-github --sample 30
    python scripts/download_data.py --introspect
    python scripts/download_data.py --resolve-dois data/openneuro/datasets.csv --out data/papers
    python scripts/download_data.py --neurovault data/openneuro/datasets.csv --out data/neurovault

Only open resources are touched. Nothing here requires credentials; ``NCBI_API_KEY`` and
``GITHUB_TOKEN`` are optional and only raise rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from es_inflation.openneuro_client import (  # noqa: E402
    GRAPHQL_URL,
    OpenNeuroClient,
    extract_dois,
    fetch_dataset_from_github,
    iter_dataset_ids,
)

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NEUROVAULT = "https://neurovault.org/api"


def cmd_index(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = OpenNeuroClient(url=args.url, page_size=args.page_size, cache_dir=out / "raw_pages")
    limit = args.sample if args.sample else None
    rows = client.fetch_datasets_table(max_datasets=limit)
    if rows.empty:
        print("No datasets returned; try --fallback-github", file=sys.stderr)
        sys.exit(1)
    rows.to_csv(out / "datasets.csv", index=False)
    print(f"wrote {len(rows)} datasets -> {out / 'datasets.csv'}")
    print(rows[["dataset_id", "n_subjects", "modalities"]].head(10).to_string(index=False))


def cmd_fallback_github(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    rows = []
    misses = 0
    for ds_id in iter_dataset_ids(start=args.start_id, stop=args.stop_id):
        rec = fetch_dataset_from_github(ds_id, session=session)
        if rec is None:
            misses += 1
            if misses >= args.max_consecutive_misses:
                print(f"{args.max_consecutive_misses} consecutive misses after {ds_id}; stopping")
                break
            continue
        misses = 0
        rows.append(rec)
        print(f"{ds_id}: n={rec['n_subjects']} {rec['name'][:50]!r}")
        if args.sample and len(rows) >= args.sample:
            break
        time.sleep(args.sleep)
    df = pd.DataFrame(rows)
    df.to_csv(out / "datasets.csv", index=False)
    print(f"wrote {len(df)} datasets -> {out / 'datasets.csv'}")


def cmd_introspect(args: argparse.Namespace) -> None:
    client = OpenNeuroClient(url=args.url)
    for type_name in ("Query", "Dataset", "Snapshot", "Summary", "Description", "Metadata"):
        fields = client.introspect_type(type_name)
        print(f"== {type_name}")
        for f in fields:
            print(f"   {f}")


def cmd_resolve_dois(args: argparse.Namespace) -> None:
    src = pd.read_csv(args.resolve_dois)
    out = Path(args.out)
    (out / "pmc").mkdir(parents=True, exist_ok=True)
    dois: set[str] = set()
    for col in ("dataset_doi", "reference_dois"):
        if col in src:
            for cell in src[col].dropna().astype(str):
                dois.update(d for d in extract_dois(cell))
    print(f"{len(dois)} unique DOIs")
    params_base = {"db": "pubmed", "retmode": "json"}
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params_base["api_key"] = key
    records = []
    for i, doi in enumerate(sorted(dois)):
        if args.sample and i >= args.sample:
            break
        try:
            r = requests.get(f"{NCBI}/esearch.fcgi", params={**params_base, "term": f"{doi}[DOI]"}, timeout=30)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:  # noqa: BLE001
            print(f"  {doi}: esearch failed ({exc})")
            ids = []
        pmid = ids[0] if ids else None
        pmcid = None
        if pmid:
            try:
                r = requests.get(
                    f"{NCBI}/elink.fcgi",
                    params={**params_base, "dbfrom": "pubmed", "db": "pmc", "id": pmid},
                    timeout=30,
                )
                r.raise_for_status()
                linksets = r.json().get("linksets", [])
                for ls in linksets:
                    for ldb in ls.get("linksetdbs", []):
                        if ldb.get("dbto") == "pmc" and ldb.get("links"):
                            pmcid = ldb["links"][0]
            except Exception as exc:  # noqa: BLE001
                print(f"  {doi}: elink failed ({exc})")
        if pmcid and args.fetch_fulltext:
            try:
                r = requests.get(
                    f"{NCBI}/efetch.fcgi",
                    params={"db": "pmc", "id": pmcid, "retmode": "xml", **({"api_key": key} if key else {})},
                    timeout=60,
                )
                r.raise_for_status()
                (out / "pmc" / f"PMC{pmcid}.xml").write_text(r.text)
            except Exception as exc:  # noqa: BLE001
                print(f"  {doi}: efetch failed ({exc})")
        records.append({"doi": doi, "pmid": pmid, "pmcid": pmcid})
        time.sleep(0.11 if key else 0.35)
    pd.DataFrame(records).to_csv(out / "doi_to_pmid.csv", index=False)
    print(f"wrote {len(records)} rows -> {out / 'doi_to_pmid.csv'}")


def cmd_neurovault(args: argparse.Namespace) -> None:
    src = pd.read_csv(args.neurovault)
    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    dois: set[str] = set()
    for col in ("dataset_doi", "reference_dois"):
        if col in src:
            for cell in src[col].dropna().astype(str):
                dois.update(extract_dois(cell))
    rows = []
    for i, doi in enumerate(sorted(dois)):
        if args.sample and i >= args.sample:
            break
        try:
            r = requests.get(f"{NEUROVAULT}/collections/", params={"DOI": doi, "format": "json"}, timeout=30)
            r.raise_for_status()
            for coll in r.json().get("results", []):
                rows.append({"doi": doi, "collection_id": coll["id"], "name": coll.get("name"), "n_images": coll.get("number_of_images")})
                if args.download_maps:
                    imgs = requests.get(f"{NEUROVAULT}/collections/{coll['id']}/images/", params={"format": "json"}, timeout=60).json()
                    for img in imgs.get("results", []):
                        if img.get("is_thresholded"):
                            continue
                        fn = out / "maps" / f"{coll['id']}_{img['id']}.nii.gz"
                        if not fn.exists():
                            with requests.get(img["file"], stream=True, timeout=120) as rr:
                                rr.raise_for_status()
                                with open(fn, "wb") as fh:
                                    for chunk in rr.iter_content(1 << 20):
                                        fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"  {doi}: NeuroVault lookup failed ({exc})")
        time.sleep(0.2)
    pd.DataFrame(rows).to_csv(out / "collections.csv", index=False)
    print(f"wrote {len(rows)} collections -> {out / 'collections.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/openneuro")
    p.add_argument("--url", default=GRAPHQL_URL)
    p.add_argument("--sample", type=int, default=0, help="stop after this many datasets / DOIs")
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--introspect", action="store_true", help="print the live GraphQL schema for key types")
    p.add_argument("--fallback-github", action="store_true", help="use OpenNeuroDatasets GitHub mirrors instead of GraphQL")
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--stop-id", type=int, default=7000)
    p.add_argument("--max-consecutive-misses", type=int, default=400)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--resolve-dois", metavar="DATASETS_CSV", help="map DOIs to PMIDs/PMCIDs via NCBI E-utilities")
    p.add_argument("--fetch-fulltext", action="store_true", help="with --resolve-dois: also fetch PMC OA XML")
    p.add_argument("--neurovault", metavar="DATASETS_CSV", help="find NeuroVault collections for the linked DOIs")
    p.add_argument("--download-maps", action="store_true", help="with --neurovault: download unthresholded maps")
    args = p.parse_args()

    if args.introspect:
        cmd_introspect(args)
    elif args.resolve_dois:
        cmd_resolve_dois(args)
    elif args.neurovault:
        cmd_neurovault(args)
    elif args.fallback_github:
        cmd_fallback_github(args)
    else:
        cmd_index(args)


if __name__ == "__main__":
    main()
