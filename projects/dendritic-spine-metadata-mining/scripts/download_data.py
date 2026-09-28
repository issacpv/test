#!/usr/bin/env python
"""Download the open inputs: Europe PMC open-access full texts and NeuroMorpho.org metadata/SWC.

Examples
--------
Smoke test (20 papers + a few NeuroMorpho records/SWC files)::

    python scripts/download_data.py --europepmc --neuromorpho --sample

Literature corpus::

    python scripts/download_data.py --europepmc --max-papers 3000

NeuroMorpho metadata and SWC files for selected species::

    python scripts/download_data.py --neuromorpho --species mouse rat human --swc --workers 4

EM connectome tables need per-dataset clients (see data/README.md) and are not automated here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spine_mining.europepmc import EuropePMC, xml_to_text  # noqa: E402

LOG = logging.getLogger("download_data")
NM_API = "https://neuromorpho.org/api"
NM_DABLE = "https://neuromorpho.org/dableFiles"

DEFAULT_QUERY = ('("spine density" OR "dendritic spine" OR "dendritic spines") AND '
                 '("per micrometer" OR "per micron" OR "per µm" OR "spines/µm" OR "spines/um" OR "per 10 µm" OR "per 100 µm") '
                 'AND OPEN_ACCESS:Y AND HAS_FT:Y')


# --------------------------------------------------------------------------- Europe PMC
def fetch_corpus(out_dir: Path, query: str, max_papers: int, sleep: float = 0.2) -> int:
    client = EuropePMC()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xml").mkdir(exist_ok=True)
    (out_dir / "text").mkdir(exist_ok=True)
    hits_path = out_dir / "hits.jsonl"
    seen = set()
    if hits_path.exists():
        with hits_path.open() as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line).get("pmcid"))
                except json.JSONDecodeError:
                    continue
    n_new = 0
    with hits_path.open("a") as fh:
        for hit in client.search(query, page_size=100, max_results=max_papers):
            pmcid = hit.get("pmcid")
            if not pmcid:
                continue
            if pmcid not in seen:
                fh.write(json.dumps(hit) + "\n")
                seen.add(pmcid)
            xml_path = out_dir / "xml" / f"{pmcid}.xml"
            if xml_path.exists():
                continue
            try:
                xml = client.full_text_xml(pmcid)
            except requests.RequestException as exc:
                LOG.warning("full text failed for %s: %s", pmcid, exc)
                continue
            xml_path.write_text(xml)
            (out_dir / "text" / f"{pmcid}.txt").write_text(xml_to_text(xml))
            n_new += 1
            time.sleep(sleep)
    LOG.info("Europe PMC: %d new full texts (total hits on disk: %d)", n_new, len(seen))
    return n_new


# --------------------------------------------------------------------------- NeuroMorpho
def _nm_verify() -> bool:
    return os.environ.get("NEUROMORPHO_VERIFY_SSL", "1").strip().lower() not in {"0", "false", "no"}


def nm_select(filters: Dict[str, Sequence[str]], page_size: int = 500, max_pages: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    page = 0
    while True:
        r = requests.post(f"{NM_API}/neuron/select", params={"page": page, "size": page_size},
                          json={k: list(v) for k, v in filters.items()}, timeout=90, verify=_nm_verify())
        r.raise_for_status()
        data = r.json()
        recs: List[Dict[str, Any]] = []
        for v in data.get("_embedded", {}).values():
            if isinstance(v, list):
                recs.extend(v)
        if not recs:
            return
        yield from recs
        page += 1
        total = int(data.get("page", {}).get("totalPages", 0))
        if (max_pages and page >= max_pages) or (total and page >= total):
            return


def nm_download_swc(rec: Dict[str, Any], out_dir: Path) -> Path:
    archive, name = str(rec["archive"]), str(rec["neuron_name"])
    target = out_dir / archive / f"{name}.CNG.swc"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{NM_DABLE}/{quote(archive.lower())}/CNG%20version/{quote(name)}.CNG.swc"
    with requests.get(url, stream=True, timeout=120, verify=_nm_verify()) as r:
        r.raise_for_status()
        tmp = target.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        tmp.replace(target)
    return target


def fetch_neuromorpho(out_dir: Path, species: List[str], swc: bool, page_size: int, max_pages: Optional[int],
                      workers: int, swc_limit: Optional[int]) -> int:
    meta_dir, swc_dir = out_dir / "metadata", out_dir / "swc"
    meta_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for sp in species:
        fname = meta_dir / f"neurons_{sp.replace(' ', '_')}.jsonl"
        if fname.exists():
            with fname.open() as fh:
                recs = [json.loads(l) for l in fh if l.strip()]
        else:
            recs = list(nm_select({"species": [sp]}, page_size=page_size, max_pages=max_pages))
            with fname.open("w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        LOG.info("%s: %d records", sp, len(recs))
        records += recs
    if not records:
        return 0
    import pandas as pd

    flat = [{k: ("|".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items() if not k.startswith("_") and not isinstance(v, dict)}
            for r in records]
    df = pd.DataFrame(flat)
    try:
        df.to_parquet(meta_dir / "neurons.parquet", index=False)
    except (ImportError, ValueError):
        df.to_csv(meta_dir / "neurons.csv", index=False)
    if swc:
        todo = records if swc_limit is None else records[:swc_limit]
        n_ok = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(nm_download_swc, r, swc_dir) for r in todo]
            for f in as_completed(futs):
                try:
                    f.result()
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("SWC failed: %s", exc)
        LOG.info("downloaded %d/%d SWC files", n_ok, len(todo))
    return len(records)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--europepmc", action="store_true")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--max-papers", type=int, default=500)
    ap.add_argument("--neuromorpho", action="store_true")
    ap.add_argument("--species", nargs="*", default=["mouse", "rat", "human"])
    ap.add_argument("--swc", action="store_true")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not (args.europepmc or args.neuromorpho):
        ap.print_help()
        return 0
    if args.europepmc:
        fetch_corpus(args.out / "europepmc", args.query, 20 if args.sample else args.max_papers)
    if args.neuromorpho:
        if args.sample:
            fetch_neuromorpho(args.out / "neuromorpho", args.species, swc=True, page_size=20, max_pages=1, workers=args.workers, swc_limit=10)
        else:
            fetch_neuromorpho(args.out / "neuromorpho", args.species, args.swc, args.page_size, args.max_pages, args.workers, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
