#!/usr/bin/env python
"""Harvest NeuroMorpho.org records with disease / aging / epilepsy conditions and their controls.

Examples
--------
Smoke test (two pages of mouse records, keyword filter, 5 SWC files)::

    python scripts/download_data.py --sample

Full metadata for the species that carry the disease data::

    python scripts/download_data.py --species mouse rat human monkey

Archive x condition count table from the local harvest::

    python scripts/download_data.py --report

Resumable: pages already present in the JSONL are skipped and SWC files on disk
are not re-downloaded. Set ``NEUROMORPHO_VERIFY_SSL=0`` only for a session in
which NeuroMorpho's TLS chain is broken.
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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disease_morph.cohort import build_cohort, condition_report, map_condition  # noqa: E402

LOG = logging.getLogger("download_data")
BASE = "https://neuromorpho.org/api"
DEFAULT_SPECIES = ["mouse", "rat", "human", "monkey"]


class NeuroMorphoClient:
    """Thin REST client with retries; only the endpoints this project needs."""

    def __init__(self, base: str = BASE, timeout: int = 60, retries: int = 5) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.verify = os.environ.get("NEUROMORPHO_VERIFY_SSL", "1") != "0"

    def _request(self, method: str, url: str, **kw) -> requests.Response:
        delay = 2.0
        for attempt in range(self.retries):
            try:
                r = self.session.request(method, url, timeout=self.timeout, verify=self.verify, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"status {r.status_code}", response=r)
                r.raise_for_status()
                return r
            except (requests.RequestException, requests.HTTPError) as exc:
                if attempt == self.retries - 1:
                    raise
                LOG.warning("%s %s failed (%s); retry in %.0fs", method, url, exc, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def iter_pages(self, query: Dict[str, List[str]], page_size: int = 500, start_page: int = 0,
                   max_pages: Optional[int] = None) -> Iterator[Tuple[int, List[Dict]]]:
        page = start_page
        while True:
            url = f"{self.base}/neuron/select?page={page}&size={page_size}"
            data = self._request("POST", url, json=query).json()
            records = data.get("_embedded", {}).get("neuronResources", [])
            yield page, records
            info = data.get("page", {})
            page += 1
            if not records or page >= int(info.get("totalPages", page)):
                break
            if max_pages is not None and page - start_page >= max_pages:
                break

    def field_values(self, field: str) -> List[str]:
        data = self._request("GET", f"{self.base}/neuron/fields/{field}").json()
        return list(data.get("fields", []))

    def download_swc(self, record: Dict, out_dir: Path) -> Path:
        archive = str(record["archive"]).lower()
        name = record["neuron_name"]
        dest = out_dir / archive / f"{name}.CNG.swc"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://neuromorpho.org/dableFiles/{quote(archive)}/{quote('CNG version')}/{quote(name)}.CNG.swc"
        r = self._request("GET", url)
        dest.write_bytes(r.content)
        return dest


def harvest_species(client: NeuroMorphoClient, species: str, out_dir: Path, page_size: int,
                    max_pages: Optional[int]) -> Path:
    """Append all records for one species to a JSONL file (skipping pages already present)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"neurons_{species.replace(' ', '_')}.jsonl"
    done: set = set()
    if fname.exists():
        with fname.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["_page"])
                except (KeyError, json.JSONDecodeError):
                    continue
    start = (max(done) + 1) if done else 0
    n = 0
    with fname.open("a") as fh:
        for page_no, page in client.iter_pages({"species": [species]}, page_size, start, max_pages):
            for rec in page:
                rec["_page"] = page_no
                fh.write(json.dumps(rec) + "\n")
                n += 1
            fh.flush()
    LOG.info("%s: %d new records -> %s", species, n, fname)
    return fname


def load_jsonl(paths: Iterable[Path]) -> List[Dict]:
    out: List[Dict] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def download_swcs(client: NeuroMorphoClient, records: List[Dict], swc_dir: Path, workers: int) -> int:
    def _one(rec: Dict) -> bool:
        try:
            client.download_swc(rec, swc_dir)
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.warning("SWC failed for %s/%s: %s", rec.get("archive"), rec.get("neuron_name"), exc)
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, r) for r in records]
        for f in as_completed(futs):
            ok += int(f.result())
    LOG.info("downloaded %d/%d SWC files into %s", ok, len(records), swc_dir)
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--sample", action="store_true", help="two pages of mouse + 5 SWC files")
    ap.add_argument("--report", action="store_true", help="print archive x condition table from local files")
    ap.add_argument("--swc", action="store_true", help="download SWC for cases and their same-archive controls")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    meta_dir = args.out / "metadata"
    swc_dir = args.out / "swc"
    species = args.species or DEFAULT_SPECIES
    page_size, max_pages = args.page_size, args.max_pages
    if args.sample:
        species, page_size, max_pages = ["mouse"], 500, 2

    if not args.report:
        client = NeuroMorphoClient()
        try:
            vocab = client.field_values("experiment_condition")
            (meta_dir).mkdir(parents=True, exist_ok=True)
            (meta_dir / "experiment_condition_values.json").write_text(json.dumps(vocab, indent=1))
            unmatched = [v for v in vocab if map_condition(v) == "other"]
            LOG.info("experiment_condition vocabulary: %d values, %d unmatched by keyword table",
                     len(vocab), len(unmatched))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not fetch condition vocabulary: %s", exc)
        for sp in species:
            try:
                harvest_species(client, sp, meta_dir, page_size, max_pages)
            except Exception as exc:  # noqa: BLE001
                LOG.error("harvest failed for %s: %s", sp, exc)

    records = load_jsonl(meta_dir / f"neurons_{sp.replace(' ', '_')}.jsonl" for sp in species)
    if not records:
        LOG.error("no records on disk; run a harvest first")
        return 1

    import pandas as pd  # local import keeps --help fast

    cohort, audit = build_cohort(records)
    meta_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(meta_dir / "condition_mapping_audit.csv", index=False)
    try:
        cohort.to_parquet(meta_dir / "cohort.parquet", index=False)
    except (ImportError, ValueError):
        cohort.to_csv(meta_dir / "cohort.csv", index=False)
    table = condition_report(cohort)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(table)
    LOG.info("cohort: %d records (%d non-control) across %d archives",
             len(cohort), int((cohort["condition_class"] != "control").sum()), cohort["archive"].nunique())

    if args.swc or args.sample:
        keep = cohort[cohort["in_contrast"]]
        recs = [r for r in records if r["neuron_name"] in set(keep["neuron_name"])]
        if args.sample:
            recs = recs[:5]
        download_swcs(NeuroMorphoClient(), recs, swc_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
