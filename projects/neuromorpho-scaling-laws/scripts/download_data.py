#!/usr/bin/env python
"""Download NeuroMorpho.org metadata, L-Measure morphometry and CNG.swc files.

Examples
--------
Smoke test (a few species, 25 records each, 5 SWC files each)::

    python scripts/download_data.py --sample

Full metadata for selected species::

    python scripts/download_data.py --species mouse rat human --metadata-only

Everything::

    python scripts/download_data.py --all --morphometry --swc --workers 4

The script is resumable: pages already written to JSONL and SWC files already on
disk are skipped. Set ``NEUROMORPHO_VERIFY_SSL=0`` only if the NeuroMorpho TLS
chain is broken at the time you run this.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nm_scaling.api_client import NeuroMorphoClient, neurons_to_dataframe  # noqa: E402

LOG = logging.getLogger("download_data")

DEFAULT_SAMPLE_SPECIES = ["mouse", "rat", "human", "monkey", "cat", "drosophila melanogaster", "zebrafish"]


def harvest_species(client: NeuroMorphoClient, species: str, out_dir: Path, page_size: int,
                    max_pages: int | None) -> Path:
    """Page through all records for one species and append them to a JSONL file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"neurons_{species.replace(' ', '_')}.jsonl"
    done_pages: set[int] = set()
    if fname.exists():
        with fname.open() as fh:
            for line in fh:
                try:
                    done_pages.add(json.loads(line)["_page"])
                except (KeyError, json.JSONDecodeError):
                    continue
    start_page = (max(done_pages) + 1) if done_pages else 0
    n_written = 0
    with fname.open("a") as fh:
        for page_no, page in client.iter_pages({"species": [species]}, page_size=page_size,
                                               start_page=start_page, max_pages=max_pages):
            for rec in page:
                rec["_page"] = page_no
                fh.write(json.dumps(rec) + "\n")
                n_written += 1
            fh.flush()
    LOG.info("%s: wrote %d new records to %s", species, n_written, fname)
    return fname


def load_jsonl(paths: Iterable[Path]) -> List[Dict]:
    records: List[Dict] = []
    for p in paths:
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def download_swcs(client: NeuroMorphoClient, records: List[Dict], swc_dir: Path, workers: int,
                  limit: int | None = None) -> int:
    """Download CNG.swc files for the given records (skips files already present)."""
    todo = records if limit is None else records[:limit]
    n_ok = 0

    def _one(rec: Dict) -> bool:
        try:
            client.download_swc(rec, swc_dir)
            return True
        except Exception as exc:  # noqa: BLE001 - keep going on individual failures
            LOG.warning("SWC failed for %s/%s: %s", rec.get("archive"), rec.get("neuron_name"), exc)
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, r) for r in todo]
        for fut in as_completed(futures):
            n_ok += int(fut.result())
    LOG.info("downloaded %d/%d SWC files into %s", n_ok, len(todo), swc_dir)
    return n_ok


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="data directory (default: ./data)")
    ap.add_argument("--species", nargs="*", default=None, help="species names as used by NeuroMorpho (e.g. mouse rat human)")
    ap.add_argument("--all", action="store_true", help="harvest every species listed by the API")
    ap.add_argument("--sample", action="store_true", help="small smoke test: 25 records and 5 SWC files per species")
    ap.add_argument("--metadata-only", action="store_true", help="do not download SWC files")
    ap.add_argument("--swc", action="store_true", help="download CNG.swc files for harvested records")
    ap.add_argument("--morphometry", action="store_true", help="also fetch NeuroMorpho L-Measure morphometry")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    client = NeuroMorphoClient()
    meta_dir = args.out / "metadata"
    swc_dir = args.out / "swc"

    if args.sample:
        species = args.species or DEFAULT_SAMPLE_SPECIES
        page_size, max_pages, swc_limit = 25, 1, 5
    else:
        if args.all:
            species = client.field_values("species")
            LOG.info("API lists %d species", len(species))
        elif args.species:
            species = args.species
        else:
            ap.error("give --species ..., --all or --sample")
        page_size, max_pages, swc_limit = args.page_size, args.max_pages, None

    files: List[Path] = []
    for sp in species:
        try:
            files.append(harvest_species(client, sp, meta_dir, page_size, max_pages))
        except Exception as exc:  # noqa: BLE001
            LOG.error("harvest failed for species %r: %s", sp, exc)

    records = load_jsonl(files)
    if not records:
        LOG.error("no records harvested; check network / API status")
        return 1
    df = neurons_to_dataframe(records)
    table = meta_dir / "neurons.parquet"
    try:
        df.to_parquet(table, index=False)
    except (ImportError, ValueError):
        table = table.with_suffix(".csv")
        df.to_csv(table, index=False)
    LOG.info("flattened table with %d rows x %d cols -> %s", *df.shape, table)

    if args.morphometry:
        rows = []
        for i, rec in enumerate(records):
            try:
                m = client.get_morphometry(rec["neuron_name"])
                m["neuron_name"] = rec["neuron_name"]
                rows.append(m)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("morphometry missing for %s: %s", rec.get("neuron_name"), exc)
            if args.sample and i >= 20:
                break
        if rows:
            import pandas as pd

            mdf = pd.DataFrame(rows)
            mpath = meta_dir / "morphometry.parquet"
            try:
                mdf.to_parquet(mpath, index=False)
            except (ImportError, ValueError):
                mpath = mpath.with_suffix(".csv")
                mdf.to_csv(mpath, index=False)
            LOG.info("morphometry table %s rows -> %s", len(mdf), mpath)

    if (args.swc or args.sample) and not args.metadata_only:
        if args.sample:
            # a few per species so the smoke test stays small
            per_species: Dict[str, List[Dict]] = {}
            for r in records:
                per_species.setdefault(str(r.get("species")), []).append(r)
            subset = [r for lst in per_species.values() for r in lst[:swc_limit]]
            download_swcs(client, subset, swc_dir, args.workers)
        else:
            download_swcs(client, records, swc_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
