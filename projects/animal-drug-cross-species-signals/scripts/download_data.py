#!/usr/bin/env python
"""Harvest openFDA CVM animal adverse-event reports, FAERS counts and Drugs@FDA records.

Examples
--------
Smoke test (~500 dog/cat reports from one month, flattened)::

    python scripts/download_data.py --sample

Full harvest and FAERS counts for the shared ingredients::

    python scripts/download_data.py --animal --start 2008-01-01 --species Dog Cat
    python scripts/download_data.py --faers --ingredients data/shared_ingredients.txt
    python scripts/download_data.py --drugsfda

Outputs are resumable (window files that already exist are skipped).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xspecies_pv.openfda_animal import (  # noqa: E402
    ANIMAL_ENDPOINT, HUMAN_ENDPOINT, OpenFDAAnimalClient, faers_search, flatten_animal_record,
)
from xspecies_pv.term_mapping import normalize_ingredient  # noqa: E402

LOG = logging.getLogger("download_data")
DATA = ROOT / "data"
SAMPLE_INGREDIENTS = ["meloxicam", "phenobarbital", "fluoxetine", "gabapentin", "levothyroxine"]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.strip().lower())[:80]


def harvest_animal(client: OpenFDAAnimalClient, start: date, end: date, species: Optional[List[str]],
                   step_days: int, max_records: Optional[int]) -> None:
    raw_dir = DATA / "animal" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=step_days - 1), end)
        fname = raw_dir / f"{cur:%Y%m%d}_{nxt:%Y%m%d}.jsonl"
        if not fname.exists():
            with fname.open("w") as fh:
                for rec in client.iter_animal_events(cur, nxt, species=species, step_days=step_days,
                                                     max_records=(max_records - n) if max_records else None):
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
            LOG.info("%s: %d records so far", fname.name, n)
        if max_records and n >= max_records:
            break
        cur = nxt + timedelta(days=1)


def flatten_all() -> None:
    import pandas as pd

    raw_dir = DATA / "animal" / "raw"
    out = DATA / "animal" / "flat" / "animal_events.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        rows = []
        with f.open() as fh:
            for line in fh:
                if line.strip():
                    rows.extend(flatten_animal_record(json.loads(line)))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        LOG.warning("no raw records to flatten")
        return
    df = pd.concat(frames, ignore_index=True)
    df["ingredient_norm"] = df["active_ingredient"].map(normalize_ingredient)
    try:
        df.to_parquet(out, index=False)
    except (ImportError, ValueError):
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    LOG.info("flattened %d rows -> %s", len(df), out)


def harvest_faers(client: OpenFDAAnimalClient, ingredients: List[str], start: date, end: date, sample: bool) -> None:
    out = DATA / "faers"
    out.mkdir(parents=True, exist_ok=True)
    nfile = out / "_N.csv"
    if not nfile.exists():
        client.daily_counts(HUMAN_ENDPOINT, "", "receivedate", start, end).to_csv(nfile, index=False)
    for ing in ingredients:
        tot = out / f"{_slug(ing)}__ALL.csv"
        if not tot.exists():
            client.daily_counts(HUMAN_ENDPOINT, faers_search(ingredient=ing), "receivedate", start, end).to_csv(tot, index=False)
        top = client.count(HUMAN_ENDPOINT, faers_search(ingredient=ing), "patient.reaction.reactionmeddrapt.exact",
                           limit=5 if sample else 300)
        for pt in list(top.get("term", [])):
            pf = out / f"{_slug(ing)}__{_slug(pt)}.csv"
            if not pf.exists():
                client.daily_counts(HUMAN_ENDPOINT, faers_search(ingredient=ing, pt=pt), "receivedate", start, end).to_csv(pf, index=False)
            tf = out / f"_PT__{_slug(pt)}.csv"
            if not tf.exists():
                client.daily_counts(HUMAN_ENDPOINT, faers_search(pt=pt), "receivedate", start, end).to_csv(tf, index=False)
        LOG.info("%s: %d PTs", ing, len(top))


def harvest_animal_counts(client: OpenFDAAnimalClient, ingredients: List[str], start: date, end: date,
                          species: str = "Dog") -> None:
    """Cheap alternative to the full harvest: per-ingredient VeDDRA term counts for one species."""
    out = DATA / "animal" / "counts"
    out.mkdir(parents=True, exist_ok=True)
    for ing in ingredients:
        f = out / f"{species.lower()}__{_slug(ing)}__terms.csv"
        if f.exists():
            continue
        search = f'animal.species:{species}+AND+drug.active_ingredients.name:"{ing.upper()}"'
        client.count(ANIMAL_ENDPOINT, search, "reaction.veddra_term_name.exact", limit=1000).to_csv(f, index=False)


def harvest_drugsfda(client: OpenFDAAnimalClient, sample: bool) -> None:
    out = DATA / "drugsfda" / "applications.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for rec in client.iter_window("drug/drugsfda", search="", limit=100):
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if sample and n >= 100:
                break
    LOG.info("wrote %d Drugs@FDA applications", n)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--animal", action="store_true")
    ap.add_argument("--animal-counts", action="store_true", help="per-ingredient VeDDRA counts only")
    ap.add_argument("--faers", action="store_true")
    ap.add_argument("--drugsfda", action="store_true")
    ap.add_argument("--flatten", action="store_true", help="flatten existing raw files")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--ingredients", type=Path)
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = OpenFDAAnimalClient()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    ingredients = ([ln.strip() for ln in args.ingredients.read_text().splitlines() if ln.strip()]
                   if args.ingredients and args.ingredients.exists() else SAMPLE_INGREDIENTS)
    if args.sample:
        harvest_animal(client, date(2022, 3, 1), date(2022, 3, 31), ["Dog", "Cat"], 31, max_records=500)
        flatten_all()
        harvest_faers(client, SAMPLE_INGREDIENTS[:2], date(2020, 1, 1), date(2020, 12, 31), sample=True)
        return 0
    if not any([args.animal, args.animal_counts, args.faers, args.drugsfda, args.flatten]):
        ap.error("nothing to do")
    if args.animal:
        harvest_animal(client, start, end, args.species, args.step_days, None)
        flatten_all()
    if args.flatten:
        flatten_all()
    if args.animal_counts:
        for sp in (args.species or ["Dog", "Cat"]):
            harvest_animal_counts(client, ingredients, start, end, species=sp)
    if args.faers:
        harvest_faers(client, ingredients, start, end, sample=False)
    if args.drugsfda:
        harvest_drugsfda(client, sample=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
