#!/usr/bin/env python
"""Download shortage episodes, FAERS monthly count series and NDC product classes from openFDA.

Examples
--------
Smoke test (100 shortage records, count series for 5 demo drugs)::

    python scripts/download_data.py --sample

Full exposure list and count series for every ingredient in a list::

    python scripts/download_data.py --shortages
    python scripts/download_data.py --counts --drugs data/drug_list.txt
    python scripts/download_data.py --ndc-classes

Set ``OPENFDA_API_KEY`` to raise the rate limit. Output goes under ``data/`` (git-ignored) and is
resumable: existing count files are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shortage_ae.openfda_client import OpenFDAClient, build_faers_search, flatten_shortage_record  # noqa: E402
from shortage_ae.shortage_panel import OUTCOME_GROUPS, normalize_name  # noqa: E402

LOG = logging.getLogger("download_data")
DATA = ROOT / "data"
SAMPLE_DRUGS = ["heparin", "norepinephrine", "semaglutide", "lorazepam", "cisplatin"]


def download_shortages(client: OpenFDAClient, out: Path, sample: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for rec in client.iter_records("drug/shortages", search="", limit=1000,
                                       max_records=100 if sample else None):
            fh.write(json.dumps(rec) + "\n")
            n += 1
    LOG.info("wrote %d shortage records to %s", n, out)
    return n


def download_counts(client: OpenFDAClient, drugs: Iterable[str], out_dir: Path, start: date, end: date,
                    groups: Iterable[str] = tuple(OUTCOME_GROUPS)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        key = normalize_name(drug) or drug.lower()
        for g in groups:
            fname = out_dir / f"{key.replace(' ', '_')}__{g}.csv"
            if fname.exists():
                continue
            pts = OUTCOME_GROUPS[g]
            search = build_faers_search(drug=drug, reactions=pts, suspect_only=True)
            df = client.count_by_date("drug/event", search, "receivedate", start, end)
            df.to_csv(fname, index=False)
            LOG.info("%s/%s: %d days, %d reports", key, g, len(df), int(df["count"].sum()) if len(df) else 0)
        # seriousness of error reports (H6)
        fname = out_dir / f"{key.replace(' ', '_')}__error_serious.csv"
        if not fname.exists():
            search = build_faers_search(drug=drug, reactions=OUTCOME_GROUPS["error"], serious=True)
            client.count_by_date("drug/event", search, "receivedate", start, end).to_csv(fname, index=False)


def download_ndc_classes(client: OpenFDAClient, out: Path, sample: bool) -> None:
    """Products with EPC class and route (for substitute definition)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        # split by first letter of generic name to stay under the skip ceiling
        for letter in "abcdefghijklmnopqrstuvwxyz":
            search = f"generic_name:{letter}*"
            for rec in client.iter_records("drug/ndc", search=search, limit=1000,
                                           max_records=200 if sample else None):
                of = rec.get("openfda", {}) or {}
                row = {
                    "generic_name": rec.get("generic_name"),
                    "drug": normalize_name(rec.get("generic_name")),
                    "route": ";".join(rec.get("route", []) or []),
                    "dosage_form": rec.get("dosage_form"),
                    "epc": of.get("pharm_class_epc", []),
                    "product_ndc": rec.get("product_ndc"),
                }
                fh.write(json.dumps(row) + "\n")
                n += 1
            if sample and n >= 200:
                break
    LOG.info("wrote %d NDC products to %s", n, out)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="small smoke-test pull")
    ap.add_argument("--shortages", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--ndc-classes", action="store_true")
    ap.add_argument("--drugs", type=Path, help="text file, one ingredient per line")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    client = OpenFDAClient()
    if args.sample:
        args.shortages = args.counts = True
    if not (args.shortages or args.counts or args.ndc_classes):
        ap.error("nothing to do: pass --sample, --shortages, --counts and/or --ndc-classes")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if args.shortages:
        download_shortages(client, DATA / "shortages" / "openfda_shortages.jsonl", args.sample)
    if args.counts:
        if args.sample:
            drugs = SAMPLE_DRUGS
        elif args.drugs and args.drugs.exists():
            drugs = [ln.strip() for ln in args.drugs.read_text().splitlines() if ln.strip()]
        else:
            ap.error("--counts needs --drugs <file> (or --sample)")
        download_counts(client, drugs, DATA / "faers_counts", start, end)
    if args.ndc_classes:
        download_ndc_classes(client, DATA / "ndc" / "ndc_products.jsonl", args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
