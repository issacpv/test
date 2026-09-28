#!/usr/bin/env python
"""Download FAERS reports and SPL labels for biosimilar / originator families.

--sample   For a few families (default: adalimumab, infliximab, pegfilgrastim)
           pull up to N reports per product brand (originator + each launched
           biosimilar) plus N INN-only reports, and the latest label for every
           brand. ~60 requests with the defaults; fine without an API key.
--full     Print bulk-download instructions (openFDA partitions + Purple Book).

Outputs: data/raw/reports_<inn>_<product>.jsonl (attributed, flattened) and
data/raw/labels_<brand>.json.

Environment: OPENFDA_API_KEY (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosim_ae.labels import fetch_label  # noqa: E402
from biosim_ae.openfda import OpenFDAClient, write_jsonl  # noqa: E402
from biosim_ae.products import FAMILIES, attribute_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("download_data")


def brand_search(client: OpenFDAClient, brand: str, start: str, end: str) -> str:
    return f"(patient.drug.openfda.brand_name:{client.quote(brand)}+OR+patient.drug.medicinalproduct:{client.quote(brand)})+AND+{client.date_range(start, end)}"


def sample(families: list, per_product: int, start: str, end: str, out: Path, client: OpenFDAClient, labels: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for inn in families:
        fam = FAMILIES[inn]
        brands = [fam.originator_brand] + [b.brand for b in fam.biosimilars if b.us_launch]
        for brand in brands:
            search = brand_search(client, brand, start, end)
            total = client.total("drug/event", search)
            log.info("%s / %s: %d reports; pulling up to %d", inn, brand, total, per_product)
            recs = (attribute_report(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_product))
            n = write_jsonl(recs, out / f"reports_{inn}_{brand}.jsonl".lower().replace(" ", "_"))
            log.info("  wrote %d", n)
            if labels:
                lab = fetch_label(client, brand)
                if lab:
                    (out / f"labels_{brand}.json".lower()).write_text(json.dumps(lab, indent=1))
        # INN-only reports (generic name present, no brand in the family) - sampled, filtered client-side
        search = f"patient.drug.openfda.generic_name:{client.quote(inn)}+AND+{client.date_range(start, end)}"
        recs = (attribute_report(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_product * 2))
        inn_only = (r for r in recs if all(p["kind"] == "inn_only" for p in r["products"] if p["inn"] == inn))
        n = write_jsonl(inn_only, out / f"reports_{inn}_inn_only.jsonl".lower().replace(" ", "_"))
        log.info("%s INN-only: wrote %d", inn, n)
    log.info("sample done -> %s", out)


def full_instructions() -> None:
    print(
        """
Full extraction:
1. openFDA bulk partitions (same schema as the API):
     curl -s https://api.fda.gov/download.json  ->  results.drug.event.partitions[*].file
   Stream every partition through biosim_ae.products.attribute_report and keep
   reports whose `inns` is non-empty (roughly 2-3% of FAERS).
2. FAERS quarterly ASCII (CASEID/CASEVERSION dedup, PRIMARYID, RPSR_COD report
   source, DRUG.NDA_NUM which carries BLA numbers for biologics):
     https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
3. Purple Book (biosimilar / interchangeable catalogue, BLA numbers, dates):
     https://purplebooksearch.fda.gov/  (monthly downloadable CSV)
4. Labels: openFDA drug/label bulk  ->  results.drug.label.partitions
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--families", nargs="*", default=["ADALIMUMAB", "INFLIXIMAB", "PEGFILGRASTIM"])
    ap.add_argument("--per-product", type=int, default=300)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    if args.full:
        full_instructions()
    if args.sample:
        sample([f.upper() for f in args.families], args.per_product, args.start, args.end, Path(args.out), OpenFDAClient(), labels=not args.no_labels)
    if not (args.sample or args.full):
        ap.print_help()


if __name__ == "__main__":
    main()
