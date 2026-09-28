#!/usr/bin/env python
"""Download CAERS (openFDA food/event) supplement reports, comparator strata,
and FAERS reports naming the same botanical ingredients.

--sample   Industry-code count table; up to N supplement reports (industry
           code 54), N cosmetic reports (53, comparator), N all-food
           background reports; supplement reaction and outcome count tables;
           FAERS reports whose verbatim product name matches a few botanicals.
           ~25 requests; fine without an API key.
--full     Print bulk-download instructions (openFDA partitions + CAERS
           quarterly CSV from FDA + DSLD).

Outputs (data/raw/): caers_supplements.jsonl, caers_cosmetics.jsonl,
caers_background.jsonl, counts_*.jsonl, faers_<ingredient>.jsonl

Environment: OPENFDA_API_KEY (optional).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caers_signals.normalize import SUPPLEMENT_INDUSTRY_CODE, flatten_caers, flatten_faers_for_supplements  # noqa: E402
from caers_signals.openfda import OpenFDAClient, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("download_data")

FAERS_BOTANICALS = ["TURMERIC", "KRATOM", "ASHWAGANDHA", "GREEN TEA", "KAVA", "MELATONIN", "GARCINIA"]


def sample(per_query: int, start: str, end: str, out: Path, client: OpenFDAClient) -> None:
    out.mkdir(parents=True, exist_ok=True)
    window = client.date_range(start, end, "date_created")
    write_jsonl(client.count("food/event", window, "products.industry_code", exact=True), out / "counts_industry_code.jsonl")
    strata = {"supplements": SUPPLEMENT_INDUSTRY_CODE, "cosmetics": "53"}
    for name, code in strata.items():
        search = f"products.industry_code:{code}+AND+{window}"
        total = client.total("food/event", search)
        log.info("CAERS %s (industry %s): %d reports; pulling up to %d", name, code, total, per_query)
        n = write_jsonl((flatten_caers(r) for r in client.iter_records("food/event", search, limit=100, max_records=per_query)), out / f"caers_{name}.jsonl")
        log.info("  wrote %d", n)
    n = write_jsonl((flatten_caers(r) for r in client.iter_records("food/event", window, limit=100, max_records=per_query)), out / "caers_background.jsonl")
    log.info("CAERS background: wrote %d", n)
    supp = f"products.industry_code:{SUPPLEMENT_INDUSTRY_CODE}+AND+{window}"
    write_jsonl(client.count("food/event", supp, "reactions", exact=True), out / "counts_supplement_reactions.jsonl")
    write_jsonl(client.count("food/event", supp, "outcomes", exact=True), out / "counts_supplement_outcomes.jsonl")
    write_jsonl(client.count("food/event", supp, "products.name_brand", exact=True), out / "counts_supplement_brands.jsonl")
    write_jsonl(client.count("food/event", supp, "date_created", exact=False), out / "counts_supplement_by_day.jsonl")
    fwindow = client.date_range(start, end, "receivedate")
    for ing in FAERS_BOTANICALS:
        search = f"patient.drug.medicinalproduct:{client.quote(ing)}+AND+{fwindow}"
        n = write_jsonl((flatten_faers_for_supplements(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_query // 2)), out / f"faers_{ing.lower().replace(' ', '_')}.jsonl")
        log.info("FAERS %s: wrote %d", ing, n)
    log.info("sample done -> %s", out)


def full_instructions() -> None:
    print(
        """
Full extraction:
1. openFDA bulk: curl -s https://api.fda.gov/download.json -> results.food.event.partitions[*].file
   (CAERS is small: roughly 100-250 k reports in total). Stream through
   caers_signals.normalize.flatten_caers.
2. FDA CAERS quarterly files (CSV/XLSX with the same fields plus CAERS Created Date):
   https://www.fda.gov/food/compliance-enforcement-food/cfsan-adverse-event-reporting-system-caers
3. FAERS supplement-named reports: bulk drug/event partitions filtered with
   caers_signals.normalize.flatten_faers_for_supplements (keep supplement == True).
4. DSLD (ingredient lists per brand): https://dsld.od.nih.gov/api-guide  (API) or the
   full database download at https://dsld.od.nih.gov/ (CSV/JSON export).
5. Reference sets: LiverTox (https://www.ncbi.nlm.nih.gov/books/NBK547852/), FDA tainted
   products list (https://www.fda.gov/drugs/medication-health-fraud/tainted-products-marketed-dietary-supplements-cder).
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--per-query", type=int, default=1000)
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    if args.full:
        full_instructions()
    if args.sample:
        sample(args.per_query, args.start, args.end, Path(args.out), OpenFDAClient())
    if not (args.sample or args.full):
        ap.print_help()


if __name__ == "__main__":
    main()
