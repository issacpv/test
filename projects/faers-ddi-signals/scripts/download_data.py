#!/usr/bin/env python
"""Download FAERS reports for DDI signal detection via openFDA.

Modes
-----
--sample   Pull a small, paginated sample (default 500 reports per query) for a
           handful of mechanism-panel drug pairs plus reports that carry an
           explicitly "interacting" drug (drugcharacterization = 3). Safe to run
           without an API key (40 req/min) - it uses ~30 requests.
--full     Print instructions for the bulk quarterly JSON dumps (openFDA
           download manifest) and the FAERS quarterly ASCII files; the full
           database (~20 M reports) should not be pulled through the paginated
           API.

Outputs (JSONL, one flattened report per line) go to ``data/raw/``.

Environment: ``OPENFDA_API_KEY`` (optional; raises the rate limit to 240/min).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faers_ddi.openfda import OpenFDAClient, flatten_record, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("download_data")

#: Mechanism panel used in --sample mode: (perpetrator, victim).
SAMPLE_PAIRS = [
    ("CLARITHROMYCIN", "SIMVASTATIN"),
    ("ITRACONAZOLE", "SIMVASTATIN"),
    ("RITONAVIR", "MIDAZOLAM"),
    ("FLUCONAZOLE", "WARFARIN"),
    ("AMIODARONE", "WARFARIN"),
    ("TRAMADOL", "SERTRALINE"),
]


def sample(out_dir: Path, per_query: int, start: str, end: str, client: OpenFDAClient) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    window = client.date_range(start, end)
    # 1) pair co-reports
    for a, b in SAMPLE_PAIRS:
        search = client.pair_clause(a, b, extra=window)
        total = client.total("drug/event", search)
        log.info("%s + %s: %d reports match; pulling up to %d", a, b, total, per_query)
        recs = (flatten_record(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_query))
        n = write_jsonl(recs, out_dir / f"pair_{a}_{b}.jsonl".lower())
        log.info("  wrote %d", n)
    # 2) single-drug backgrounds for each drug in the panel (needed for n10+, n01+)
    for d in sorted({x for p in SAMPLE_PAIRS for x in p}):
        search = f"{client.drug_clause(d)}+AND+{window}"
        recs = (flatten_record(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_query))
        n = write_jsonl(recs, out_dir / f"drug_{d}.jsonl".lower())
        log.info("%s: wrote %d", d, n)
    # 3) reporter-flagged interactions (partial labels)
    search = f"patient.drug.drugcharacterization:3+AND+{window}"
    recs = (flatten_record(r) for r in client.iter_records("drug/event", search, limit=100, max_records=per_query * 2))
    n = write_jsonl(recs, out_dir / "interacting_role.jsonl")
    log.info("interacting-role reports: wrote %d", n)
    # 4) sex x year marginal counts (cheap count queries) for denominators
    for sex_code, sex in (("1", "male"), ("2", "female")):
        buckets = client.count("drug/event", f"patient.patientsex:{sex_code}+AND+{window}", "receivedate", exact=False)
        write_jsonl(buckets, out_dir / f"counts_receivedate_{sex}.jsonl")
    log.info("sample done -> %s", out_dir)


def full_instructions() -> None:
    print(
        """
Full FAERS extraction (do NOT page through the API for this):

1. openFDA bulk JSON (same schema as the API, quarterly partitions):
     curl -s https://api.fda.gov/download.json | python -c "import json,sys; d=json.load(sys.stdin)['results']['drug']['event']; print(d['total_records']); [print(p['file']) for p in d['partitions']]"
   Download every listed .json.zip into data/bulk/ (roughly 20 M reports, ~10 GB zipped)
   and stream them through faers_ddi.openfda.flatten_record.

2. FAERS quarterly ASCII files (DEMO/DRUG/REAC/OUTC/RPSR/THER/INDI tables):
     https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   These carry CASEID/CASEVERSION for proper deduplication (keep the latest
   version per case) and the DRUG.ROLE_COD field (PS/SS/C/I; "I" = interacting).

3. Drug-name harmonisation: use the DiAna dictionary (Fusaroli et al., 2024,
   Drug Saf) or openFDA's `openfda.generic_name` mapping; see README.md.
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="small paginated sample via the API")
    ap.add_argument("--full", action="store_true", help="print bulk-download instructions")
    ap.add_argument("--per-query", type=int, default=500)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    if args.full:
        full_instructions()
    if args.sample:
        sample(Path(args.out), args.per_query, args.start, args.end, OpenFDAClient())
    if not (args.sample or args.full):
        ap.print_help()


if __name__ == "__main__":
    main()
