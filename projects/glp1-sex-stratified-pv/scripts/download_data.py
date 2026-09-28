#!/usr/bin/env python
"""Download GLP-1 receptor agonist FAERS reports, comparator reports, sex-specific
count tables and labels via openFDA.

--sample   Up to N reports per GLP-1 agent (default 400), N per comparator
           class, sex-specific top-PT count tables for each agent, and the
           latest label per brand. ~50 requests; fine without an API key.
--full     Print bulk-download instructions.

Outputs (data/raw/): reports_<agent>.jsonl, reports_cmp_<class>.jsonl,
counts_<agent>_<sex>.jsonl, counts_background_<sex>.jsonl, labels_<brand>.json

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

from glp1_sexpv.cohort import COMPARATORS, GLP1_AGENTS, flatten_report  # noqa: E402
from glp1_sexpv.openfda import OpenFDAClient, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("download_data")

LABEL_SECTIONS = ("boxed_warning", "warnings_and_cautions", "adverse_reactions", "use_in_specific_populations", "drug_interactions")


def agent_clause(client: OpenFDAClient, agent: str, brands: list) -> str:
    parts = [f"patient.drug.openfda.generic_name:{client.quote(agent)}", f"patient.drug.medicinalproduct:{client.quote(agent)}"]
    parts += [f"patient.drug.openfda.brand_name:{client.quote(b)}" for b in brands]
    return "(" + "+OR+".join(parts) + ")"


def sample(per_query: int, start: str, end: str, out: Path, client: OpenFDAClient, labels: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    window = client.date_range(start, end)
    for agent, meta in GLP1_AGENTS.items():
        clause = f"{agent_clause(client, agent, list(meta['brands']))}+AND+{window}"
        total = client.total("drug/event", clause)
        log.info("%s: %d reports; pulling up to %d", agent, total, per_query)
        n = write_jsonl((flatten_report(r) for r in client.iter_records("drug/event", clause, limit=100, max_records=per_query)), out / f"reports_{agent.lower()}.jsonl")
        log.info("  wrote %d", n)
        for code, sex in (("2", "female"), ("1", "male")):
            buckets = client.count("drug/event", f"{clause}+AND+patient.patientsex:{code}", "patient.reaction.reactionmeddrapt", limit=1000)
            write_jsonl(buckets, out / f"counts_{agent.lower()}_{sex}.jsonl")
    for cls, drugs in COMPARATORS.items():
        clause = "(" + "+OR+".join(f"patient.drug.openfda.generic_name:{client.quote(d)}" for d in drugs) + f")+AND+{window}"
        n = write_jsonl((flatten_report(r) for r in client.iter_records("drug/event", clause, limit=100, max_records=per_query)), out / f"reports_cmp_{cls}.jsonl")
        log.info("comparator %s: wrote %d", cls, n)
    for code, sex in (("2", "female"), ("1", "male")):
        buckets = client.count("drug/event", f"patient.patientsex:{code}+AND+{window}", "patient.reaction.reactionmeddrapt", limit=1000)
        write_jsonl(buckets, out / f"counts_background_{sex}.jsonl")
        write_jsonl([{"total": client.total("drug/event", f"patient.patientsex:{code}+AND+{window}")}], out / f"total_background_{sex}.jsonl")
    if labels:
        for meta in GLP1_AGENTS.values():
            for brand in meta["brands"]:
                recs = list(client.iter_records("drug/label", f"openfda.brand_name:{client.quote(brand)}", limit=5, max_records=5, sort="effective_time:desc"))
                if recs:
                    lab = {k: recs[0].get(k) for k in ("set_id", "effective_time", "version", *LABEL_SECTIONS)}
                    (out / f"labels_{brand.lower()}.json").write_text(json.dumps(lab, indent=1))
    log.info("sample done -> %s", out)


def full_instructions() -> None:
    print(
        """
Full extraction:
1. openFDA bulk partitions: curl -s https://api.fda.gov/download.json -> results.drug.event.partitions[*].file
   Stream through glp1_sexpv.cohort.flatten_report; keep exposed reports AND a
   random 10% background sample (needed for sex-specific backgrounds).
2. FAERS ASCII quarterly files for CASEID dedup and INDI (indication) table:
   https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
3. Denominators: MEPS HC Prescribed Medicines + Full-Year Consolidated files
   (https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp) and
   NHANES RXQ_RX + DEMO (https://wwwn.cdc.gov/nchs/nhanes/). See data/README.md.
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--per-query", type=int, default=400)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    if args.full:
        full_instructions()
    if args.sample:
        sample(args.per_query, args.start, args.end, Path(args.out), OpenFDAClient(), labels=not args.no_labels)
    if not (args.sample or args.full):
        ap.print_help()


if __name__ == "__main__":
    main()
