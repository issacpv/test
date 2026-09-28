#!/usr/bin/env python
"""Download paediatric FAERS reports and SPL labels for a drug panel.

--sample   For each panel drug: up to N paediatric reports (numeric age 0-17
           years OR patientagegroup 1-4), the latest label's pediatric_use and
           indications_and_usage sections, and age-group count tables. Also a
           paediatric background sample (all drugs). ~60 requests with defaults.
--full     Print bulk-download instructions.

Outputs (data/raw/): peds_<drug>.jsonl, labels_<drug>.json,
counts_agegroup_<drug>.jsonl, peds_background.jsonl

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

from peds_offlabel.age import flatten_report  # noqa: E402
from peds_offlabel.label_ages import floor_from_label  # noqa: E402
from peds_offlabel.openfda import OpenFDAClient, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("download_data")

#: Drugs with well-known paediatric off-label use and heterogeneous labelled floors.
PANEL = [
    "QUETIAPINE", "RISPERIDONE", "ARIPIPRAZOLE", "OLANZAPINE", "SERTRALINE", "ESCITALOPRAM", "FLUOXETINE",
    "ONDANSETRON", "OMEPRAZOLE", "MONTELUKAST", "TOPIRAMATE", "GABAPENTIN", "LEVETIRACETAM", "LACOSAMIDE",
    "METHYLPHENIDATE", "GUANFACINE", "CLONIDINE", "MELATONIN", "SEMAGLUTIDE", "DUPILUMAB", "ADALIMUMAB",
    "AZITHROMYCIN", "DEXMEDETOMIDINE", "PROPRANOLOL", "TRAMADOL", "CODEINE",
]
PEDS_CLAUSE = "((patient.patientonsetage:[0+TO+17]+AND+patient.patientonsetageunit:801)+OR+patient.patientonsetageunit:(802+OR+803+OR+804+OR+805)+OR+patient.patientagegroup:(1+OR+2+OR+3+OR+4))"
LABEL_SECTIONS = ("pediatric_use", "indications_and_usage", "dosage_and_administration", "boxed_warning", "adverse_reactions")


def sample(panel: list, per_drug: int, start: str, end: str, out: Path, client: OpenFDAClient) -> None:
    out.mkdir(parents=True, exist_ok=True)
    window = client.date_range(start, end)
    for drug in panel:
        clause = f"patient.drug.openfda.generic_name:{client.quote(drug)}+AND+{PEDS_CLAUSE}+AND+{window}"
        total = client.total("drug/event", clause)
        log.info("%s: %d paediatric reports; pulling up to %d", drug, total, per_drug)
        n = write_jsonl((flatten_report(r) for r in client.iter_records("drug/event", clause, limit=100, max_records=per_drug)), out / f"peds_{drug.lower()}.jsonl")
        log.info("  wrote %d", n)
        buckets = client.count("drug/event", f"patient.drug.openfda.generic_name:{client.quote(drug)}+AND+{window}", "patient.patientagegroup", exact=False)
        write_jsonl(buckets, out / f"counts_agegroup_{drug.lower()}.jsonl")
        recs = list(client.iter_records("drug/label", f"openfda.generic_name:{client.quote(drug)}+AND+_exists_:pediatric_use", limit=5, max_records=5, sort="effective_time:desc"))
        if recs:
            lab = {k: recs[0].get(k) for k in ("set_id", "effective_time", "version", *LABEL_SECTIONS)}
            lab["openfda_brand_name"] = (recs[0].get("openfda", {}) or {}).get("brand_name")
            lab["floor"] = floor_from_label(lab)
            (out / f"labels_{drug.lower()}.json").write_text(json.dumps(lab, indent=1))
            log.info("  label floor: %s (%s)", lab["floor"]["min_age_years"], lab["floor"]["status"])
    clause = f"{PEDS_CLAUSE}+AND+{window}"
    n = write_jsonl((flatten_report(r) for r in client.iter_records("drug/event", clause, limit=100, max_records=per_drug * 4)), out / "peds_background.jsonl")
    log.info("paediatric background: wrote %d", n)
    log.info("sample done -> %s", out)


def full_instructions() -> None:
    print(
        """
Full extraction:
1. openFDA bulk partitions (curl -s https://api.fda.gov/download.json ->
   results.drug.event.partitions and results.drug.label.partitions). Stream
   drug/event through peds_offlabel.age.flatten_report and keep reports with
   pediatric == True (about 7-8% of FAERS) plus all adult reports for the
   drugs of interest if adult comparisons are wanted.
2. FAERS quarterly ASCII (DEMO.AGE, AGE_COD, AGE_GRP; CASEID dedup):
   https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
3. FDA Pediatric Labeling Changes table (validation set for age floors and
   dates for the ITS analysis): https://www.fda.gov/science-research/pediatrics/pediatric-labeling-changes
4. DailyMed SPL archive for label history (LOINC 34081-0 = pediatric use):
   https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--drugs", nargs="*", default=PANEL)
    ap.add_argument("--per-drug", type=int, default=300)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    if args.full:
        full_instructions()
    if args.sample:
        sample([d.upper() for d in args.drugs], args.per_drug, args.start, args.end, Path(args.out), OpenFDAClient())
    if not (args.sample or args.full):
        ap.print_help()


if __name__ == "__main__":
    main()
