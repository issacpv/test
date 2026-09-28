#!/usr/bin/env python
"""Download the open inputs for faers-reporting-bias.

Sub-commands / flags
--------------------
--sample            Small, fast pull via the openFDA API (default drug set, count
                    queries only: sex, age, reporter type, monthly series, top PTs,
                    and 2x2 tables for a few drug-event pairs). ~150 API calls.
--faers-bulk        Download the quarterly FAERS JSON partitions listed in
                    https://api.fda.gov/download.json for the drug/event endpoint
                    (several GB; filter with --years).
--partd             Download the CMS "Medicare Part D Spending by Drug" CSV via the
                    data.cms.gov data API (needs --cms-dataset-id, see data/README.md).
--meps              Print MEPS acquisition instructions and attempt to download the
                    files named by --meps-files from AHRQ.
--drugs             Comma-separated generic names for --sample (default: a curated
                    set spanning the DSC seed file).

Environment
-----------
OPENFDA_API_KEY     optional, raises the rate limit from 40 to 240 requests/min.

Outputs go under data/raw/<source>/ (see data/README.md for the layout).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faers_bias.openfda_client import OpenFDAClient, QUALIFICATION_LABELS, SEX_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")

DATA = ROOT / "data"
RAW = DATA / "raw"

DEFAULT_DRUGS = [
    "ATORVASTATIN",
    "SIMVASTATIN",
    "ROSUVASTATIN",
    "CIPROFLOXACIN",
    "LEVOFLOXACIN",
    "CANAGLIFLOZIN",
    "EMPAGLIFLOZIN",
    "FEBUXOSTAT",
    "MONTELUKAST",
    "TOFACITINIB",
    "SEMAGLUTIDE",
    "METFORMIN",
    "LISINOPRIL",
    "SERTRALINE",
    "ZOLPIDEM",
    "LEVOTHYROXINE",
]

# drug-event pairs with known or suspected sex differences (for 2x2 sampling)
SAMPLE_PAIRS = [
    ("ATORVASTATIN", "Diabetes mellitus"),
    ("ATORVASTATIN", "Myalgia"),
    ("ZOLPIDEM", "Somnambulism"),
    ("SEMAGLUTIDE", "Suicidal ideation"),
    ("CIPROFLOXACIN", "Tendon rupture"),
    ("MONTELUKAST", "Depression"),
    ("LEVOFLOXACIN", "Aortic aneurysm"),
    ("CANAGLIFLOZIN", "Diabetic ketoacidosis"),
]


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))


# ------------------------------------------------------------------ sample
def run_sample(drugs: List[str], start: str, end: str) -> None:
    client = OpenFDAClient()
    out = RAW / "openfda_sample"
    out.mkdir(parents=True, exist_ok=True)
    log.info("openFDA sample pull for %d drugs (%s..%s); key=%s", len(drugs), start, end, bool(client.api_key))

    # all-FAERS monthly totals = ITS offset
    total = client.faers_monthly_series(search="", start=start, end=end)
    _save(total, out / "faers_monthly_total.csv")
    for sex_code, sex in (("1", "male"), ("2", "female")):
        s = client.faers_monthly_series(search=f"patient.patientsex:{sex_code}", start=start, end=end)
        _save(s, out / f"faers_monthly_total_{sex}.csv")

    sex_rows, rep_rows, age_rows = [], [], []
    for drug in drugs:
        q = client.faers_drug_search(drug)
        sx = client.faers_sex_counts(drug)
        sx["drug_norm"] = drug
        sex_rows.append(sx)
        rp = client.faers_reporter_counts(drug)
        rp["drug_norm"] = drug
        rep_rows.append(rp)
        age = client.count("drug/event", q, "patient.patientonsetage", exact=False)
        age["drug_norm"] = drug
        age_rows.append(age)
        monthly = client.faers_monthly_series(search=q, start=start, end=end)
        _save(monthly, out / "monthly" / f"{drug}.csv")
        for sex_code, sex in (("1", "male"), ("2", "female")):
            m = client.faers_monthly_series(search=f"{q}+AND+patient.patientsex:{sex_code}", start=start, end=end)
            _save(m, out / "monthly" / f"{drug}_{sex}.csv")
        top = client.faers_reaction_counts(drug, limit=200)
        _save(top, out / "reactions" / f"{drug}.csv")
        # sex-specific top reactions (numerators for sex-stratified 2x2)
        for sex_code, sex in (("1", "male"), ("2", "female")):
            t = client.count("drug/event", f"{q}+AND+patient.patientsex:{sex_code}", "patient.reaction.reactionmeddrapt", limit=200)
            _save(t, out / "reactions" / f"{drug}_{sex}.csv")
    _save(pd.concat(sex_rows, ignore_index=True), out / "sex_counts.csv")
    _save(pd.concat(rep_rows, ignore_index=True), out / "reporter_counts.csv")
    _save(pd.concat(age_rows, ignore_index=True), out / "age_counts.csv")

    # 2x2 tables, overall and by sex, for sample pairs
    rows = []
    for drug, pt in SAMPLE_PAIRS:
        for stratum, extra in (("all", None), ("female", "patient.patientsex:2"), ("male", "patient.patientsex:1")):
            t = client.faers_two_by_two(drug, pt, search_extra=extra)
            t.update({"drug_norm": drug, "reaction_pt": pt, "stratum": stratum})
            rows.append(t)
            log.info("2x2 %s / %s [%s]: %s", drug, pt, stratum, t)
    _save(pd.DataFrame(rows), out / "two_by_two_sample.csv")
    log.info("sample pull complete -> %s", out)


# --------------------------------------------------------------- bulk FAERS
def run_faers_bulk(years: Optional[Iterable[int]], dest: Path) -> None:
    """Download quarterly drug/event JSON zips listed in openFDA's download manifest."""
    manifest = requests.get("https://api.fda.gov/download.json", timeout=60).json()
    parts = manifest["results"]["drug"]["event"]["partitions"]
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(int(y) for y in years) if years else None
    n = 0
    for p in parts:
        url = p["file"]
        name = url.rsplit("/", 1)[-1]  # e.g. drug-event-0001-of-0035.json.zip under a YYYYqN folder
        folder = url.rstrip("/").split("/")[-2]  # e.g. 2023q4
        year = int(folder[:4]) if folder[:4].isdigit() else None
        if wanted and (year is None or year not in wanted):
            continue
        target = dest / folder / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s (%s)", url, p.get("size_mb", "?"))
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        n += 1
        time.sleep(0.5)
    log.info("downloaded %d partitions to %s", n, dest)


# --------------------------------------------------------------- Part D
def run_partd(dataset_id: str, dest: Path, page_size: int = 5000) -> None:
    """Page through the data.cms.gov data API for a dataset id.

    The "Medicare Part D Spending by Drug" dataset page exposes an "API" link of
    the form https://data.cms.gov/data-api/v1/dataset/<UUID>/data ; copy the UUID
    into --cms-dataset-id (or set CMS_PARTD_DATASET_ID). Alternatively download
    the CSV from the dataset page and place it at data/raw/partd/partd_spending_by_drug.csv.
    """
    base = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
    frames = []
    offset = 0
    while True:
        r = requests.get(base, params={"size": page_size, "offset": offset}, timeout=120)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        log.info("Part D rows fetched: %d", offset)
        if len(rows) < page_size:
            break
    df = pd.concat(frames, ignore_index=True)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "partd_spending_by_drug.csv", index=False)
    log.info("wrote %s", dest / "partd_spending_by_drug.csv")


# ------------------------------------------------------------------ MEPS
MEPS_INSTRUCTIONS = """
MEPS Household Component public use files (AHRQ) — free, no registration.
  1. Prescribed Medicines event file for the year, e.g. 2023 = HC-248A:
     https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-248A
     Download the "Data file, CSV" (or the .xlsx / SAS transport) and unzip to data/raw/meps/.
  2. Full-Year Consolidated file for the same year (person-level SEX, AGE23X, PERWT23F):
     https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp  (search "Full Year Consolidated")
  3. Repeat for each year you need (2018-2023 recommended; file numbers change yearly).
The direct file URLs follow the pattern
  https://meps.ahrq.gov/mepsweb/data_files/pufs/h248a/h248acsv.zip  (CSV)  and
  https://meps.ahrq.gov/mepsweb/data_files/pufs/h248/h248csv.zip
but AHRQ occasionally changes hosting; use --meps-files to pass explicit URLs.
"""


def run_meps(urls: List[str], dest: Path) -> None:
    print(MEPS_INSTRUCTIONS)
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        target = dest / name
        if target.exists():
            log.info("exists: %s", target)
            continue
        log.info("downloading %s", url)
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except requests.RequestException as exc:
            log.error("failed %s: %s (download manually, see instructions above)", url, exc)


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="small openFDA pull (counts only)")
    ap.add_argument("--faers-bulk", action="store_true", help="download quarterly FAERS JSON partitions")
    ap.add_argument("--years", type=str, default="", help="comma-separated years for --faers-bulk, e.g. 2019,2020")
    ap.add_argument("--partd", action="store_true", help="download Part D spending-by-drug via data.cms.gov API")
    ap.add_argument("--cms-dataset-id", type=str, default=os.environ.get("CMS_PARTD_DATASET_ID", ""))
    ap.add_argument("--meps", action="store_true", help="print MEPS instructions / attempt download")
    ap.add_argument("--meps-files", type=str, default="", help="comma-separated MEPS zip URLs")
    ap.add_argument("--drugs", type=str, default=",".join(DEFAULT_DRUGS))
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default="2025-06-30")
    args = ap.parse_args(argv)

    if not any((args.sample, args.faers_bulk, args.partd, args.meps)):
        ap.print_help()
        return 1
    if args.sample:
        run_sample([d.strip().upper() for d in args.drugs.split(",") if d.strip()], args.start, args.end)
    if args.faers_bulk:
        years = [int(y) for y in args.years.split(",") if y.strip()] or None
        run_faers_bulk(years, RAW / "faers_bulk")
    if args.partd:
        if not args.cms_dataset_id:
            log.error("--partd needs --cms-dataset-id or CMS_PARTD_DATASET_ID (see data/README.md)")
            return 2
        run_partd(args.cms_dataset_id, RAW / "partd")
    if args.meps:
        urls = [u.strip() for u in args.meps_files.split(",") if u.strip()]
        run_meps(urls, RAW / "meps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
