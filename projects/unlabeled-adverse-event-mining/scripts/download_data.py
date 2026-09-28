#!/usr/bin/env python
"""Download open inputs for unlabeled-adverse-event-mining.

Modes
-----
--sample              For a small drug list: current SPL labels (openFDA drug/label),
                      FAERS per-quarter reaction counts for each drug (numerators),
                      per-quarter top-1000 reaction totals and total report counts
                      (margins) — enough to run the count-based time scan. A few
                      hundred requests.
--labels-all          Every current prescription SPL via the openFDA bulk label
                      partitions (https://api.fda.gov/download.json, ~1 GB zipped).
--faers-bulk          Quarterly FAERS JSON partitions (see --years).
--dailymed-history    DailyMed version history for set ids in --setid-file.
--potential-signals   Fetch the FDA index of quarterly "potential signals" pages and
                      parse each quarter's table into data/raw/fda_potential_signals.csv.
--srlc                Print instructions for the FDA Safety-related Labeling Changes
                      (SrLC) database export.

Environment: OPENFDA_API_KEY (optional).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unlabeled_ae.label_client import LabelClient, labels_to_frame  # noqa: E402
from unlabeled_ae.label_lag import parse_potential_signals_html  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")
RAW = ROOT / "data" / "raw"

DEFAULT_DRUGS = [
    "ATORVASTATIN", "SEMAGLUTIDE", "TOFACITINIB", "MONTELUKAST", "CIPROFLOXACIN", "FEBUXOSTAT",
    "ZOLPIDEM", "CANAGLIFLOZIN", "APIXABAN", "PEMBROLIZUMAB", "DULOXETINE", "GABAPENTIN",
]
FDA_SIGNALS_INDEX = "https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers/potential-signals-serious-risksnew-safety-information-identified-fda-adverse-event-reporting-system"
UA = {"User-Agent": "unlabeled_ae/0.1 (research)"}


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))


def _quarters(start: str, end: str) -> List[pd.Period]:
    return list(pd.period_range(pd.Period(start, "Q"), pd.Period(end, "Q"), freq="Q"))


def run_sample(drugs: List[str], start_q: str, end_q: str) -> None:
    client = LabelClient()
    out = RAW / "openfda_sample"
    out.mkdir(parents=True, exist_ok=True)
    # 1. labels
    label_rows, section_rows = [], []
    for d in drugs:
        docs = client.labels_for_generic(d)
        log.info("%s: %d current SPL set ids", d, len(docs))
        label_rows.append(labels_to_frame(docs))
        for doc in docs:
            for sec, text in doc.sections.items():
                section_rows.append({"set_id": doc.set_id, "generic_name": doc.generic_name, "version": doc.version, "effective_time": doc.effective_time, "section": sec, "text": text})
    if label_rows:
        _save(pd.concat(label_rows, ignore_index=True), out / "labels.csv")
    _save(pd.DataFrame(section_rows), out / "label_sections.csv")

    # 2. FAERS per-quarter counts
    quarters = _quarters(start_q, end_q)
    pair_rows, drug_rows, pt_rows, all_rows = [], [], [], []
    for q in quarters:
        rng = f"receivedate:[{q.start_time:%Y%m%d} TO {q.end_time:%Y%m%d}]"
        tot = client.count("drug/event", rng, "receivedate")
        all_rows.append({"quarter": str(q), "n": int(tot["count"].sum()) if not tot.empty else 0})
        top = client.count("drug/event", rng, "patient.reaction.reactionmeddrapt.exact", limit=1000)
        for _, r in top.iterrows():
            pt_rows.append({"pt": r["term"], "quarter": str(q), "n": int(r["count"])})
        for d in drugs:
            dq = f'patient.drug.openfda.generic_name:"{d}"+AND+{rng}'
            nd = client.count("drug/event", dq, "receivedate")
            drug_rows.append({"drug": d, "quarter": str(q), "n": int(nd["count"].sum()) if not nd.empty else 0})
            pc = client.count("drug/event", dq, "patient.reaction.reactionmeddrapt.exact", limit=1000)
            for _, r in pc.iterrows():
                pair_rows.append({"drug": d, "pt": r["term"], "quarter": str(q), "n": int(r["count"])})
        log.info("quarter %s done", q)
    _save(pd.DataFrame(pair_rows), out / "faers_pair_counts.csv")
    _save(pd.DataFrame(drug_rows), out / "faers_drug_totals.csv")
    _save(pd.DataFrame(pt_rows), out / "faers_pt_totals.csv")
    _save(pd.DataFrame(all_rows), out / "faers_all_totals.csv")
    log.info("sample complete -> %s (use signal_scan.cumulative_from_counts)", out)


def _download(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600, headers=UA) as r:
        r.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def run_labels_all(dest: Path) -> None:
    manifest = requests.get("https://api.fda.gov/download.json", timeout=60).json()["results"]["drug"]["label"]
    for p in manifest["partitions"]:
        url = p["file"]
        _download(url, dest / url.rsplit("/", 1)[-1])
        time.sleep(0.3)


def run_faers_bulk(years: Optional[Iterable[int]], dest: Path) -> None:
    manifest = requests.get("https://api.fda.gov/download.json", timeout=60).json()["results"]["drug"]["event"]
    wanted = set(years) if years else None
    for p in manifest["partitions"]:
        url = p["file"]
        folder = url.rstrip("/").split("/")[-2]
        year = int(folder[:4]) if folder[:4].isdigit() else None
        if wanted and year not in wanted:
            continue
        _download(url, dest / folder / url.rsplit("/", 1)[-1])
        time.sleep(0.3)


def run_dailymed_history(setid_file: Path, dest: Path) -> None:
    client = LabelClient(max_per_minute=60)
    ids = [l.strip() for l in setid_file.read_text().splitlines() if l.strip()]
    frames = []
    for sid in ids:
        try:
            frames.append(client.fetch_dailymed_history(sid))
        except Exception as exc:  # keep going
            log.warning("%s: %s", sid, exc)
    if frames:
        _save(pd.concat(frames, ignore_index=True), dest / "dailymed_history.csv")


def run_potential_signals(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        html = requests.get(FDA_SIGNALS_INDEX, timeout=60, headers=UA).text
    except requests.RequestException as exc:
        log.error("index fetch failed: %s — save the quarterly pages manually to data/raw/fda_signals_html/ and re-run", exc)
        html = ""
    links = sorted(set(re.findall(r'href="(/drugs/[^"]*potential-signals[^"]*)"', html)))
    log.info("found %d quarterly pages", len(links))
    frames = []
    html_dir = dest / "fda_signals_html"
    html_dir.mkdir(exist_ok=True)
    for href in links:
        url = "https://www.fda.gov" + href
        name = href.rstrip("/").split("/")[-1]
        target = html_dir / f"{name}.html"
        try:
            if not target.exists():
                target.write_text(requests.get(url, timeout=60, headers=UA).text, encoding="utf-8")
                time.sleep(1.0)
            df = parse_potential_signals_html(target.read_text(encoding="utf-8"), quarter_label=name.replace("-", " "))
            frames.append(df)
        except Exception as exc:
            log.warning("%s: %s", url, exc)
    # also parse any manually saved pages
    for f in html_dir.glob("*.html"):
        if not any(f.stem in l for l in links):
            try:
                frames.append(parse_potential_signals_html(f.read_text(encoding="utf-8"), quarter_label=f.stem.replace("-", " ")))
            except Exception as exc:
                log.warning("%s: %s", f, exc)
    if frames:
        allq = pd.concat(frames, ignore_index=True).dropna(subset=["quarter"])
        _save(allq, dest / "fda_potential_signals.csv")
    else:
        log.warning("no tables parsed; check the FDA page layout and parse_potential_signals_html")


SRLC_INSTRUCTIONS = """
FDA Drug Safety-related Labeling Changes (SrLC) database
  https://www.accessdata.fda.gov/scripts/cder/safetylabelingchanges/
  * Search by drug or leave blank and export/download results; each entry gives the
    application number, approval date of the labeling change, the section(s) changed
    (Boxed Warning, Contraindications, Warnings and Precautions, Adverse Reactions ...)
    and a summary of the change.
  * Save as data/raw/srlc.csv with columns: drug, application_number, approval_date,
    section, summary   (rename as needed) and run label_lag.srlc_terms().
  * Coverage starts in 2016 for most drugs; use DailyMed version history + SPL XML
    (--dailymed-history and label_client.fetch_dailymed_spl_xml) for earlier changes.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--drugs", default=",".join(DEFAULT_DRUGS))
    ap.add_argument("--start-quarter", default="2018Q1")
    ap.add_argument("--end-quarter", default="2024Q4")
    ap.add_argument("--labels-all", action="store_true")
    ap.add_argument("--faers-bulk", action="store_true")
    ap.add_argument("--years", default="")
    ap.add_argument("--dailymed-history", action="store_true")
    ap.add_argument("--setid-file", default=str(RAW / "set_ids.txt"))
    ap.add_argument("--potential-signals", action="store_true")
    ap.add_argument("--srlc", action="store_true")
    args = ap.parse_args(argv)
    if not any((args.sample, args.labels_all, args.faers_bulk, args.dailymed_history, args.potential_signals, args.srlc)):
        ap.print_help()
        return 1
    if args.sample:
        run_sample([d.strip().upper() for d in args.drugs.split(",") if d.strip()], args.start_quarter, args.end_quarter)
    if args.labels_all:
        run_labels_all(RAW / "labels_bulk")
    if args.faers_bulk:
        years = [int(y) for y in args.years.split(",") if y.strip()] or None
        run_faers_bulk(years, RAW / "faers_bulk")
    if args.dailymed_history:
        run_dailymed_history(Path(args.setid_file), RAW)
    if args.potential_signals:
        run_potential_signals(RAW)
    if args.srlc:
        print(SRLC_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
