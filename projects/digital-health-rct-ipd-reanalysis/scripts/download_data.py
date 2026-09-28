#!/usr/bin/env python3
"""Build the registry of digital-health RCTs and their IPD-sharing statements (ClinicalTrials.gov v2 API).

This is the only automatically downloadable "data" in the project: the IPD
itself is obtained by application (Vivli, YODA, CSDR, NIMH Data Archive) or
from open repositories, see data/README.md.

Examples
--------
    python scripts/download_data.py --sample                  # one term, one page
    python scripts/download_data.py --start-year 2015         # all terms, all pages (~minutes)
    python scripts/download_data.py --ipd-yes-only            # only trials stating IPD sharing = Yes
    python scripts/download_data.py --terms "digital therapeutic" "remote monitoring"

Output: data/registry/studies.csv (one row per trial), data/registry/sharing_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dh_ipd.registry import DIGITAL_TERMS, build_params, fetch_studies, sharing_summary, studies_to_frame  # noqa: E402

OUT = ROOT / "data" / "registry"


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terms", nargs="*", default=list(DIGITAL_TERMS))
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--status", default="COMPLETED", help="comma-separated overall statuses")
    ap.add_argument("--ipd-yes-only", action="store_true")
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--sample", action="store_true", help="first term, one page of 20")
    args = ap.parse_args()

    terms = args.terms[:1] if args.sample else args.terms
    studies = []
    for term in terms:
        params = build_params(term, args.start_year, tuple(args.status.split(",")), page_size=20 if args.sample else 100, ipd_yes_only=args.ipd_yes_only)
        _log(f"query.intr={term!r}")
        got = fetch_studies(params, max_pages=1 if args.sample else args.max_pages)
        _log(f"  {len(got)} studies")
        studies.extend(got)
    df = studies_to_frame(studies)
    if df.empty:
        sys.exit("no studies returned")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "studies.csv", index=False)
    dig = df[df["digital"]]
    _log(f"{len(df)} unique trials, {len(dig)} classified as digital interventions, {int(dig['ipd_yes'].sum())} with IPD sharing = YES")
    summ = sharing_summary(dig, by=("sponsor_class",))
    summ.to_csv(OUT / "sharing_summary.csv", index=False)
    print(summ.to_string(index=False))
    _log(f"wrote {OUT.relative_to(ROOT)}/studies.csv and sharing_summary.csv")


if __name__ == "__main__":
    main()
