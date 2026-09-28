#!/usr/bin/env python
"""Download DailyMed SPL histories/versions, openFDA FAERS quarterly counts and Drugs@FDA records.

Examples
--------
Smoke test (3 drugs: SPL history + top PTs + one pair count series each)::

    python scripts/download_data.py --sample

Full pulls::

    python scripts/download_data.py --dailymed --drugs data/drug_list.txt --versions
    python scripts/download_data.py --faers --drugs data/drug_list.txt [--pts data/pt_list.txt]
    python scripts/download_data.py --drugsfda

The SrLC export is a manual download (see data/README.md). Outputs are resumable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from label_change.dailymed_spl import DailyMedClient  # noqa: E402
from label_change.openfda_counts import OpenFDACounts, search_clause  # noqa: E402

LOG = logging.getLogger("download_data")
DATA = ROOT / "data"
SAMPLE_DRUGS = ["rosiglitazone", "montelukast", "fluoroquinolone"]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.strip().lower())[:80]


def pull_dailymed(client: DailyMedClient, drugs: List[str], versions: bool, sample: bool) -> None:
    for drug in drugs:
        spls = client.search_spls(drug, max_pages=1 if sample else 20)
        LOG.info("%s: %d SPL set ids", drug, len(spls))
        for spl in spls[: (2 if sample else None)]:
            setid = spl.get("setid")
            if not setid:
                continue
            out = DATA / "dailymed" / setid
            out.mkdir(parents=True, exist_ok=True)
            hfile = out / "history.json"
            if not hfile.exists():
                hist = client.history(setid)
                hfile.write_text(json.dumps({"setid": setid, "title": spl.get("title"), "history": hist}, indent=1))
            if versions and not sample:
                hist = json.loads(hfile.read_text())["history"]
                for h in hist:
                    v = h.get("spl_version")
                    if v is None:
                        continue
                    vfile = out / f"v{v}.xml"
                    if vfile.exists():
                        continue
                    try:
                        vfile.write_text(client.spl_xml(setid, version=int(v)))
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("version %s of %s failed: %s", v, setid, exc)
            cur = out / "current.xml"
            if not cur.exists():
                cur.write_text(client.spl_xml(setid))


def pull_faers(client: OpenFDACounts, drugs: List[str], pts: List[str] | None, start: date, end: date,
               sample: bool) -> None:
    out = DATA / "faers"
    out.mkdir(parents=True, exist_ok=True)
    nfile = out / "_N.csv"
    if not nfile.exists():
        client.daily_counts("", start, end).to_csv(nfile, index=False)
    for drug in drugs:
        dfile = out / f"{_slug(drug)}__ALL.csv"
        if not dfile.exists():
            client.daily_counts(search_clause(drug=drug), start, end).to_csv(dfile, index=False)
        top = client.top_pts(drug, limit=5 if sample else 300)
        pt_list = list(top["term"]) if pts is None else sorted(set(pts) | set(top["term"]))
        if sample:
            pt_list = pt_list[:1]
        for pt in pt_list:
            pfile = out / f"{_slug(drug)}__{_slug(pt)}.csv"
            if not pfile.exists():
                client.daily_counts(search_clause(drug=drug, pt=pt), start, end).to_csv(pfile, index=False)
            tfile = out / f"_PT__{_slug(pt)}.csv"
            if not tfile.exists():
                client.daily_counts(search_clause(pt=pt), start, end).to_csv(tfile, index=False)
        LOG.info("%s: %d PT series", drug, len(pt_list))


def pull_drugsfda(client: OpenFDACounts, sample: bool) -> None:
    out = DATA / "drugsfda" / "applications.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    import requests  # local import: only needed here

    base = "https://api.fda.gov/drug/drugsfda.json"
    n, skip = 0, 0
    with out.open("w") as fh:
        while True:
            params = {"limit": 100, "skip": skip}
            if client.api_key:
                params["api_key"] = client.api_key
            r = requests.get(base, params=params, timeout=60)
            if r.status_code == 404:
                break
            r.raise_for_status()
            rows = r.json().get("results", [])
            if not rows:
                break
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                n += 1
            skip += len(rows)
            if (sample and n >= 100) or skip >= 25000:
                break
    LOG.info("wrote %d Drugs@FDA applications", n)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--dailymed", action="store_true")
    ap.add_argument("--versions", action="store_true", help="also download every SPL version")
    ap.add_argument("--faers", action="store_true")
    ap.add_argument("--drugsfda", action="store_true")
    ap.add_argument("--drugs", type=Path)
    ap.add_argument("--pts", type=Path)
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.sample:
        args.dailymed = args.faers = True
        drugs = SAMPLE_DRUGS
    elif args.drugs and args.drugs.exists():
        drugs = [ln.strip() for ln in args.drugs.read_text().splitlines() if ln.strip()]
    else:
        drugs = []
    if not (args.dailymed or args.faers or args.drugsfda):
        ap.error("nothing to do")
    if (args.dailymed or args.faers) and not drugs:
        ap.error("--drugs <file> required (or --sample)")
    pts = [ln.strip() for ln in args.pts.read_text().splitlines() if ln.strip()] if args.pts and args.pts.exists() else None
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if args.dailymed:
        pull_dailymed(DailyMedClient(), drugs, args.versions, args.sample)
    if args.faers:
        pull_faers(OpenFDACounts(), drugs, pts, start, end, args.sample)
    if args.drugsfda:
        pull_drugsfda(OpenFDACounts(), args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
