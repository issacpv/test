#!/usr/bin/env python
"""Fetch Allen connectivity data, convert MouseLight JSON exports, and pull NeuroMorpho SWCs.

Examples
--------
    python scripts/download_data.py --allen --sample
    python scripts/download_data.py --allen [--cre]
    python scripts/download_data.py --mouselight-json data/mouselight/mouselight_all.json
    python scripts/download_data.py --neuromorpho --pmid 34616072 [--source-version] [--sample]
    python scripts/download_data.py --neuromorpho --archive Mouselight --sample
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meso_vs_axon.allen_connectivity import HAVE_ALLENSDK, ConnectivityFetcher  # noqa: E402
from meso_vs_axon.ccf_assign import mouselight_json_to_swc_files  # noqa: E402

LOG = logging.getLogger("download_data")

SAMPLE_SOURCES = ["MOp", "SSp-bfd", "VISp", "ACAd", "VPM", "LGd"]


def do_allen(out: Path, sample: bool, cre: bool) -> int:
    if not HAVE_ALLENSDK:
        LOG.error("allensdk not installed (pip install allensdk)")
        return 1
    fetcher = ConnectivityFetcher(out / "allen", resolution=25)
    summary = fetcher.summary_structures()
    summary.to_csv(out / "allen" / "structures_summary.csv", index=False)
    LOG.info("%d summary structures", len(summary))
    fetcher.annotation()  # downloads the 25 um CCF annotation into the cache
    exps = fetcher.experiments(cre=None if cre else False,
                               injection_structure_acronyms=SAMPLE_SOURCES if sample else None)
    if sample:
        exps = exps.head(10)
    exps.to_csv(out / "allen" / "experiments.csv", index=False)
    LOG.info("%d experiments", len(exps))
    for parameter in ("normalized_projection_volume", "projection_density"):
        mat = fetcher.projection_matrix(exps["id"].tolist(), parameter=parameter)
        path = out / "allen" / f"projection_matrix_{parameter}.csv"
        mat.to_csv(path)
        LOG.info("wrote %s (%d x %d)", path, *mat.shape)
    return 0


def do_neuromorpho(out: Path, pmid: Optional[str], archive: Optional[str], source_version: bool, sample: bool,
                   max_pages: int) -> int:
    from urllib.parse import quote

    import pandas as pd
    import requests

    api = "https://neuromorpho.org/api/neuron/select"
    body = {}
    if pmid:
        body["reference_pmid"] = [str(pmid)]
    if archive:
        body["archive"] = [archive]
    if not body:
        LOG.error("give --pmid and/or --archive")
        return 1
    dest = out / "swc" / ("neuromorpho_" + (archive or f"pmid{pmid}"))
    dest.mkdir(parents=True, exist_ok=True)
    rows, n_files = [], 0
    with requests.Session() as sess:
        for page in range(1 if sample else max_pages):
            r = sess.post(api, json=body, params={"page": page, "size": 25 if sample else 500}, timeout=60)
            r.raise_for_status()
            recs = [x for v in r.json().get("_embedded", {}).values() if isinstance(v, list) for x in v]
            if not recs:
                break
            for rec in recs:
                rows.append({k: rec.get(k) for k in ("neuron_id", "neuron_name", "archive", "species", "brain_region",
                                                     "cell_type", "original_format", "reference_pmid")})
                if sample and n_files >= 5:
                    continue
                arch, name = str(rec["archive"]), str(rec["neuron_name"])
                if source_version:
                    ext = str(rec.get("original_format") or "swc").split(".")[-1]
                    url = f"https://neuromorpho.org/dableFiles/{quote(arch.lower())}/Source-Version/{quote(name)}.{ext}"
                    target = dest / f"{name}.{ext}"
                else:
                    url = f"https://neuromorpho.org/dableFiles/{quote(arch.lower())}/CNG%20version/{quote(name)}.CNG.swc"
                    target = dest / f"{name}.CNG.swc"
                if target.exists():
                    continue
                resp = sess.get(url, timeout=120)
                if resp.status_code == 200:
                    target.write_bytes(resp.content)
                    n_files += 1
                else:
                    LOG.debug("missing %s (%s)", url, resp.status_code)
    pd.DataFrame(rows).to_csv(dest / "metadata.csv", index=False)
    LOG.info("%d records, %d files -> %s", len(rows), n_files, dest)
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--allen", action="store_true")
    ap.add_argument("--cre", action="store_true", help="include Cre-line experiments")
    ap.add_argument("--mouselight-json", type=Path, default=None, help="MouseLight neuron-browser JSON export")
    ap.add_argument("--neuromorpho", action="store_true")
    ap.add_argument("--pmid", default=None, help="NeuroMorpho reference PMID (e.g. 34616072 for SEU-ALLEN)")
    ap.add_argument("--archive", default=None, help="NeuroMorpho archive name (e.g. Mouselight)")
    ap.add_argument("--source-version", action="store_true", help="fetch original-coordinate files")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    if args.allen:
        rc |= do_allen(args.out, args.sample, args.cre)
    if args.mouselight_json:
        n = mouselight_json_to_swc_files(args.mouselight_json, args.out / "swc" / "mouselight")
        LOG.info("converted %d MouseLight neurons", n)
    if args.neuromorpho:
        rc |= do_neuromorpho(args.out, args.pmid, args.archive, args.source_version, args.sample, args.max_pages)
    if not (args.allen or args.mouselight_json or args.neuromorpho):
        ap.error("nothing to do: use --allen, --mouselight-json or --neuromorpho")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
