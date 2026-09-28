#!/usr/bin/env python
"""Download NeuroMorpho.org retinal ganglion cell metadata and CNG.swc files.

Examples
--------
Smoke test (3 species, 25 records each, 10 SWC files)::

    python scripts/download_data.py --sample

Full::

    python scripts/download_data.py --metadata --swc
    python scripts/download_data.py --metadata --swc --species mouse rat rabbit cat monkey human

Eyewire museum exports and empirical threshold tables are manual (see data/README.md).
Resumable: existing metadata files and SWC files are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neuromorpho"
BASE = "https://neuromorpho.org/api"
DABLE = "https://neuromorpho.org/dableFiles"
LOG = logging.getLogger("download_data")
DEFAULT_SPECIES = ["mouse", "rat", "rabbit", "cat", "monkey", "human", "salamander", "goldfish", "zebrafish"]


def _verify() -> bool:
    return os.environ.get("NEUROMORPHO_VERIFY_SSL", "1").strip() not in {"0", "false", "no"}


def _get(session: requests.Session, url: str, params=None, retries: int = 5) -> requests.Response:
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=60, verify=_verify())
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            LOG.warning("%s failed (%s); retrying", url, exc)
            time.sleep(2.0 ** attempt)
    raise RuntimeError("unreachable")


def iter_retina(session: requests.Session, species: str, size: int = 500, max_pages: Optional[int] = None) -> Iterator[Dict]:
    page = 0
    while True:
        params = [("q", "brain_region:retina"), ("q", f"species:{species}"), ("page", page), ("size", size)]
        data = _get(session, f"{BASE}/neuron/select", params=params).json()
        recs = data.get("_embedded", {}).get("neuronResources", [])
        for r in recs:
            yield r
        total_pages = data.get("page", {}).get("totalPages", page + 1)
        page += 1
        if not recs or page >= total_pages or (max_pages is not None and page >= max_pages):
            return


def is_ganglion(rec: Dict) -> bool:
    ct = rec.get("cell_type", [])
    ct = ct if isinstance(ct, list) else [ct]
    return any("ganglion" in str(c).lower() for c in ct)


def swc_url(rec: Dict) -> str:
    return f"{DABLE}/{quote(str(rec.get('archive', '')).lower())}/CNG version/{quote(rec.get('neuron_name', ''))}.CNG.swc"


def download_swc(session: requests.Session, rec: Dict, out_dir: Path) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{rec.get('neuron_name')}.CNG.swc"
    if dest.exists():
        return dest
    try:
        r = _get(session, swc_url(rec))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("SWC for %s failed: %s", rec.get("neuron_name"), exc)
        return None
    dest.write_text(r.text)
    return dest


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--swc", action="store_true")
    ap.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    ap.add_argument("--max-swc", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.sample:
        args.metadata = args.swc = True
        species, max_pages, size, max_swc = ["mouse", "rat", "cat"], 1, 25, 10
    else:
        species, max_pages, size, max_swc = args.species, None, 500, args.max_swc
    if not (args.metadata or args.swc):
        ap.error("pass --sample, --metadata and/or --swc")
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "rgc_prosthesis/0.1 (research)"})
    meta_dir, swc_dir = DATA / "meta", DATA / "swc"
    meta_dir.mkdir(parents=True, exist_ok=True)
    n_swc = 0
    for sp in species:
        fname = meta_dir / f"{sp}.jsonl"
        recs: List[Dict] = []
        if args.metadata and not fname.exists():
            with fname.open("w") as fh:
                for rec in iter_retina(session, sp, size=size, max_pages=max_pages):
                    if is_ganglion(rec):
                        fh.write(json.dumps(rec) + "\n")
                        recs.append(rec)
            LOG.info("%s: %d ganglion-cell records", sp, len(recs))
        elif fname.exists():
            recs = [json.loads(l) for l in fname.read_text().splitlines() if l.strip()]
        if args.swc:
            for rec in recs:
                if max_swc is not None and n_swc >= max_swc:
                    break
                if download_swc(session, rec, swc_dir) is not None:
                    n_swc += 1
                time.sleep(0.2)
    LOG.info("done; %d SWC files under %s", n_swc, swc_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
