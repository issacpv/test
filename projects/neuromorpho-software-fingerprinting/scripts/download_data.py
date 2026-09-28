#!/usr/bin/env python
"""Harvest NeuroMorpho.org metadata stratified by reconstruction software and download CNG.swc files.

Examples
--------
Smoke test::

    python scripts/download_data.py --sample

Software labels with record counts::

    python scripts/download_data.py --software-list

Balanced harvest (cap cells per archive within each software class) + SWC files::

    python scripts/download_data.py --software Neurolucida neuTube Vaa3D --cap-per-archive 300 --swc

Set ``NEUROMORPHO_VERIFY_SSL=0`` only if the NeuroMorpho TLS chain is temporarily broken.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://neuromorpho.org/api"
DABLE = "https://neuromorpho.org/dableFiles"
LOG = logging.getLogger("download_data")


def _verify() -> bool:
    return os.environ.get("NEUROMORPHO_VERIFY_SSL", "1").strip().lower() not in {"0", "false", "no"}


class Client:
    """Minimal resumable NeuroMorpho REST client with retries."""

    def __init__(self, timeout: float = 60.0, retries: int = 5) -> None:
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json", "User-Agent": "nm_fingerprint/0.1"})
        self.timeout, self.retries = timeout, retries

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        for attempt in range(self.retries + 1):
            try:
                r = self.s.request(method, url, timeout=self.timeout, verify=_verify(), **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if attempt == self.retries:
                    raise
                wait = 2.0 ** attempt
                LOG.warning("%s %s failed (%s); retry in %.0fs", method, url, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def field_values(self, field: str) -> List[str]:
        return list(self._req("GET", f"{API}/neuron/fields/{field}").json().get("fields", []))

    def select(self, filters: Dict[str, Sequence[str]], page_size: int = 500, max_pages: Optional[int] = None
               ) -> Iterator[Dict[str, Any]]:
        page = 0
        while True:
            data = self._req("POST", f"{API}/neuron/select", params={"page": page, "size": page_size},
                             json={k: list(v) for k, v in filters.items()}).json()
            recs: List[Dict[str, Any]] = []
            for v in data.get("_embedded", {}).values():
                if isinstance(v, list):
                    recs.extend(v)
            if not recs:
                return
            yield from recs
            page += 1
            total_pages = int(data.get("page", {}).get("totalPages", 0))
            if (max_pages and page >= max_pages) or (total_pages and page >= total_pages):
                return

    def count(self, filters: Dict[str, Sequence[str]]) -> int:
        data = self._req("POST", f"{API}/neuron/select", params={"page": 0, "size": 1},
                         json={k: list(v) for k, v in filters.items()}).json()
        return int(data.get("page", {}).get("totalElements", 0))

    def download_swc(self, rec: Dict[str, Any], out_dir: Path) -> Path:
        archive, name = str(rec["archive"]), str(rec["neuron_name"])
        target = out_dir / archive / f"{name}.CNG.swc"
        if target.exists() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{DABLE}/{quote(archive.lower())}/CNG%20version/{quote(name)}.CNG.swc"
        r = self._req("GET", url, stream=True)
        tmp = target.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        tmp.replace(target)
        return target


def flatten(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in rec.items():
        if k.startswith("_") or isinstance(v, dict):
            continue
        out[k] = "|".join(map(str, v)) if isinstance(v, list) else v
    return out


def harvest_software(client: Client, software: str, out_dir: Path, cap_per_archive: Optional[int],
                     max_pages: Optional[int], page_size: int) -> List[Dict[str, Any]]:
    """Fetch all records for one software label, capped per archive; append to a JSONL file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"neurons_{software.replace(' ', '_').replace('/', '_')}.jsonl"
    if fname.exists():
        with fname.open() as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        LOG.info("%s: %d records already on disk (%s)", software, len(recs), fname)
        return recs
    per_archive: Dict[str, int] = defaultdict(int)
    kept: List[Dict[str, Any]] = []
    for rec in client.select({"reconstruction_software": [software]}, page_size=page_size, max_pages=max_pages):
        arch = str(rec.get("archive"))
        if cap_per_archive is not None and per_archive[arch] >= cap_per_archive:
            continue
        per_archive[arch] += 1
        kept.append(rec)
    with fname.open("w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    LOG.info("%s: kept %d records from %d archives -> %s", software, len(kept), len(per_archive), fname)
    return kept


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--software-list", action="store_true", help="print software labels with record counts")
    ap.add_argument("--software", nargs="*", default=None, help="software labels as listed by the API")
    ap.add_argument("--all-software", action="store_true")
    ap.add_argument("--cap-per-archive", type=int, default=None)
    ap.add_argument("--swc", action="store_true", help="download CNG.swc files for harvested records")
    ap.add_argument("--sample", action="store_true", help="smoke test: 3 software labels, 1 page, 5 SWC each")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = Client()
    meta_dir, swc_dir = args.out / "metadata", args.out / "swc"

    if args.software_list:
        labels = client.field_values("reconstruction_software")
        counts = {}
        for lab in labels:
            try:
                counts[lab] = client.count({"reconstruction_software": [lab]})
            except Exception as exc:  # noqa: BLE001
                LOG.warning("count failed for %r: %s", lab, exc)
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "software_counts.json").write_text(json.dumps(counts, indent=2))
        for lab, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{n:8d}  {lab}")
        return 0

    if args.sample:
        software, max_pages, page_size, swc_limit = ["Neurolucida", "neuTube", "Vaa3D"], 1, 20, 5
    else:
        if args.all_software:
            software = client.field_values("reconstruction_software")
        elif args.software:
            software = args.software
        else:
            ap.error("give --software ..., --all-software, --software-list or --sample")
        max_pages, page_size, swc_limit = args.max_pages, args.page_size, None

    records: List[Dict[str, Any]] = []
    for sw in software:
        try:
            records += harvest_software(client, sw, meta_dir, args.cap_per_archive, max_pages, page_size)
        except Exception as exc:  # noqa: BLE001
            LOG.error("harvest failed for %r: %s", sw, exc)
    if not records:
        LOG.error("no records harvested")
        return 1

    import pandas as pd

    df = pd.DataFrame([flatten(r) for r in records])
    table = meta_dir / "neurons.parquet"
    try:
        df.to_parquet(table, index=False)
    except (ImportError, ValueError):
        table = table.with_suffix(".csv")
        df.to_csv(table, index=False)
    LOG.info("metadata table %s rows -> %s", len(df), table)

    if args.swc or args.sample:
        todo = records
        if swc_limit is not None:
            by_sw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in records:
                by_sw[str(r.get("reconstruction_software"))].append(r)
            todo = [r for lst in by_sw.values() for r in lst[:swc_limit]]
        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(client.download_swc, r, swc_dir): r for r in todo}
            for fut in as_completed(futs):
                try:
                    fut.result()
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    r = futs[fut]
                    LOG.warning("SWC failed %s/%s: %s", r.get("archive"), r.get("neuron_name"), exc)
        LOG.info("downloaded %d/%d SWC files -> %s", n_ok, len(todo), swc_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
