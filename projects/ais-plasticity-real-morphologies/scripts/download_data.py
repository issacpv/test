#!/usr/bin/env python
"""Download NeuroMorpho.org somatodendritic reconstructions for the AIS plasticity study.

Examples
--------
Smoke test (20 mouse pyramidal records, 5 SWC files)::

    python scripts/download_data.py --sample

Full harvest::

    python scripts/download_data.py --species mouse rat human --classes pyramidal granule interneuron --swc

ModelDB and Allen Cell Types downloads are documented in data/README.md (they need
either a browser or allensdk); this script only handles the NeuroMorpho part.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("download_data")
BASE = "https://neuromorpho.org/api"

CLASS_QUERY = {
    "pyramidal": {"cell_type": ["pyramidal"]},
    "granule": {"cell_type": ["granule"]},
    "interneuron": {"cell_type": ["interneuron"]},
}


class Client:
    def __init__(self, timeout: int = 60, retries: int = 5) -> None:
        self.s = requests.Session()
        self.timeout, self.retries = timeout, retries
        self.verify = os.environ.get("NEUROMORPHO_VERIFY_SSL", "1") != "0"

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        delay = 2.0
        for i in range(self.retries):
            try:
                r = self.s.request(method, url, timeout=self.timeout, verify=self.verify, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(str(r.status_code), response=r)
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                if i == self.retries - 1:
                    raise
                LOG.warning("%s failed (%s); retrying in %.0fs", url, exc, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def pages(self, query: Dict[str, List[str]], size: int = 500, max_pages: Optional[int] = None
              ) -> Iterator[Tuple[int, List[Dict]]]:
        page = 0
        while True:
            d = self._req("POST", f"{BASE}/neuron/select?page={page}&size={size}", json=query).json()
            recs = d.get("_embedded", {}).get("neuronResources", [])
            yield page, recs
            page += 1
            if not recs or page >= int(d.get("page", {}).get("totalPages", page)):
                break
            if max_pages and page >= max_pages:
                break

    def swc(self, rec: Dict, out: Path) -> Path:
        arch = str(rec["archive"]).lower()
        name = rec["neuron_name"]
        dest = out / arch / f"{name}.CNG.swc"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://neuromorpho.org/dableFiles/{quote(arch)}/{quote('CNG version')}/{quote(name)}.CNG.swc"
        dest.write_bytes(self._req("GET", url).content)
        return dest


def dendrites_complete(rec: Dict) -> bool:
    integ = str(rec.get("physical_Integrity") or "").lower()
    dom = str(rec.get("structural_domains") or "").lower()
    return "dendrites complete" in integ and "dendrit" in dom


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--species", nargs="*", default=["mouse", "rat", "human"])
    ap.add_argument("--classes", nargs="*", default=list(CLASS_QUERY), choices=list(CLASS_QUERY))
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--swc", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    species, classes, size, max_pages, swc_limit = a.species, a.classes, 500, a.max_pages, None
    if a.sample:
        species, classes, size, max_pages, swc_limit = ["mouse"], ["pyramidal"], 20, 1, 5

    c = Client()
    meta = a.out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    kept: List[Dict] = []
    for sp in species:
        fname = meta / f"neurons_{sp}.jsonl"
        with fname.open("a") as fh:
            for cls in classes:
                q = {"species": [sp], **CLASS_QUERY[cls]}
                try:
                    for page, recs in c.pages(q, size, max_pages):
                        for r in recs:
                            r["_class_query"] = cls
                            fh.write(json.dumps(r) + "\n")
                            if dendrites_complete(r):
                                kept.append(r)
                        LOG.info("%s/%s page %d: %d records", sp, cls, page, len(recs))
                except Exception as exc:  # noqa: BLE001
                    LOG.error("harvest failed for %s/%s: %s", sp, cls, exc)
    LOG.info("%d records with complete dendrites", len(kept))
    if not kept:
        return 1
    try:
        import pandas as pd

        df = pd.json_normalize(kept)
        for col in ("brain_region", "cell_type", "experiment_condition"):
            if col in df:
                df[col] = df[col].map(lambda v: "; ".join(map(str, v)) if isinstance(v, list) else v)
        df.to_parquet(meta / "neurons.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not write parquet (%s); JSONL is still available", exc)

    if a.swc or a.sample:
        todo = kept if swc_limit is None else kept[:swc_limit]
        n_ok = 0
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(c.swc, r, a.out / "swc"): r for r in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("SWC failed for %s: %s", futs[f].get("neuron_name"), exc)
        LOG.info("downloaded %d/%d SWC files", n_ok, len(todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
