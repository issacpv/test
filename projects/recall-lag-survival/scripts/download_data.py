#!/usr/bin/env python3
"""Download openFDA device recall / enforcement / MAUDE / clearance data for ``recall_lag``.

All sources are open; ``OPENFDA_API_KEY`` (optional) is read from the environment.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --full
    python scripts/download_data.py --full --endpoints recall,enforcement
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recall_lag.openfda_client import OpenFDAClient  # noqa: E402

PAGED = {
    "recall": "device/recall",
    "enforcement": "device/enforcement",
    "510k": "device/510k",
    "pma": "device/pma",
    "classification": "device/classification",
    "registrationlisting": "device/registrationlisting",
}


def _write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


def download_sample(client: OpenFDAClient, out: Path, n_recalls: int, per_code: int) -> None:
    base = out / "openfda" / "sample"
    recalls = client.fetch_records("device/recall", None, max_records=n_recalls, sort="event_date_initiated:desc")
    print(f"recalls: {len(recalls)}")
    _write_jsonl(base / "recall.jsonl", recalls)
    enforcement = []
    for r in recalls[:100]:
        ren = r.get("res_event_number")
        if ren:
            enforcement += client.fetch_records("device/enforcement", f"res_event_number:{ren}", max_records=5)
    print(f"enforcement rows for first 100 recalls: {len(enforcement)}")
    _write_jsonl(base / "enforcement.jsonl", enforcement)
    codes = Counter(str(r.get("product_code", "")).upper() for r in recalls if r.get("product_code"))
    for code, _ in codes.most_common(20):
        recs = client.fetch_records("device/event", f"device.device_report_product_code:{code}", max_records=per_code)
        n = _write_jsonl(base / f"event_{code}.jsonl", recs)
        print(f"  MAUDE {code}: {n}")


def download_full(client: OpenFDAClient, out: Path, endpoints: list[str]) -> None:
    base = out / "openfda"
    if "event" in endpoints:
        parts = client.bulk_partitions("device", "event")
        dest = base / "device_event"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{len(parts)} MAUDE partitions")
        for p in parts:
            f = dest / p["file"].rsplit("/", 1)[-1]
            if f.exists():
                continue
            with requests.get(p["file"], stream=True, timeout=600) as r:
                r.raise_for_status()
                with f.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            print("  saved", f.name)
    for name, ep in PAGED.items():
        if name in endpoints:
            n = _write_jsonl(base / f"{name}.jsonl", client.iter_records(ep, None))
            print(f"{name}: {n} records")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--full", action="store_true")
    p.add_argument("--n-recalls", type=int, default=500)
    p.add_argument("--per-code", type=int, default=200)
    p.add_argument("--endpoints", type=str, default="event,recall,enforcement,510k,pma,classification")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args(argv)
    client = OpenFDAClient()
    if a.sample:
        download_sample(client, a.out, a.n_recalls, a.per_code)
    else:
        download_full(client, a.out, [e.strip() for e in a.endpoints.split(",")])


if __name__ == "__main__":
    main()
