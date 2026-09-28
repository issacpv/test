#!/usr/bin/env python3
"""Download openFDA device data (MAUDE, 510k, PMA, classification, recall) and, optionally, the FDA AI list.

Everything is open. ``OPENFDA_API_KEY`` (optional) is read from the environment.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --sample --product-codes QAS,QFM
    python scripts/download_data.py --full
    python scripts/download_data.py --full --endpoints event,recall
    python scripts/download_data.py --ai-list-url "https://www.fda.gov/media/<id>/download?attachment"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_device_taxonomy.openfda_client import OpenFDAClient  # noqa: E402

DEFAULT_CODES = ["QAS", "QFM", "POK", "LLZ"]  # product codes that contain AI-enabled devices (verify against the FDA list)
PAGED_ENDPOINTS = {"classification": "device/classification", "510k": "device/510k", "pma": "device/pma", "recall": "device/recall"}


def _write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


def download_ai_list(url: str, out: Path) -> None:
    dest = out / "fda_ai_devices.xlsx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"saved {dest} ({len(r.content)/1e6:.1f} MB)")


def download_sample(client: OpenFDAClient, out: Path, codes: list[str], max_per_code: int) -> None:
    base = out / "openfda" / "sample"
    for code in codes:
        recs = client.fetch_records("device/event", f"device.device_report_product_code:{code}", max_records=max_per_code)
        n = _write_jsonl(base / f"event_{code}.jsonl", recs)
        print(f"{code}: {n} MAUDE reports")
        cls = client.fetch_records("device/classification", f"product_code:{code}", max_records=50)
        _write_jsonl(base / f"classification_{code}.jsonl", cls)
    ai_list = out / "fda_ai_devices.xlsx"
    if ai_list.exists():
        from ai_device_taxonomy.linkage import load_ai_device_list  # noqa: WPS433

        devices = load_ai_device_list(ai_list)
        ks = [s for s in devices["submission_number"].astype(str) if s.upper().startswith("K")][:50]
        recs = []
        for k in ks:
            recs += client.fetch_records("device/510k", f"k_number:{k}", max_records=5)
        print(f"510k records for {len(ks)} submissions: {len(recs)}")
        _write_jsonl(base / "k510_ai_sample.jsonl", recs)
    else:
        print("AI list not found at data/raw/fda_ai_devices.xlsx (manual download, see data/README.md)")


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
    for name, ep in PAGED_ENDPOINTS.items():
        if name not in endpoints:
            continue
        n = _write_jsonl(base / f"{name}.jsonl", client.iter_records(ep, search=None))
        print(f"{name}: {n} records")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--ai-list-url", type=str, default=None)
    p.add_argument("--product-codes", type=str, default=",".join(DEFAULT_CODES))
    p.add_argument("--max-per-code", type=int, default=300)
    p.add_argument("--endpoints", type=str, default="event,classification,510k,pma,recall")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args(argv)
    if not (a.sample or a.full or a.ai_list_url):
        p.error("choose --sample, --full and/or --ai-list-url")
    client = OpenFDAClient()
    if a.ai_list_url:
        download_ai_list(a.ai_list_url, a.out)
    if a.sample:
        download_sample(client, a.out, [c.strip().upper() for c in a.product_codes.split(",") if c.strip()], a.max_per_code)
    if a.full:
        download_full(client, a.out, [e.strip() for e in a.endpoints.split(",")])


if __name__ == "__main__":
    main()
