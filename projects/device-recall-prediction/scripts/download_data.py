#!/usr/bin/env python
"""Download open inputs for device-recall-prediction.

Modes
-----
--sample        Small API pull for a few device areas (insulin pumps, hip
                prostheses, coronary stents, and any product codes passed with
                --product-codes): classification, 510(k)s, PMAs, recalls +
                enforcement (with class), MAUDE reports with narratives (capped
                by --max-maude per code) and monthly MAUDE counts. A few hundred
                requests; minutes with an API key.
--bulk          Download openFDA bulk JSON partitions for the endpoints in
                --endpoints (default: 510k, recall, enforcement, classification,
                pma). device/event (MAUDE) is ~20 GB; add it explicitly.
--summaries     Download 510(k) summary PDFs for K numbers listed in --k-file
                (one per line) to data/raw/510k_summaries/ (rate limited).
--ai-list       Fetch the FDA AI-enabled medical device list page and try to
                download the CSV/XLSX it links (falls back to instructions).

Environment: OPENFDA_API_KEY (optional; 240 req/min instead of 40).
"""

from __future__ import annotations

import argparse
import json
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

from device_recall.openfda_device import (  # noqa: E402
    CLASSIFICATION,
    ENFORCEMENT,
    K510,
    MAUDE,
    PMA,
    RECALL,
    UDI,
    DeviceClient,
    flatten_510k,
    flatten_classification,
    flatten_enforcement,
    flatten_maude,
    flatten_pma,
    flatten_recall,
    frame,
    join_recall_class,
)
from device_recall.predicate_graph import summary_pdf_url  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")
RAW = ROOT / "data" / "raw"

SAMPLE_QUERIES = {
    "insulin_pump": 'device_name:"insulin"+AND+device_name:"pump"',
    "hip_prosthesis": 'device_name:"hip"+AND+device_name:"prosthesis"',
    "coronary_stent": 'device_name:"stent"+AND+device_name:"coronary"',
}
AI_LIST_PAGE = "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices"


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False) if path.suffix == ".parquet" else df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))


def _to_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Serialise list columns as JSON strings for CSV output."""
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (list, tuple))).any():
            out[c] = out[c].map(lambda v: json.dumps(list(v)) if isinstance(v, (list, tuple)) else v)
    return out


def run_sample(product_codes: List[str], max_maude: int, start: str, end: str) -> None:
    client = DeviceClient()
    out = RAW / "openfda_sample"
    out.mkdir(parents=True, exist_ok=True)
    codes = set(c.upper() for c in product_codes)
    # discover product codes from the classification endpoint
    for label, q in SAMPLE_QUERIES.items():
        cls = frame(client.fetch(CLASSIFICATION, q, limit=50), flatten_classification)
        if not cls.empty:
            log.info("%s -> product codes %s", label, sorted(cls["product_code"].unique())[:10])
            codes.update(cls["product_code"].dropna().unique()[:3])
    codes = sorted(codes)
    log.info("sample product codes: %s", codes)

    cls_all = frame(client.fetch(CLASSIFICATION, "+OR+".join(f"product_code:{c}" for c in codes), limit=100), flatten_classification)
    _save(cls_all, out / "classification.csv")

    k_frames, p_frames, r_frames, e_frames, m_frames, series = [], [], [], [], [], []
    for code in codes:
        q = f"product_code:{code}"
        k = frame(client.fetch(K510, q, limit=1000, max_records=5000), flatten_510k)
        k_frames.append(k)
        p = frame(client.fetch(PMA, q, limit=1000, max_records=2000), flatten_pma)
        p_frames.append(p)
        r = frame(client.fetch(RECALL, q, limit=1000, max_records=5000), flatten_recall)
        r_frames.append(r)
        # enforcement records carry the recall class; fetch them by the recall event ids
        # (device/enforcement has no reliable product_code field to search on)
        e = pd.DataFrame()
        if not r.empty:
            ids = [i for i in r["res_event_number"].dropna().unique()]
            for chunk_start in range(0, len(ids), 50):
                chunk = ids[chunk_start : chunk_start + 50]
                clause = "+OR+".join(f"event_id:{i}" for i in chunk)
                e = pd.concat([e, frame(client.fetch(ENFORCEMENT, clause, limit=1000), flatten_enforcement)], ignore_index=True)
        e_frames.append(e)
        m = frame(
            client.iter_records(MAUDE, f"device.device_report_product_code:{code}+AND+{client.date_range('date_received', start, end)}", limit=100, max_records=max_maude),
            flatten_maude,
        )
        m_frames.append(m)
        s = client.maude_monthly_counts(f"device.device_report_product_code:{code}", start, end)
        s["product_code"] = code
        series.append(s)
        log.info("code %s: 510k=%d pma=%d recalls=%d enforcement=%d maude=%d", code, len(k), len(p), len(r), len(e), len(m))

    k510 = pd.concat(k_frames, ignore_index=True)
    _save(k510, out / "k510.csv")
    _save(pd.concat(p_frames, ignore_index=True), out / "pma.csv")
    recalls = pd.concat(r_frames, ignore_index=True)
    enforcement = pd.concat(e_frames, ignore_index=True)
    _save(_to_csv_safe(recalls), out / "recalls.csv")
    _save(enforcement, out / "enforcement.csv")
    if not recalls.empty and not enforcement.empty:
        _save(_to_csv_safe(join_recall_class(recalls, enforcement)), out / "recalls_with_class.csv")
    maude = pd.concat(m_frames, ignore_index=True)
    _save(_to_csv_safe(maude), out / "maude_sample.csv")
    _save(pd.concat(series, ignore_index=True), out / "maude_monthly_counts.csv")
    # UDI records referencing the sampled 510(k)s (first 200)
    udi_rows = []
    for kn in k510["k_number"].dropna().head(200):
        udi_rows += client.fetch(UDI, f"premarket_submissions.submission_number:{kn}", limit=100, max_records=100)
    if udi_rows:
        from device_recall.openfda_device import flatten_udi

        _save(_to_csv_safe(frame(udi_rows, flatten_udi)), out / "udi_sample.csv")
    log.info("sample complete -> %s", out)


def run_bulk(endpoints: Iterable[str], dest: Path) -> None:
    manifest = requests.get("https://api.fda.gov/download.json", timeout=60).json()["results"]["device"]
    for ep in endpoints:
        parts = manifest.get(ep, {}).get("partitions", [])
        if not parts:
            log.warning("no partitions for device/%s", ep)
            continue
        d = dest / ep
        d.mkdir(parents=True, exist_ok=True)
        for p in parts:
            url = p["file"]
            target = d / "__".join(url.split("/")[-2:])
            if target.exists():
                continue
            log.info("downloading %s (%s MB)", url, p.get("size_mb", "?"))
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            time.sleep(0.3)


def run_summaries(k_file: Path, dest: Path, sleep: float = 1.0) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ks = [l.strip().upper() for l in k_file.read_text().splitlines() if l.strip()]
    ok = 0
    for k in ks:
        target = dest / f"{k}.pdf"
        if target.exists():
            ok += 1
            continue
        url = summary_pdf_url(k)
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "device_recall/0.1 research"})
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                target.write_bytes(r.content)
                ok += 1
            else:
                log.warning("%s: HTTP %s at %s (check https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=%s)", k, r.status_code, url, k)
        except requests.RequestException as exc:
            log.warning("%s: %s", k, exc)
        time.sleep(sleep)
    log.info("summaries available: %d / %d", ok, len(ks))


def run_ai_list(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        html = requests.get(AI_LIST_PAGE, timeout=60, headers={"User-Agent": "device_recall/0.1 research"}).text
    except requests.RequestException as exc:
        log.error("could not fetch %s: %s", AI_LIST_PAGE, exc)
        html = ""
    links = re.findall(r'href="([^"]+\.(?:xlsx|csv))"', html, flags=re.IGNORECASE)
    if not links:
        log.warning("no CSV/XLSX link found; download manually from %s and save as data/raw/fda_ai_enabled_devices.xlsx", AI_LIST_PAGE)
        return
    for href in links[:2]:
        url = href if href.startswith("http") else "https://www.fda.gov" + href
        name = url.rsplit("/", 1)[-1]
        r = requests.get(url, timeout=120)
        if r.ok:
            (dest / name).write_bytes(r.content)
            log.info("saved %s", dest / name)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--product-codes", default="", help="extra product codes, comma separated (e.g. LZG,QAS)")
    ap.add_argument("--max-maude", type=int, default=500)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--bulk", action="store_true")
    ap.add_argument("--endpoints", default="510k,recall,enforcement,classification,pma")
    ap.add_argument("--summaries", action="store_true")
    ap.add_argument("--k-file", default=str(RAW / "k_numbers.txt"))
    ap.add_argument("--ai-list", action="store_true")
    args = ap.parse_args(argv)
    if not any((args.sample, args.bulk, args.summaries, args.ai_list)):
        ap.print_help()
        return 1
    if args.sample:
        run_sample([c for c in args.product_codes.split(",") if c.strip()], args.max_maude, args.start, args.end)
    if args.bulk:
        run_bulk([e.strip() for e in args.endpoints.split(",") if e.strip()], RAW / "bulk")
    if args.summaries:
        run_summaries(Path(args.k_file), RAW / "510k_summaries")
    if args.ai_list:
        run_ai_list(RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
