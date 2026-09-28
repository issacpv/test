#!/usr/bin/env python3
"""Discover and download open EEG neurofeedback datasets from OpenNeuro.

Two real APIs are used:

* OpenNeuro GraphQL (``https://openneuro.org/crn/graphql``) to *scan* for EEG
  datasets whose name or README mentions neurofeedback (the candidate list is
  written to ``data/candidates.csv`` for manual screening).
* The public OpenNeuro S3 bucket (``s3://openneuro.org/<dsid>``), read over
  plain HTTPS with the ListObjectsV2 XML API (paginated), so no AWS credentials
  or CLI are needed.  ``--sample`` fetches the metadata files and the first
  subject's EEG only.

Examples
--------
    python scripts/download_data.py --scan
    python scripts/download_data.py --dataset ds002338 --sample
    python scripts/download_data.py --dataset ds002336 ds002338 ds005846 ds005878
    # alternatives (equivalent):  openneuro-py download --dataset ds002338
    #                             aws s3 sync --no-sign-request s3://openneuro.org/ds002338 data/ds002338
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GRAPHQL = "https://openneuro.org/crn/graphql"
S3 = "https://s3.amazonaws.com/openneuro.org"
KNOWN = {
    "ds002336": "XP1: simultaneous EEG-fMRI motor-imagery neurofeedback (Lioi et al., 2020, Sci Data)",
    "ds002338": "XP2: bimodal EEG-fMRI motor-imagery neurofeedback, 1-D vs 2-D feedback (Lioi et al., 2020, Sci Data)",
    "ds005846": "EEG parietal-alpha down-regulation neurofeedback in immersive VR (Front. Neurosci., 2025)",
    "ds005878": "EEG parietal-alpha down-regulation neurofeedback in immersive VR, companion dataset (Front. Neurosci., 2025)",
}


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def graphql(query: str, variables: Optional[dict] = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(GRAPHQL, data=body, headers={"Content-Type": "application/json", "User-Agent": "nf-multiverse"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def scan(keywords=("neurofeedback", "neuro-feedback", "closed-loop", "brain-computer interface")) -> Path:
    """List EEG datasets on OpenNeuro whose name/README matches the keywords (cursor-paginated)."""
    q = """
    query($after: String) {
      datasets(first: 100, after: $after, modality: "EEG", filterBy: {all: true}) {
        pageInfo { hasNextPage endCursor }
        edges { node { id created latestSnapshot { tag description { Name License }
                 readme summary { subjects tasks sessions modalities } } } }
      }
    }"""
    after = None
    rows = []
    while True:
        payload = graphql(q, {"after": after})
        if "errors" in payload:
            sys.exit(f"GraphQL error: {payload['errors'][:1]}")
        data = payload["data"]["datasets"]
        for e in data["edges"]:
            n = e["node"]
            snap = n.get("latestSnapshot") or {}
            name = ((snap.get("description") or {}).get("Name") or "")
            readme = snap.get("readme") or ""
            text = f"{name} {readme}".lower()
            if any(k in text for k in keywords):
                summ = snap.get("summary") or {}
                rows.append({"id": n["id"], "name": name, "n_subjects": len(summ.get("subjects") or []), "sessions": ";".join(summ.get("sessions") or []), "tasks": ";".join(summ.get("tasks") or []), "license": (snap.get("description") or {}).get("License"), "created": n.get("created"), "known": KNOWN.get(n["id"], "")})
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "candidates.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "name", "n_subjects", "sessions", "tasks", "license", "created", "known"])
        w.writeheader()
        w.writerows(rows)
    _log(f"{len(rows)} candidate datasets -> {out.relative_to(ROOT)} (screen manually: EEG-based feedback? online feedback values shared?)")
    return out


def s3_list(prefix: str) -> Iterator[Dict[str, object]]:
    """Iterate over keys in the public OpenNeuro bucket under ``prefix`` (ListObjectsV2, paginated)."""
    token = None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"{S3}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            tree = ET.fromstring(resp.read())
        for c in tree.findall(f"{ns}Contents"):
            yield {"key": c.find(f"{ns}Key").text, "size": int(c.find(f"{ns}Size").text)}
        if tree.find(f"{ns}IsTruncated").text != "true":
            break
        token = tree.find(f"{ns}NextContinuationToken").text


def download_key(key: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(f"{S3}/{urllib.parse.quote(key)}", timeout=120) as resp, open(dest, "wb") as fh:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)


def download_dataset(dsid: str, sample: bool) -> None:
    prefix = f"{dsid}/"
    keys = list(s3_list(prefix))
    if not keys:
        sys.exit(f"no objects under s3://openneuro.org/{prefix}; check the accession number")
    total = sum(k["size"] for k in keys) / 1e9
    _log(f"{dsid}: {len(keys)} files, {total:.2f} GB  ({KNOWN.get(dsid, '')})")
    if sample:
        top = [k for k in keys if "/" not in k["key"][len(prefix):]]  # dataset-level metadata
        subs = sorted({k["key"][len(prefix):].split("/")[0] for k in keys if k["key"][len(prefix):].startswith("sub-")})
        first = [k for k in keys if subs and k["key"].startswith(f"{prefix}{subs[0]}/") and ("eeg" in k["key"] or "beh" in k["key"] or k["key"].endswith((".tsv", ".json")))]
        keys = top + first
        _log(f"--sample: {len(keys)} files (metadata + {subs[0] if subs else 'no subject'})")
    for i, k in enumerate(keys, 1):
        rel = k["key"][len(prefix):]
        download_key(k["key"], DATA / dsid / rel)
        if i % 25 == 0 or i == len(keys):
            _log(f"  {i}/{len(keys)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="list candidate neurofeedback EEG datasets via the OpenNeuro GraphQL API")
    ap.add_argument("--dataset", nargs="*", default=[], help="OpenNeuro accession(s), e.g. ds002338")
    ap.add_argument("--sample", action="store_true", help="metadata + first subject only")
    args = ap.parse_args()
    if args.scan:
        scan()
    for ds in args.dataset:
        download_dataset(ds, args.sample)
    if not (args.scan or args.dataset):
        ap.print_help()


if __name__ == "__main__":
    main()
