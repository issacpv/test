#!/usr/bin/env python3
"""Data helpers: OpenNeuro multi-shell discovery (open), HCP diffusion via S3 (registration),
IXI DTI (open), and instructions for HCP-Aging/-Development (NDA) and Cam-CAN (DUA).

Examples
--------
Discover OpenNeuro datasets whose bval files contain >= 2 non-zero shells:
    python scripts/download_data.py openneuro-discover --out data/openneuro --max-datasets 200

HCP-YA gradient tables (bvals/bvecs) for 3 subjects (needs boto3 + S3 keys in env):
    python scripts/download_data.py hcp --out data/hcp --sample 3 --bvals-only

IXI gradient tables only:
    python scripts/download_data.py ixi --out data/ixi --tables-only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

GRAPHQL = "https://openneuro.org/crn/graphql"
S3_HTTP = "https://s3.amazonaws.com/openneuro.org"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ---------------------------------------------------------------------------
# OpenNeuro discovery
# ---------------------------------------------------------------------------
def _graphql(query: str, variables: Optional[dict] = None) -> dict:
    r = requests.post(GRAPHQL, json={"query": query, "variables": variables or {}}, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "errors" in js:
        raise RuntimeError(js["errors"])
    return js["data"]


def list_dwi_datasets(max_datasets: int) -> List[Dict[str, str]]:
    """Page through OpenNeuro datasets and keep those whose summary mentions diffusion data."""
    query = """
    query($after: String) {
      datasets(first: 100, after: $after, orderBy: {created: descending}) {
        edges { node { id name latestSnapshot { tag summary { modalities secondaryModalities subjects } } } }
        pageInfo { hasNextPage endCursor }
      }
    }"""
    out, after = [], None
    while True:
        data = _graphql(query, {"after": after})["datasets"]
        for e in data["edges"]:
            node = e["node"]
            snap = node.get("latestSnapshot") or {}
            summ = snap.get("summary") or {}
            mods = [m.lower() for m in (summ.get("modalities") or []) + (summ.get("secondaryModalities") or [])]
            if any("diffusion" in m or "dwi" in m for m in mods):
                out.append({"id": node["id"], "name": node.get("name", ""), "tag": snap.get("tag", ""),
                            "n_subjects": len(summ.get("subjects") or [])})
        if not data["pageInfo"]["hasNextPage"] or len(out) >= max_datasets:
            break
        after = data["pageInfo"]["endCursor"]
    return out[:max_datasets]


def s3_list(prefix: str, max_keys: int = 1000) -> List[str]:
    """Anonymous listing of the public OpenNeuro bucket via the HTTP ListObjectsV2 API."""
    r = requests.get(S3_HTTP, params={"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}, timeout=120)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    return [c.find(S3_NS + "Key").text for c in root.findall(S3_NS + "Contents")]


def first_bval(dsid: str) -> Optional[str]:
    for key in s3_list(f"{dsid}/sub-", max_keys=2000):
        if key.endswith(".bval"):
            return key
    # some datasets keep a shared dwi.bval at the root
    for key in s3_list(f"{dsid}/dwi.bval"):
        if key.endswith(".bval"):
            return key
    return None


def shells_from_bval_text(text: str, tol: float = 100.0) -> List[int]:
    vals = [float(v) for v in text.split()]
    shells: List[float] = []
    for v in vals:
        if v < tol:
            continue
        if not any(abs(v - s) <= tol for s in shells):
            shells.append(v)
    return sorted(int(round(s / 50.0) * 50) for s in shells)


def openneuro_discover(out: Path, max_datasets: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in list_dwi_datasets(max_datasets):
        try:
            key = first_bval(d["id"])
            if key is None:
                d.update(bval="", shells="", multishell=False)
            else:
                txt = requests.get(f"{S3_HTTP}/{key}", timeout=60).text
                sh = shells_from_bval_text(txt)
                d.update(bval=key, shells=";".join(map(str, sh)), multishell=len(sh) >= 2)
        except Exception as e:  # noqa: BLE001
            d.update(bval="", shells=f"error: {type(e).__name__}", multishell=False)
        print(f"{d['id']:<10} n={d['n_subjects']:<4} shells={d['shells']}")
        rows.append(d)
    path = out / "multishell_candidates.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "tag", "n_subjects", "bval", "shells", "multishell"])
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['multishell'] for r in rows)} multi-shell datasets -> {path}")


# ---------------------------------------------------------------------------
# HCP via S3
# ---------------------------------------------------------------------------
HCP_DIFF_FILES = ("bvals", "bvecs", "data.nii.gz", "nodif_brain_mask.nii.gz")


def hcp(out: Path, sample: Optional[int], bvals_only: bool, retest: bool) -> None:
    key = os.environ.get("HCP_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("HCP_AWS_SECRET_ACCESS_KEY")
    if not (key and secret):
        sys.exit("Set HCP_AWS_ACCESS_KEY_ID / HCP_AWS_SECRET_ACCESS_KEY (enable S3 access in ConnectomeDB).")
    try:
        import boto3
    except ImportError:
        sys.exit("pip install boto3")
    s3 = boto3.client("s3", aws_access_key_id=key, aws_secret_access_key=secret)
    bucket, root = "hcp-openaccess", ("HCP_Retest" if retest else "HCP_1200")
    subjects: List[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{root}/", Delimiter="/"):
        subjects += [cp["Prefix"].split("/")[1] for cp in page.get("CommonPrefixes", [])]
        if sample and len(subjects) >= sample:
            break
    subjects = subjects[:sample] if sample else subjects
    files = HCP_DIFF_FILES[:2] if bvals_only else HCP_DIFF_FILES
    for sub in subjects:
        for fname in files:
            dest = out / sub / "Diffusion" / fname
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, f"{root}/{sub}/T1w/Diffusion/{fname}", str(dest))
                print(f"ok {sub} {fname}")
            except Exception as e:  # noqa: BLE001
                print(f"missing {sub} {fname}: {type(e).__name__}")


# ---------------------------------------------------------------------------
# IXI
# ---------------------------------------------------------------------------
IXI_BASE = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"


def ixi(out: Path, tables_only: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("bvals.txt", "bvecs.txt", "IXI.xls"):
        dest = out / name
        if dest.exists():
            continue
        r = requests.get(f"{IXI_BASE}/{name}", timeout=120)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            print(f"ok {name}")
        else:
            print(f"{name}: HTTP {r.status_code}; check https://brain-development.org/ixi-dataset/")
    if tables_only:
        return
    dest = out / "IXI-DTI.tar"
    if not dest.exists():
        with requests.get(f"{IXI_BASE}/IXI-DTI.tar", stream=True, timeout=600) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"ok IXI-DTI.tar ({dest.stat().st_size / 1e9:.1f} GB); extract with: tar -xf {dest}")


def instructions() -> None:
    print("HCP-Aging / HCP-Development: NDA Data Use Certification at https://nda.nih.gov/ccf ; use nda-tools downloadcmd.")
    print("Cam-CAN: https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/ (DUA); cc700 dwi/ per subject.")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("openneuro-discover")
    a.add_argument("--out", type=Path, default=Path("data/openneuro"))
    a.add_argument("--max-datasets", type=int, default=100)
    h = sub.add_parser("hcp")
    h.add_argument("--out", type=Path, default=Path("data/hcp"))
    h.add_argument("--sample", type=int, default=None)
    h.add_argument("--bvals-only", action="store_true")
    h.add_argument("--retest", action="store_true")
    i = sub.add_parser("ixi")
    i.add_argument("--out", type=Path, default=Path("data/ixi"))
    i.add_argument("--tables-only", action="store_true")
    sub.add_parser("instructions")
    args = p.parse_args(argv)
    if args.cmd == "openneuro-discover":
        openneuro_discover(args.out, args.max_datasets)
    elif args.cmd == "hcp":
        hcp(args.out, args.sample, args.bvals_only, args.retest)
    elif args.cmd == "ixi":
        ixi(args.out, args.tables_only)
    else:
        instructions()


if __name__ == "__main__":
    main()
