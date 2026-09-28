#!/usr/bin/env python3
"""Fetch OpenNeuro metadata, MRIQC WebAPI records and git-annex MD5 indexes.

Usage
-----
    python scripts/download_data.py openneuro     [--sample N]
    python scripts/download_data.py participants  [--sample N]
    python scripts/download_data.py mriqc --modality T1w|bold|T2w [--sample N_PAGES] [--field-strength 3]
    python scripts/download_data.py annex-index   [--sample N]
    python scripts/download_data.py link          # join cached IQMs with cached MD5 indexes

Only public resources are used. ``OPENNEURO_API_KEY`` is optional.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"

from mriqc_audit.openneuro_client import OpenNeuroClient, annex_md5_index, link_iqms_to_openneuro  # noqa: E402
from mriqc_audit.mriqc_client import MRIQCClient, dedupe_records, load_jsonl  # noqa: E402


def cmd_openneuro(sample: Optional[int]) -> None:
    out = DATA / "openneuro"
    out.mkdir(parents=True, exist_ok=True)
    client = OpenNeuroClient()
    datasets = client.list_datasets(max_datasets=sample)
    datasets.to_csv(out / "datasets.csv", index=False)
    subj = client.subject_metadata(max_datasets=sample)
    subj.to_csv(out / "subject_metadata.csv", index=False)
    print(f"{len(datasets)} datasets, {len(subj)} participant rows -> {out}")


def cmd_participants(sample: Optional[int]) -> None:
    out = DATA / "openneuro" / "participants"
    out.mkdir(parents=True, exist_ok=True)
    listing = DATA / "openneuro" / "datasets.csv"
    if not listing.exists():
        sys.exit("run `openneuro` first")
    df = pd.read_csv(listing)
    if sample:
        df = df.head(sample)
    client = OpenNeuroClient()
    for _, row in df.iterrows():
        target = out / f"{row['dataset_id']}.tsv"
        if target.exists() or pd.isna(row.get("tag")):
            continue
        try:
            tsv = client.fetch_participants_tsv(row["dataset_id"], row["tag"])
        except Exception as e:  # noqa: BLE001
            print(f"{row['dataset_id']}: {e}", file=sys.stderr)
            continue
        if tsv is not None:
            tsv.to_csv(target, sep="\t", index=False)
            print(f"{row['dataset_id']}: {len(tsv)} participants")


def cmd_mriqc(modality: str, sample_pages: Optional[int], field_strength: Optional[float]) -> None:
    out = DATA / "mriqc_webapi"
    out.mkdir(parents=True, exist_ok=True)
    where = {"bids_meta.MagneticFieldStrength": field_strength} if field_strength else None
    client = MRIQCClient()
    try:
        total = client.count(modality, where)
        print(f"{modality}: {total} records on the server")
    except Exception as e:  # noqa: BLE001
        print(f"count failed ({e}); continuing", file=sys.stderr)
    cache = out / f"{modality}.jsonl"
    df = client.fetch(modality, where=where, max_pages=sample_pages, cache=cache)
    if cache.exists():
        df = load_jsonl(cache, modality)
    df = dedupe_records(df)
    df.to_parquet(out / f"{modality}.parquet", index=False)
    print(f"{len(df)} unique records -> {out / (modality + '.parquet')}")


def cmd_annex_index(sample: Optional[int]) -> None:
    listing = DATA / "openneuro" / "datasets.csv"
    if not listing.exists():
        sys.exit("run `openneuro` first")
    ids = pd.read_csv(listing)["dataset_id"].dropna().tolist()
    if sample:
        ids = ids[:sample]
    git_root = DATA / "openneuro_git"
    git_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for dsid in ids:
        dest = git_root / dsid
        if not dest.exists():
            cmd = ["git", "clone", "--quiet", "--depth", "1", f"https://github.com/OpenNeuroDatasets/{dsid}", str(dest)]
            rc = subprocess.run(cmd, capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"{dsid}: clone failed: {rc.stderr.strip()[:200]}", file=sys.stderr)
                continue
        idx = annex_md5_index(dest)
        idx.insert(0, "dataset_id", dsid)
        frames.append(idx)
        print(f"{dsid}: {len(idx)} annexed images indexed")
    if frames:
        pd.concat(frames).to_parquet(DATA / "openneuro" / "annex_md5_index.parquet", index=False)


def cmd_link() -> None:
    idx_path = DATA / "openneuro" / "annex_md5_index.parquet"
    if not idx_path.exists():
        sys.exit("run `annex-index` first")
    idx = pd.read_parquet(idx_path)
    frames = []
    for modality in ("T1w", "bold", "T2w"):
        p = DATA / "mriqc_webapi" / f"{modality}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit("run `mriqc` first")
    iqms = pd.concat(frames, ignore_index=True)
    linked = []
    for dsid, sub in idx.groupby("dataset_id"):
        linked.append(link_iqms_to_openneuro(iqms, sub.drop(columns="dataset_id"), dsid))
    out = pd.concat(linked, ignore_index=True)
    (DATA / "linked").mkdir(exist_ok=True)
    out.to_parquet(DATA / "linked" / "iqms_linked.parquet", index=False)
    print(f"{len(out)} of {len(iqms)} IQM records linked to {out['dataset_id'].nunique()} datasets")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["openneuro", "participants", "mriqc", "annex-index", "link"])
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--modality", default="T1w", choices=["T1w", "T2w", "bold"])
    p.add_argument("--field-strength", type=float, default=None)
    a = p.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    if a.what == "openneuro":
        cmd_openneuro(a.sample)
    elif a.what == "participants":
        cmd_participants(a.sample)
    elif a.what == "mriqc":
        cmd_mriqc(a.modality, a.sample, a.field_strength)
    elif a.what == "annex-index":
        cmd_annex_index(a.sample)
    else:
        cmd_link()


if __name__ == "__main__":
    main()
