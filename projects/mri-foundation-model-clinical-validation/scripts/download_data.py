#!/usr/bin/env python
"""Downloader for the open parts of mri-foundation-model-clinical-validation.

- ``--mrart``      OpenNeuro ds004173 via anonymous S3 (boto3) with a --sample mode.
- ``--isles22``    Zenodo record 7153326 (ISLES 2022) via the Zenodo REST API.
- ``--mindboggle`` MindBoggle-101 archives via the OSF API (node nhtur).
- ``--synthetic``  offline phantom volumes for tests.
DUA datasets (ATLAS, BraTS, OASIS, CANDI, Hammers) are documented in data/README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MRART = "ds004173"


def _stream(url: str, dest: Path, params: dict | None = None) -> None:
    import requests

    if dest.exists():
        print("exists", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print("downloaded", dest)


def download_mrart(sample: bool) -> None:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install boto3") from exc
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    out = DATA / "mrart"
    subjects_seen: set[str] = set()
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket="openneuro.org", Prefix=f"{MRART}/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(MRART) + 1:]
            top = "/" not in rel
            if not (top or rel.endswith(("_T1w.nii.gz", "_T1w.json"))):
                continue
            if rel.startswith("sub-"):
                subj = rel.split("/")[0]
                if sample and subj not in subjects_seen and len(subjects_seen) >= 3:
                    continue
                subjects_seen.add(subj)
            dest = out / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file("openneuro.org", obj["Key"], str(dest))
            print("downloaded", dest)


def download_isles22(sample: bool) -> None:
    import requests

    rec = requests.get("https://zenodo.org/api/records/7153326", timeout=120)
    rec.raise_for_status()
    files = rec.json().get("files", [])
    out = DATA / "isles22"
    for f in files:
        url = f["links"]["self"]
        _stream(url, out / f["key"])
        if sample:
            break


def download_mindboggle(sample: bool) -> None:
    import requests

    url = "https://api.osf.io/v2/nodes/nhtur/files/osfstorage/"
    out = DATA / "mindboggle101"
    while url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        js = r.json()
        for item in js.get("data", []):
            attrs = item["attributes"]
            if attrs.get("kind") != "file":
                continue
            name = attrs["name"]
            if not name.lower().endswith((".tar.gz", ".tgz", ".zip")):
                continue
            _stream(item["links"]["download"], out / name)
            if sample:
                return
        url = (js.get("links") or {}).get("next")


def synthetic() -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from fm_clinical_val.degradations import make_phantom

    out = DATA / "sample"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        img, lab = make_phantom(size=48, seed=i)
        np.save(out / f"phantom{i}_img.npy", img)
        np.save(out / f"phantom{i}_lab.npy", lab)
    print("wrote 3 phantoms to", out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mrart", action="store_true")
    ap.add_argument("--isles22", action="store_true")
    ap.add_argument("--mindboggle", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args(argv)
    if a.synthetic:
        synthetic()
    elif a.mrart:
        download_mrart(a.sample)
    elif a.isles22:
        download_isles22(a.sample)
    elif a.mindboggle:
        download_mindboggle(a.sample)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
