#!/usr/bin/env python
"""Downloader for sex-stratified-brain-age.

Open parts (IXI, INDI SALD/DLBS) are downloaded directly; HCP-YA via S3 with the
user's ConnectomeDB-linked AWS keys; OASIS-3 via XNAT with NITRC credentials.
Everything reads credentials from environment variables. ``--sample`` modes keep
downloads small; ``--sample-synthetic`` needs no network at all.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IXI_URLS = {
    "IXI-T1.tar": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    "IXI.xls": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls",
}
INDI_BUCKET = "fcp-indi"
INDI_PREFIX = {"SALD": "data/Projects/SALD/", "DLBS": "data/Projects/DLBS/"}


def _stream(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("exists", dest)
        return
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print("downloaded", dest)


def download_ixi(sample: bool) -> None:
    out = DATA / "ixi"
    _stream(IXI_URLS["IXI.xls"], out / "IXI.xls")
    if sample:
        print("sample mode: demographics only (T1 archive is ~4.5 GB; rerun without --sample)")
        return
    _stream(IXI_URLS["IXI-T1.tar"], out / "IXI-T1.tar")


def download_indi(project: str, sample: bool) -> None:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install boto3") from exc
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = INDI_PREFIX[project]
    out = DATA / project.lower()
    out.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=INDI_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            keep = key.endswith((".csv", ".tsv", ".txt")) or ("anat" in key and "T1" in key)
            if not keep:
                continue
            dest = out / key[len(prefix):]
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(INDI_BUCKET, key, str(dest))
            print("downloaded", dest)
            n += 1
            if sample and n >= 5:
                return


def download_hcp_ya(subjects_file: Path, retest: bool) -> None:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install boto3") from exc
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise SystemExit("Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from ConnectomeDB (Amazon S3 access).")
    s3 = boto3.client("s3")
    bucket = "hcp-openaccess"
    root = "HCP_Retest" if retest else "HCP_1200"
    out = DATA / ("hcp_ya_retest" if retest else "hcp_ya")
    for subj in [ln.strip() for ln in subjects_file.read_text().splitlines() if ln.strip()]:
        prefix = f"{root}/{subj}/T1w/{subj}/stats/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            name = obj["Key"].split("/")[-1]
            if name in {"aseg.stats", "lh.aparc.stats", "rh.aparc.stats"}:
                dest = out / subj / "stats" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(dest))
                print("downloaded", dest)


def download_oasis3(subjects_file: Path | None) -> None:
    import requests

    base = os.environ.get("XNAT_BASE", "https://central.xnat.org")
    user, pw = os.environ.get("XNAT_USER"), os.environ.get("XNAT_PASS")
    if not user or not pw:
        raise SystemExit("Set XNAT_USER / XNAT_PASS (NITRC credentials with OASIS-3 DUA).")
    s = requests.Session()
    s.auth = (user, pw)
    out = DATA / "oasis3"
    out.mkdir(parents=True, exist_ok=True)
    r = s.get(f"{base}/data/projects/OASIS3/resources", params={"format": "json"}, timeout=120)
    r.raise_for_status()
    for row in r.json()["ResultSet"]["Result"]:
        rr = s.get(f"{base}/data/projects/OASIS3/resources/{row['label']}/files", params={"format": "json"}, timeout=120)
        for f in rr.json()["ResultSet"]["Result"]:
            if f["Name"].lower().endswith(".csv"):
                dest = out / "tables" / f["Name"]
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(s.get(f"{base}{f['URI']}", timeout=600).content)
                print("downloaded", dest)
    if subjects_file is None:
        print("tables done; pass --subjects to fetch FreeSurfer stats per MR session")
        return
    for subj in [ln.strip() for ln in subjects_file.read_text().splitlines() if ln.strip()]:
        r = s.get(f"{base}/data/projects/OASIS3/subjects/{subj}/experiments",
                  params={"format": "json", "xsiType": "xnat:mrSessionData"}, timeout=120)
        for exp in [x["label"] for x in r.json()["ResultSet"]["Result"]]:
            ra = s.get(f"{base}/data/experiments/{exp}/assessors", params={"format": "json"}, timeout=120)
            for a in ra.json()["ResultSet"]["Result"]:
                rf = s.get(f"{base}/data/experiments/{exp}/assessors/{a['ID']}/files", params={"format": "json"}, timeout=120)
                if not rf.ok:
                    continue
                for f in rf.json()["ResultSet"]["Result"]:
                    if f["Name"] in {"aseg.stats", "lh.aparc.stats", "rh.aparc.stats"}:
                        dest = out / "freesurfer" / exp / "stats" / f["Name"]
                        if dest.exists():
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(s.get(f"{base}{f['URI']}", timeout=600).content)
                        print("downloaded", dest)


def sample_synthetic() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from sexstrat_brainage.simulation import simulate_lifespan_cohort

    out = DATA / "sample"
    out.mkdir(parents=True, exist_ok=True)
    df = simulate_lifespan_cohort(n=600, seed=0)
    df.to_csv(out / "synthetic_freesurfer_table.csv", index=False)
    print("wrote", out / "synthetic_freesurfer_table.csv", df.shape)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ixi", action="store_true")
    ap.add_argument("--indi", choices=sorted(INDI_PREFIX))
    ap.add_argument("--hcp-ya", action="store_true")
    ap.add_argument("--retest", action="store_true")
    ap.add_argument("--oasis3", action="store_true")
    ap.add_argument("--subjects", type=Path)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--sample-synthetic", action="store_true")
    a = ap.parse_args(argv)
    if a.sample_synthetic:
        sample_synthetic()
    elif a.ixi:
        download_ixi(a.sample)
    elif a.indi:
        download_indi(a.indi, a.sample)
    elif a.hcp_ya:
        if not a.subjects:
            raise SystemExit("--subjects file required")
        download_hcp_ya(a.subjects, a.retest)
    elif a.oasis3:
        download_oasis3(a.subjects)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
