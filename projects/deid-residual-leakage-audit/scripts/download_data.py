#!/usr/bin/env python
"""Stage data for deid_audit: the open `deid` gold-standard corpus (no login) and the credentialed
MIMIC-IV-Note / MIMIC-IV / MIMIC-III files (credentials from PHYSIONET_USERNAME / PHYSIONET_PASSWORD).

Examples
--------
python scripts/download_data.py --dataset deid-gold --out data/deid-gold
python scripts/download_data.py --dataset mimic-iv-note --out data/mimic-iv-note/2.2 --sample
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
OPEN_SOURCES = {
    # PhysioNet `deid` package: rule-based de-identifier + gold-standard corpus of nursing notes
    "deid-gold": {"base": f"{PHYSIONET}/deid/1.1/", "files": None},
}
CREDENTIALED_SOURCES = {
    "mimic-iv-note": {
        "base": f"{PHYSIONET}/mimic-iv-note/2.2/",
        "files": ["note/discharge.csv.gz", "note/discharge_detail.csv.gz",
                  "note/radiology.csv.gz", "note/radiology_detail.csv.gz"],
        "sample": ["note/discharge_detail.csv.gz"],
    },
    "mimiciv-hosp": {
        "base": f"{PHYSIONET}/mimiciv/3.1/",
        "files": ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/services.csv.gz"],
        "sample": ["hosp/patients.csv.gz"],
    },
    "mimiciii-notes": {
        "base": f"{PHYSIONET}/mimiciii/1.4/",
        "files": ["NOTEEVENTS.csv.gz", "ADMISSIONS.csv.gz"],
        "sample": ["ADMISSIONS.csv.gz"],
    },
}


def physionet_auth() -> tuple[str, str] | None:
    user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (user, pwd) if user and pwd else None


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None = None) -> bool:
    if requests is None:
        sys.exit("pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
        if r.status_code in (401, 403):
            print(f"  [{r.status_code}] {url}: credentials rejected or DUA not signed")
            return False
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(dest)
    print(f"  [ok] {dest}")
    return True


def list_physionet_dir(url: str) -> list[str]:
    """Parse the file names from a public PhysioNet directory listing (HTML)."""
    if requests is None:
        sys.exit("pip install requests")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    names = re.findall(r'href="([^"?/][^"]*)"', r.text)
    return [n for n in names if not n.startswith(("..", "http")) and n != "SHA256SUMS.txt"]


def download_open(name: str, out: Path) -> None:
    base = OPEN_SOURCES[name]["base"]
    print(f"== {name} -> {out}")
    files = OPEN_SOURCES[name]["files"] or list_physionet_dir(base)
    if not files:
        sys.exit(f"no files listed at {base}; check network or fetch manually from the PhysioNet page")
    for rel in files:
        if rel.endswith("/"):
            for sub in list_physionet_dir(base + rel):
                stream_download(base + rel + sub, out / rel / sub)
        else:
            stream_download(base + rel, out / rel)


def download_credentialed(name: str, out: Path, sample: bool) -> None:
    src = CREDENTIALED_SOURCES[name]
    auth = physionet_auth()
    if auth is None:
        print(f"{name} is credentialed. Complete CITI training + DUA at "
              f"{src['base'].replace('/files/', '/content/')} then export PHYSIONET_USERNAME/PASSWORD.")
        return
    print(f"== {name} -> {out}")
    for rel in (src["sample"] if sample else src["files"]):
        stream_download(src["base"] + rel, out / rel.split("/")[-1], auth)


def verify(root: Path) -> None:
    checks = {
        "mimic-iv-note/2.2": ["discharge.csv.gz", "radiology.csv.gz"],
        "mimiciv/3.1/hosp": ["patients.csv.gz", "admissions.csv.gz"],
        "mimiciii/1.4": ["NOTEEVENTS.csv.gz"],
    }
    for d, files in checks.items():
        for f in files:
            p = root / d / f
            print(f"{d:22s} {f:22s} {'present' if p.exists() else 'MISSING'}")
    gold = root / "deid-gold"
    print(f"deid-gold files: {len(list(gold.rglob('*'))) if gold.exists() else 0}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(OPEN_SOURCES) + list(CREDENTIALED_SOURCES))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true", help="credentialed: only the small metadata tables")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
    elif args.dataset in OPEN_SOURCES:
        download_open(args.dataset, args.out)
    elif args.dataset in CREDENTIALED_SOURCES:
        download_credentialed(args.dataset, args.out, args.sample)
    else:
        ap.error("--dataset or --verify required")


if __name__ == "__main__":
    main()
