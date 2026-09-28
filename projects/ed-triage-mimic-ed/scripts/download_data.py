#!/usr/bin/env python
"""Download MIMIC-IV-ED / MIMIC-IV / MIMIC-IV-Note tables for ed_triage.

Credentialed modules read PHYSIONET_USERNAME / PHYSIONET_PASSWORD from the
environment.  ``--sample`` downloads the *open* MIMIC-IV-ED demo and the open
MIMIC-IV demo (100 patients) so the whole pipeline can be smoke-tested without
credentials.

Examples
--------
python scripts/download_data.py --sample --out data
python scripts/download_data.py --module ed --out data
python scripts/download_data.py --module hosp --labevents --out data
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"

MODULES: dict[str, tuple[str, list[str]]] = {
    # module -> (base url relative to PHYSIONET, files)
    "ed": ("mimic-iv-ed/2.2/ed/", ["edstays.csv.gz", "triage.csv.gz", "vitalsign.csv.gz",
                                   "medrecon.csv.gz", "pyxis.csv.gz", "diagnosis.csv.gz"]),
    "hosp": ("mimiciv/3.1/hosp/", ["patients.csv.gz", "admissions.csv.gz", "transfers.csv.gz",
                                   "procedures_icd.csv.gz", "d_labitems.csv.gz", "diagnoses_icd.csv.gz"]),
    "icu": ("mimiciv/3.1/icu/", ["icustays.csv.gz"]),
    "note": ("mimic-iv-note/2.2/note/", ["discharge.csv.gz", "radiology.csv.gz"]),
}
LABEVENTS = "labevents.csv.gz"

DEMO_MODULES: dict[str, tuple[str, list[str]]] = {
    "ed": ("mimic-iv-ed-demo/2.2/ed/", MODULES["ed"][1]),
    "hosp": ("mimic-iv-demo/2.2/hosp/", ["patients.csv.gz", "admissions.csv.gz", "transfers.csv.gz",
                                         "procedures_icd.csv.gz", "d_labitems.csv.gz", "labevents.csv.gz"]),
    "icu": ("mimic-iv-demo/2.2/icu/", ["icustays.csv.gz"]),
}


def auth() -> tuple[str, str] | None:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (u, p) if u and p else None


def fetch(url: str, dest: Path, creds: tuple[str, str] | None) -> bool:
    if requests is None:
        sys.exit("pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=creds, timeout=120) as r:
        if r.status_code in (401, 403):
            print(f"  [{r.status_code}] {url}: credentials rejected or DUA not signed")
            return False
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(dest)
    print(f"  [ok] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def download_module(module: str, out: Path, demo: bool, labevents: bool) -> None:
    table = DEMO_MODULES if demo else MODULES
    if module not in table:
        print(f"  no demo for module {module!r}; skipping")
        return
    rel, files = table[module]
    creds = None if demo else auth()
    if not demo and creds is None:
        print(f"{module}: credentialed module. Complete CITI + DUA for {PHYSIONET.replace('/files', '/content')}/{rel}"
              " and export PHYSIONET_USERNAME / PHYSIONET_PASSWORD.")
        return
    if labevents and module == "hosp" and LABEVENTS not in files:
        files = files + [LABEVENTS]
    print(f"== {'demo ' if demo else ''}{module}: {rel}")
    for fn in files:
        fetch(f"{PHYSIONET}/{rel}{fn}", out / rel / fn, creds)


def verify(out: Path) -> None:
    for demo in (False, True):
        table = DEMO_MODULES if demo else MODULES
        for module, (rel, files) in table.items():
            present = [fn for fn in files if (out / rel / fn).exists()]
            tag = "demo" if demo else "full"
            print(f"{tag:4s} {module:5s} {len(present)}/{len(files)} files under {out / rel}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", choices=list(MODULES), action="append", help="repeatable; default: all")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true", help="open MIMIC-IV(-ED) demo instead of the credentialed release")
    ap.add_argument("--labevents", action="store_true", help="also fetch hosp/labevents.csv.gz (13 GB)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
        return
    modules = args.module or list(MODULES)
    for m in modules:
        download_module(m, args.out, args.sample, args.labevents)


if __name__ == "__main__":
    main()
