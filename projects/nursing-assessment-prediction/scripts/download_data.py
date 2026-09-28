#!/usr/bin/env python
"""Downloader for nursing-assessment-prediction.

Credentialed MIMIC-IV tables use wget with PHYSIONET_USERNAME /
PHYSIONET_PASSWORD; the open MIMIC-IV demo needs no credentials.
--resolve-items runs the item-label resolver on whichever d_items is present.

Examples
--------
python scripts/download_data.py --sample
python scripts/download_data.py --resolve-items
python scripts/download_data.py --mimiciv          # credentialed, chartevents is ~3 GB
python scripts/download_data.py --check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PN = "https://physionet.org/files"
TABLES = ["icu/d_items", "icu/icustays", "icu/chartevents", "hosp/diagnoses_icd", "hosp/admissions", "hosp/patients"]


def _wget(url: str, dest_dir: Path, credentialed: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-N", "-c", "-q", "--show-progress", "-P", str(dest_dir)]
    if credentialed:
        user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
        if not user or not pwd:
            print("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD for credentialed resources.", file=sys.stderr)
            return 2
        cmd += ["--user", user, "--password", pwd]
    cmd.append(url)
    return subprocess.call(cmd)


def fetch(base: str, dest: Path, credentialed: bool) -> None:
    for t in TABLES:
        sub, name = t.split("/")
        if _wget(f"{PN}/{base}/{sub}/{name}.csv.gz", dest / sub, credentialed) != 0:
            print(f"failed: {base}/{t}", file=sys.stderr)
            return


def resolve_items() -> None:
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from nursing_risk.items import EXPECTED_BRADEN_ITEMIDS, resolve_items as _resolve, verify_expected

    for root in (DATA / "mimiciv", DATA / "mimic-iv-demo"):
        p = root / "icu" / "d_items.csv.gz"
        if not p.exists():
            continue
        d_items = pd.read_csv(p)
        res = _resolve(d_items)
        print(f"{p}:")
        print(res.sort_values(["concept", "itemid"]).to_string(index=False))
        problems = verify_expected(res, EXPECTED_BRADEN_ITEMIDS)
        print("verification:", "OK" if not problems else problems)
        return
    print("no d_items found; run --sample or --mimiciv first")


def check() -> None:
    for root in ("mimiciv", "mimic-iv-demo"):
        for t in TABLES:
            p = DATA / root / f"{t}.csv.gz"
            print(f"{'OK ' if p.exists() else '-- '} {root}/{t}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--mimiciv", action="store_true")
    ap.add_argument("--resolve-items", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.sample:
        fetch("mimic-iv-demo/2.2", DATA / "mimic-iv-demo", credentialed=False)
    if a.mimiciv:
        fetch("mimiciv/3.1", DATA / "mimiciv", credentialed=True)
    if a.resolve_items:
        resolve_items()
    if a.check or not any([a.sample, a.mimiciv, a.resolve_items]):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
