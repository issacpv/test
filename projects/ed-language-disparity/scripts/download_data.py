#!/usr/bin/env python
"""Downloader for ed-language-disparity.

Open demos (MIMIC-IV-ED demo, MIMIC-IV demo) are fetched without credentials.
Credentialed resources (MIMIC-IV-ED, MIMIC-IV hosp/icu, MIMIC-IV-Note) use wget
with PHYSIONET_USERNAME / PHYSIONET_PASSWORD from the environment.

Examples
--------
python scripts/download_data.py --sample
python scripts/download_data.py --ed --hosp --note      # credentialed
python scripts/download_data.py --language-audit        # value counts of admissions.language
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

ED_TABLES = ["edstays", "triage", "vitalsign", "pyxis", "medrecon", "diagnosis"]
HOSP_TABLES = ["admissions", "patients", "transfers"]
ICU_TABLES = ["icustays"]


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


def _fetch_set(base: str, sub: str, tables: list[str], dest: Path, credentialed: bool) -> None:
    for t in tables:
        rc = _wget(f"{PN}/{base}/{sub}/{t}.csv.gz", dest / sub, credentialed)
        if rc != 0:
            print(f"failed: {base}/{sub}/{t}", file=sys.stderr)
            return


def fetch_sample() -> None:
    _fetch_set("mimic-iv-ed-demo/2.2", "ed", ED_TABLES, DATA / "mimic-iv-ed-demo", credentialed=False)
    _fetch_set("mimic-iv-demo/2.2", "hosp", HOSP_TABLES, DATA / "mimic-iv-demo", credentialed=False)
    _fetch_set("mimic-iv-demo/2.2", "icu", ICU_TABLES, DATA / "mimic-iv-demo", credentialed=False)


def language_audit(root: Path) -> None:
    import pandas as pd

    p = root / "hosp" / "admissions.csv.gz"
    if not p.exists():
        print(f"missing {p}")
        return
    adm = pd.read_csv(p, usecols=["subject_id", "language"])
    print(adm["language"].value_counts(dropna=False).to_string())
    per_subj = adm.groupby("subject_id")["language"].nunique()
    print(f"subjects with >1 distinct language across admissions: {(per_subj > 1).sum()} / {len(per_subj)}")


def check() -> None:
    parts = {
        "ED edstays": DATA / "mimic-iv-ed" / "ed" / "edstays.csv.gz",
        "hosp admissions": DATA / "mimiciv" / "hosp" / "admissions.csv.gz",
        "icu icustays": DATA / "mimiciv" / "icu" / "icustays.csv.gz",
        "note discharge": DATA / "mimic-iv-note" / "note" / "discharge.csv.gz",
        "ED demo edstays": DATA / "mimic-iv-ed-demo" / "ed" / "edstays.csv.gz",
        "hosp demo admissions": DATA / "mimic-iv-demo" / "hosp" / "admissions.csv.gz",
    }
    for k, p in parts.items():
        print(f"{'OK ' if p.exists() else '-- '} {k}: {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--ed", action="store_true")
    ap.add_argument("--hosp", action="store_true")
    ap.add_argument("--note", action="store_true")
    ap.add_argument("--language-audit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.sample:
        fetch_sample()
    if a.ed:
        _fetch_set("mimic-iv-ed/2.2", "ed", ED_TABLES, DATA / "mimic-iv-ed", credentialed=True)
    if a.hosp:
        _fetch_set("mimiciv/3.1", "hosp", HOSP_TABLES, DATA / "mimiciv", credentialed=True)
        _fetch_set("mimiciv/3.1", "icu", ICU_TABLES, DATA / "mimiciv", credentialed=True)
    if a.note:
        _fetch_set("mimic-iv-note/2.2", "note", ["discharge"], DATA / "mimic-iv-note", credentialed=True)
    if a.language_audit:
        root = DATA / "mimiciv" if (DATA / "mimiciv" / "hosp" / "admissions.csv.gz").exists() else DATA / "mimic-iv-demo"
        language_audit(root)
    if a.check or not any([a.sample, a.ed, a.hosp, a.note, a.language_audit]):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
