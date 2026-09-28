#!/usr/bin/env python
"""Access helper for the OAI progression project.

OAI is distributed by the NIMH Data Archive (free registration + OAI data
access terms); there is no anonymous download. This script (a) checks that
nda-tools and credentials are in place and prints the expected file list,
(b) generates a synthetic OAI-like sample for offline development.

    python scripts/download_data.py --check-nda
    python scripts/download_data.py --sample
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"

EXPECTED_CLINICAL = [
    "Enrollees.txt",
    "AllClinical00.txt", "AllClinical01.txt", "AllClinical03.txt", "AllClinical05.txt",
    "AllClinical06.txt", "AllClinical08.txt", "AllClinical10.txt",
    "Outcomes99.txt",
]
EXPECTED_XRAY = ["kXR_SQ_BU00.txt", "kXR_SQ_BU01.txt", "kXR_SQ_BU03.txt", "kXR_SQ_BU05.txt",
                 "kXR_SQ_BU06.txt", "kXR_SQ_BU08.txt", "kXR_SQ_BU10.txt"]


def cmd_check_nda() -> None:
    print("OAI access: https://nda.nih.gov/oai (RAS login + NDA account + OAI data access terms)")
    print("nda-tools (downloadcmd):", "found" if shutil.which("downloadcmd") else "missing -> pip install nda-tools")
    print("NDA_USERNAME:", "set" if os.environ.get("NDA_USERNAME") else "not set")
    print("\nExpected clinical files in data/oai/clinical/:")
    for f in EXPECTED_CLINICAL:
        print("  ", f, "(present)" if (DATA / "oai" / "clinical" / f).exists() else "")
    print("Expected x-ray reading files in data/oai/xray_readings/:")
    for f in EXPECTED_XRAY:
        print("  ", f, "(present)" if (DATA / "oai" / "xray_readings" / f).exists() else "")
    print("\nImage packages: create a package on NDA, then run")
    print('  downloadcmd -dp <packageID> -d data/oai/images -u "$NDA_USERNAME"')


def cmd_sample(n_participants: int, seed: int) -> None:
    from oai_prog.cohort import simulate_oai_like

    readings, tkr, followup = simulate_oai_like(n_participants=n_participants, seed=seed)
    out = DATA / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    readings.to_csv(out / "kl_readings.csv", index=False)
    tkr.to_csv(out / "tkr.csv", index=False)
    followup.to_csv(out / "followup.csv", index=False)
    print(f"wrote synthetic sample for {n_participants} participants to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-nda", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.check_nda:
        cmd_check_nda()
    if args.sample:
        cmd_sample(args.n, args.seed)
    if not (args.check_nda or args.sample):
        ap.print_help()


if __name__ == "__main__":
    main()
