#!/usr/bin/env python
"""Downloader for cuffless-bp-pregnancy.

Open parts (MIMIC-III waveform matched subset index, one sample record, VitalDB
case index and cases) are fetched directly.  Credentialed PhysioNet parts
(MIMIC-III clinical, MIMIC-IV hosp, MIMIC-IV waveform) are fetched with wget
using PHYSIONET_USERNAME / PHYSIONET_PASSWORD from the environment.  Nothing
is downloaded without an explicit flag.

Examples
--------
python scripts/download_data.py --mimic3-index
python scripts/download_data.py --sample
python scripts/download_data.py --vitaldb-index
python scripts/download_data.py --vitaldb-cases 10
python scripts/download_data.py --mimic3-clinical     # needs credentials
python scripts/download_data.py --mimic4              # needs credentials
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

PHYSIONET_FILES = "https://physionet.org/files"
MIMIC3_WDB = "mimic3wdb-matched/1.0"
MIMIC3_CLIN = "mimiciii/1.4"
MIMIC4_HOSP = "mimiciv/3.1/hosp"
MIMIC4_WDB = "mimic4wdb/0.1.0"
VITALDB_API = "https://api.vitaldb.net"

MIMIC3_TABLES = ["DIAGNOSES_ICD", "PATIENTS", "ADMISSIONS", "ICUSTAYS", "INPUTEVENTS_MV", "D_ITEMS"]
MIMIC4_TABLES = ["diagnoses_icd", "patients", "admissions", "transfers"]


def _wget(url: str, dest_dir: Path, credentialed: bool) -> int:
    """Run wget for one URL; returns the exit code."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-N", "-c", "-q", "--show-progress", "-P", str(dest_dir)]
    if credentialed:
        user = os.environ.get("PHYSIONET_USERNAME")
        pwd = os.environ.get("PHYSIONET_PASSWORD")
        if not user or not pwd:
            print("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (credentialed resource).", file=sys.stderr)
            return 2
        cmd += ["--user", user, "--password", pwd]
    cmd.append(url)
    return subprocess.call(cmd)


def fetch_mimic3_index() -> None:
    dest = DATA / "mimic3wdb-matched"
    for name in ["RECORDS", "RECORDS-waveforms", "RECORDS-numerics"]:
        _wget(f"{PHYSIONET_FILES}/{MIMIC3_WDB}/{name}", dest, credentialed=False)
    print(f"Index files written to {dest}")


def fetch_sample_record() -> None:
    """Read 60 s of one matched-subset record through the wfdb PhysioNet reader."""
    try:
        import wfdb
    except ImportError:
        print("pip install wfdb", file=sys.stderr)
        return
    # A record from the matched subset with ECG, PLETH and ABP channels (first patient folder).
    record = "p000020-2183-04-28-17-47"
    pn_dir = f"{MIMIC3_WDB}/p00/p000020"
    rec = wfdb.rdrecord(record, pn_dir=pn_dir, sampto=125 * 60)
    print(f"Record {record}: fs={rec.fs}, signals={rec.sig_name}, samples={rec.sig_len}")
    out = DATA / "mimic3wdb-matched" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(out / f"{record}.npz", p_signal=rec.p_signal, sig_name=np.array(rec.sig_name), fs=rec.fs)
    print(f"Saved 60 s sample to {out}")


def fetch_mimic3_clinical() -> None:
    dest = DATA / "mimiciii"
    for t in MIMIC3_TABLES:
        rc = _wget(f"{PHYSIONET_FILES}/{MIMIC3_CLIN}/{t}.csv.gz", dest, credentialed=True)
        if rc != 0:
            print(f"failed: {t} (rc={rc})", file=sys.stderr)
            return


def fetch_mimic4() -> None:
    dest = DATA / "mimiciv" / "hosp"
    for t in MIMIC4_TABLES:
        rc = _wget(f"{PHYSIONET_FILES}/{MIMIC4_HOSP}/{t}.csv.gz", dest, credentialed=True)
        if rc != 0:
            print(f"failed: {t} (rc={rc})", file=sys.stderr)
            return
    _wget(f"{PHYSIONET_FILES}/{MIMIC4_WDB}/RECORDS", DATA / "mimic4wdb", credentialed=True)


def fetch_vitaldb_index() -> None:
    import pandas as pd

    dest = DATA / "vitaldb"
    dest.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(f"{VITALDB_API}/cases")
    trks = pd.read_csv(f"{VITALDB_API}/trks")
    cases.to_csv(dest / "cases.csv", index=False)
    trks.to_csv(dest / "trks.csv", index=False)
    print(f"{len(cases)} cases, {len(trks)} tracks")
    op = cases.get("opname", pd.Series(dtype=str)).astype(str).str.lower()
    n_cs = int(op.str.contains("cesarean|caesarean|c-sec|c/s").sum())
    print(f"cases with obstetric (caesarean) opname: {n_cs}")
    if "ane_type" in cases:
        print(cases["ane_type"].value_counts(dropna=False).to_string())


def fetch_vitaldb_cases(n: int) -> None:
    """Download the first n eligible VitalDB cases (women 18-45 with ECG_II, PLETH and ART)."""
    import pandas as pd

    try:
        import vitaldb
    except ImportError:
        print("pip install vitaldb", file=sys.stderr)
        return
    dest = DATA / "vitaldb"
    cases_path = dest / "cases.csv"
    trks_path = dest / "trks.csv"
    if not cases_path.exists():
        fetch_vitaldb_index()
    cases = pd.read_csv(cases_path)
    trks = pd.read_csv(trks_path)
    need = ["SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"]
    have = trks[trks["tname"].isin(need)].groupby("caseid")["tname"].nunique()
    ok_ids = set(have[have == len(need)].index)
    sel = cases[(cases["sex"] == "F") & (cases["age"].between(18, 45)) & (cases["caseid"].isin(ok_ids))]
    sel = sel.head(n)
    for cid in sel["caseid"]:
        vf = vitaldb.VitalFile(int(cid), need)
        arr = vf.to_numpy(need, 1 / 100.0)  # 100 Hz common grid
        out = dest / f"case_{int(cid)}.npz"
        import numpy as np

        np.savez_compressed(out, data=arr, tracks=np.array(need), fs=100.0)
        print(f"case {cid}: {arr.shape} -> {out}")


def check() -> None:
    parts = {
        "mimic3 waveform index": DATA / "mimic3wdb-matched" / "RECORDS",
        "mimic3 clinical DIAGNOSES_ICD": DATA / "mimiciii" / "DIAGNOSES_ICD.csv.gz",
        "mimic4 hosp diagnoses_icd": DATA / "mimiciv" / "hosp" / "diagnoses_icd.csv.gz",
        "mimic4 waveform index": DATA / "mimic4wdb" / "RECORDS",
        "vitaldb cases": DATA / "vitaldb" / "cases.csv",
    }
    for k, p in parts.items():
        print(f"{'OK ' if p.exists() else '-- '} {k}: {p}")
    vd = list((DATA / "vitaldb").glob("case_*.npz")) if (DATA / "vitaldb").exists() else []
    print(f"vitaldb cases downloaded: {len(vd)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic3-index", action="store_true")
    ap.add_argument("--sample", action="store_true", help="read one 60-s matched-subset record via wfdb")
    ap.add_argument("--mimic3-clinical", action="store_true")
    ap.add_argument("--mimic4", action="store_true")
    ap.add_argument("--vitaldb-index", action="store_true")
    ap.add_argument("--vitaldb-cases", type=int, default=0, metavar="N")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.mimic3_index:
        fetch_mimic3_index()
    if a.sample:
        fetch_sample_record()
    if a.mimic3_clinical:
        fetch_mimic3_clinical()
    if a.mimic4:
        fetch_mimic4()
    if a.vitaldb_index:
        fetch_vitaldb_index()
    if a.vitaldb_cases:
        fetch_vitaldb_cases(a.vitaldb_cases)
    if a.check or not any([a.mimic3_index, a.sample, a.mimic3_clinical, a.mimic4, a.vitaldb_index, a.vitaldb_cases]):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
