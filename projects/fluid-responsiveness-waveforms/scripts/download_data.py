#!/usr/bin/env python3
"""Download the data used by ``fluid_resp``.

Open parts (no credentials): MIMIC-IV demo, MIMIC-III matched waveform sample records, MIMIC-IV waveform
records, VitalDB cases.  Credentialed parts (MIMIC-III / MIMIC-IV clinical tables) read
``PHYSIONET_USER`` / ``PHYSIONET_PASS`` from the environment and pass them to ``wget`` only.

Examples
--------
    python scripts/download_data.py --sample                 # demo + 3 waveform records + 2 VitalDB cases
    python scripts/download_data.py --mimic3-clinical
    python scripts/download_data.py --mimic4-clinical
    python scripts/download_data.py --mimic4-waveforms --max-records 50
    python scripts/download_data.py --vitaldb --max-cases 200
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

PHYSIONET_FILES = "https://physionet.org/files"
DEMO = "mimic-iv-demo/2.2"
M3WDB = "mimic3wdb-matched/1.0"
M4WDB = "mimic4wdb/0.1.0"
M3_TABLES = ["INPUTEVENTS_MV", "CHARTEVENTS", "D_ITEMS", "ICUSTAYS", "PATIENTS", "ADMISSIONS"]
M4_TABLES = ["icu/inputevents", "icu/chartevents", "icu/d_items", "icu/icustays", "hosp/patients", "hosp/admissions"]
VITALDB_API = "https://api.vitaldb.net"
ART_TRACK = "SNUADC/ART"
SVV_TRACKS = ("EV1000/SVV", "Vigileo/SVV")


def _wget() -> str:
    exe = shutil.which("wget")
    if exe is None:
        sys.exit("wget not found; see data/README.md for manual commands")
    return exe


def _credentials() -> tuple[str, str]:
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not pw:
        sys.exit("PHYSIONET_USER / PHYSIONET_PASS not set (credentialed download).")
    return user, pw


def _run(cmd: list[str]) -> None:
    print("+", " ".join(c if not c.startswith("--password=") else "--password=***" for c in cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit("download failed")


def _get(url: str, timeout: float = 60.0) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


# ------------------------------------------------------------------------------------------ open resources
def download_demo(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _run([_wget(), "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-q", "--show-progress", f"{PHYSIONET_FILES}/{DEMO}/", "-P", str(out)])


def download_waveform_records(db: str, out: Path, max_records: int, list_file: str) -> None:
    """Download the first ``max_records`` multi-segment records (headers + segment files) of a PhysioNet waveform DB.

    Uses ``wfdb.dl_files``; segment names come from each record's master header (``seg_name``).
    """
    try:
        import wfdb  # noqa: WPS433
    except ImportError:
        sys.exit("pip install wfdb  (needed for waveform downloads)")
    dest = out / db
    dest.mkdir(parents=True, exist_ok=True)
    records = _get(f"{PHYSIONET_FILES}/{db}/{list_file}").text.split()
    records = [r for r in records if r and not r.startswith("#")][:max_records]
    for rec in records:
        rec_dir, rec_name = (rec.rsplit("/", 1) + [""])[:2] if "/" in rec else ("", rec)
        pn_dir = f"{db}/{rec_dir}".rstrip("/")
        try:
            hdr = wfdb.rdheader(rec_name, pn_dir=pn_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {rec}: {exc}")
            continue
        files = [f"{rec_dir}/{rec_name}.hea".lstrip("/")]
        seg_names = getattr(hdr, "seg_name", None) or []
        for seg in seg_names:
            if seg == "~":
                continue
            files += [f"{rec_dir}/{seg}.hea".lstrip("/"), f"{rec_dir}/{seg}.dat".lstrip("/")]
        print(f"{rec}: {len(files)} files")
        wfdb.dl_files(db, str(dest), files, keep_subdirs=True, overwrite=False)


def download_vitaldb(out: Path, max_cases: int) -> None:
    """Fetch VitalDB case/track lists and the ART + SVV tracks of cases that have both."""
    import pandas as pd  # local import keeps the script light

    dest = out / "vitaldb"
    (dest / "tracks").mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(io.StringIO(_get(f"{VITALDB_API}/cases").text))
    trks = pd.read_csv(io.StringIO(_get(f"{VITALDB_API}/trks").text))
    cases.to_csv(dest / "cases.csv", index=False)
    trks.to_csv(dest / "trks.csv", index=False)
    have_art = set(trks.loc[trks["tname"] == ART_TRACK, "caseid"])
    have_svv = set(trks.loc[trks["tname"].isin(SVV_TRACKS), "caseid"])
    ids = sorted(have_art & have_svv)[:max_cases]
    print(f"{len(have_art & have_svv)} cases with ART and SVV; fetching {len(ids)}")
    for cid in ids:
        sub = trks[(trks["caseid"] == cid) & (trks["tname"].isin((ART_TRACK,) + SVV_TRACKS))]
        for tid, tname in zip(sub["tid"], sub["tname"]):
            f = dest / "tracks" / f"{cid}_{tname.replace('/', '_')}.csv"
            if f.exists():
                continue
            f.write_text(_get(f"{VITALDB_API}/{tid}", timeout=300).text)
            print("  saved", f.name)


# ---------------------------------------------------------------------------------- credentialed resources
def download_mimic3_clinical(out: Path) -> None:
    user, pw = _credentials()
    dest = out / "mimiciii/1.4"
    dest.mkdir(parents=True, exist_ok=True)
    for t in M3_TABLES:
        _run([_wget(), "-N", "-c", "-q", "--show-progress", f"--user={user}", f"--password={pw}", f"{PHYSIONET_FILES}/mimiciii/1.4/{t}.csv.gz", "-P", str(dest)])


def download_mimic4_clinical(out: Path) -> None:
    user, pw = _credentials()
    for t in M4_TABLES:
        dest = out / "mimiciv/3.1" / f"{t}.csv.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run([_wget(), "-N", "-c", "-q", "--show-progress", f"--user={user}", f"--password={pw}", "-O", str(dest), f"{PHYSIONET_FILES}/mimiciv/3.1/{t}.csv.gz"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="open smoke-test subset")
    p.add_argument("--mimic3-clinical", action="store_true")
    p.add_argument("--mimic4-clinical", action="store_true")
    p.add_argument("--mimic4-waveforms", action="store_true")
    p.add_argument("--vitaldb", action="store_true")
    p.add_argument("--max-records", type=int, default=20)
    p.add_argument("--max-cases", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    a = p.parse_args(argv)
    if not any([a.sample, a.mimic3_clinical, a.mimic4_clinical, a.mimic4_waveforms, a.vitaldb]):
        p.error("choose at least one of --sample / --mimic3-clinical / --mimic4-clinical / --mimic4-waveforms / --vitaldb")
    if a.sample:
        download_demo(a.out)
        download_waveform_records(M3WDB, a.out, 3, "RECORDS-waveforms")
        download_vitaldb(a.out, 2)
    if a.mimic3_clinical:
        download_mimic3_clinical(a.out)
    if a.mimic4_clinical:
        download_mimic4_clinical(a.out)
    if a.mimic4_waveforms:
        download_waveform_records(M4WDB, a.out, a.max_records, "RECORDS")
    if a.vitaldb:
        download_vitaldb(a.out, a.max_cases)


if __name__ == "__main__":
    main()
