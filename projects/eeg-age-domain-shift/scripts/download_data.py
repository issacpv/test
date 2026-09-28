#!/usr/bin/env python3
"""Download the corpora used by eegage and build age-aware manifests.

Open corpora (CHB-MIT, Siena, Helsinki) are downloaded directly; TUSZ/TUAB need
credentials read from environment variables and are fetched with rsync.

Examples
--------
    python scripts/download_data.py --dataset chbmit --sample
    python scripts/download_data.py --dataset siena
    TUH_USERNAME=... TUH_PASSWORD=... python scripts/download_data.py --dataset tusz --sample
    TUH_USERNAME=... TUH_PASSWORD=... TUAB_VERSION=v3.0.1 python scripts/download_data.py --dataset tuab
    HELSINKI_ZENODO_RECORD=1234567 python scripts/download_data.py --dataset helsinki --sample
    python scripts/download_data.py --build-manifests

No credentials are ever written to disk by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

PHYSIONET = "https://physionet.org/files"
CHBMIT_URL = f"{PHYSIONET}/chbmit/1.0.0/"
SIENA_URL = f"{PHYSIONET}/siena-scalp-eeg/1.0.0/"
TUH_HOST = "www.isip.piconepress.com"
TUSZ_PATH = "data/tuh_eeg_seizure/v2.0.3/"
TUAB_PATH = "data/tuh_eeg_abnormal/{version}/"
ZENODO_API = "https://zenodo.org/api/records/{record}"

CHBMIT_SAMPLE_FILES = ["SUBJECT-INFO", "RECORDS", "RECORDS-WITH-SEIZURES", "chb01/chb01-summary.txt",
                       "chb01/chb01_03.edf", "chb01/chb01_01.edf"]
SIENA_SAMPLE_FILES = ["subject_info.csv", "PN00/Seizures-list-PN00.txt", "PN00/PN00-1.edf"]


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd: List[str], env: Optional[dict] = None) -> int:
    _log(" ".join(c if "PASSWORD" not in c else "***" for c in cmd))
    return subprocess.call(cmd, env=env)


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Resumable single-file HTTP download using only the standard library."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if existing and resp.status != 206:
                existing = 0
            with open(dest, "ab" if existing else "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        if exc.code == 416:
            return
        raise
    _log(f"saved {dest.relative_to(ROOT)}")


def _wget_mirror(url: str, dest: Path, files: Optional[Iterable[str]] = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if files is not None:
        for rel in files:
            _download_file(url + rel, dest / rel)
        return
    if not _have("wget"):
        sys.exit("wget not found; install it or use --sample (pure-python subset).")
    cut = url.replace("https://physionet.org/", "").count("/")
    _run(["wget", "-r", "-N", "-c", "-np", "-nH", f"--cut-dirs={cut}", "-P", str(dest), url])


def _rsync_tuh(remote_path: str, dest: Path, sample_subdir: Optional[str]) -> None:
    user = os.environ.get("TUH_USERNAME")
    pw = os.environ.get("TUH_PASSWORD")
    if not user:
        sys.exit("Set TUH_USERNAME (and TUH_PASSWORD) after registering at "
                 "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")
    if not _have("rsync"):
        sys.exit("rsync is required for TUH corpora.")
    src = f"{user}@{TUH_HOST}:{remote_path}" + (sample_subdir or "")
    tgt = dest / (sample_subdir or "")
    tgt.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-auxvL", "--partial", src, str(tgt) + "/"]
    env = None
    if pw and _have("sshpass"):
        cmd = ["sshpass", "-e"] + cmd
        env = dict(os.environ, SSHPASS=pw)
    else:
        _log("sshpass not found or TUH_PASSWORD unset: rsync will prompt for the password")
    _run(cmd, env=env)


def download_chbmit(sample: bool) -> None:
    _wget_mirror(CHBMIT_URL, DATA / "chbmit", CHBMIT_SAMPLE_FILES if sample else None)


def download_siena(sample: bool) -> None:
    _wget_mirror(SIENA_URL, DATA / "siena", SIENA_SAMPLE_FILES if sample else None)


def download_tusz(sample: bool) -> None:
    _rsync_tuh(TUSZ_PATH, DATA / "tusz", "edf/dev/" if sample else None)


def download_tuab(sample: bool) -> None:
    version = os.environ.get("TUAB_VERSION", "v3.0.1")
    _rsync_tuh(TUAB_PATH.format(version=version), DATA / "tuab", "edf/eval/" if sample else None)


def download_helsinki(sample: bool) -> None:
    record = os.environ.get("HELSINKI_ZENODO_RECORD")
    if not record:
        sys.exit("Set HELSINKI_ZENODO_RECORD to the Zenodo record id of "
                 "'A dataset of neonatal EEG recordings with seizure annotations'.")
    with urllib.request.urlopen(ZENODO_API.format(record=record), timeout=60) as resp:
        meta = json.load(resp)
    files = meta.get("files", [])
    dest = DATA / "helsinki"
    edfs = sorted(f for f in files if f["key"].lower().endswith(".edf"))
    others = [f for f in files if not f["key"].lower().endswith(".edf")]
    todo = others + (edfs[:2] if sample else edfs)
    for f in todo:
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if url:
            _download_file(url, dest / f["key"])


# --------------------------------------------------------------------------- #
def build_manifests() -> None:
    from eegage import ages as A  # noqa: WPS433 - local package

    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    events: List[dict] = []

    # CHB-MIT
    info = DATA / "chbmit" / "SUBJECT-INFO"
    chb_ages = A.parse_chbmit_subject_info(info.read_text(errors="ignore")) if info.exists() else {}
    for summ in sorted((DATA / "chbmit").glob("chb*/chb*-summary.txt")):
        sub = summ.parent.name
        age = chb_ages.get(sub, {}).get("age", float("nan"))
        text = summ.read_text(errors="ignore")
        for blk in re.split(r"\n(?=File Name:)", text):
            m = re.search(r"File Name:\s*(\S+)", blk)
            if not m:
                continue
            fname = m.group(1)
            records.append(dict(dataset="chbmit", record_id=fname[:-4], subject_id=sub,
                                path=str(summ.parent / fname), fs=256, age_years=age, age_bin=A.age_bin(age)))
            starts = [int(x) for x in re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)", blk)]
            ends = [int(x) for x in re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)", blk)]
            for s, e in zip(starts, ends):
                events.append(dict(dataset="chbmit", record_id=fname[:-4], onset_s=s, offset_s=e))

    # Siena
    sinfo = DATA / "siena" / "subject_info.csv"
    siena_ages = A.parse_siena_subject_info(sinfo.read_text(errors="ignore")) if sinfo.exists() else {}
    for edf in sorted((DATA / "siena").glob("PN*/PN*.edf")):
        age = siena_ages.get(edf.parent.name, float("nan"))
        records.append(dict(dataset="siena", record_id=edf.stem, subject_id=edf.parent.name, path=str(edf),
                            fs=512, age_years=age, age_bin=A.age_bin(age)))

    # TUSZ / TUAB: ages from EDF headers
    for name in ("tusz", "tuab"):
        for edf in sorted((DATA / name).rglob("*.edf")):
            age = A.parse_edf_header_age(edf)
            parts = edf.parts
            try:
                pid = parts[parts.index("edf") + 2] if name == "tusz" else parts[parts.index("edf") + 3]
            except (ValueError, IndexError):
                pid = edf.parents[2].name
            records.append(dict(dataset=name, record_id=edf.stem, subject_id=pid, path=str(edf), fs="",
                                age_years=age if age is not None else float("nan"), age_bin=A.age_bin(age)))
            csvbi = edf.with_suffix(".csv_bi")
            if csvbi.exists():
                for line in csvbi.read_text(errors="ignore").splitlines():
                    cols = line.split(",")
                    if len(cols) >= 4 and cols[3].strip() == "seiz":
                        events.append(dict(dataset=name, record_id=edf.stem, onset_s=float(cols[1]),
                                           offset_s=float(cols[2])))

    # Helsinki: age 0 (term neonates)
    for edf in sorted((DATA / "helsinki").glob("eeg*.edf")):
        records.append(dict(dataset="helsinki", record_id=edf.stem, subject_id=edf.stem, path=str(edf), fs=256,
                            age_years=0.0, age_bin=A.age_bin(0.0)))

    # ages.csv: one row per subject
    per_sub: dict = {}
    for r in records:
        key = (r["dataset"], r["subject_id"])
        d = per_sub.setdefault(key, dict(dataset=r["dataset"], subject_id=r["subject_id"],
                                         age_years=r["age_years"], age_bin=r["age_bin"], n_records=0))
        d["n_records"] += 1

    for name, rows, cols in (
        ("records.csv", records, ["dataset", "record_id", "subject_id", "path", "fs", "age_years", "age_bin"]),
        ("events.csv", events, ["dataset", "record_id", "onset_s", "offset_s"]),
        ("ages.csv", list(per_sub.values()), ["dataset", "subject_id", "age_years", "age_bin", "n_records"]),
    ):
        with open(man / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _log(f"wrote {name}: {len(rows)} rows")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["chbmit", "siena", "tusz", "tuab", "helsinki", "all"])
    p.add_argument("--sample", action="store_true", help="download a small subset only")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)

    if args.dataset:
        todo = ["chbmit", "siena", "helsinki", "tusz", "tuab"] if args.dataset == "all" else [args.dataset]
        for ds in todo:
            {"chbmit": download_chbmit, "siena": download_siena, "tusz": download_tusz, "tuab": download_tuab,
             "helsinki": download_helsinki}[ds](args.sample)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
