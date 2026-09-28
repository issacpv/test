#!/usr/bin/env python3
"""Download the four scalp-EEG seizure corpora used by xseizure.

Open corpora (CHB-MIT, Siena, Helsinki) are downloaded directly; TUSZ needs
credentials that are read from environment variables and is fetched with rsync.

Examples
--------
    python scripts/download_data.py --dataset chbmit --sample
    python scripts/download_data.py --dataset siena
    TUH_USERNAME=... TUH_PASSWORD=... python scripts/download_data.py --dataset tusz --sample
    HELSINKI_ZENODO_RECORD=1234567 python scripts/download_data.py --dataset helsinki --sample
    python scripts/download_data.py --build-manifests

No credentials are ever written to disk by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PHYSIONET = "https://physionet.org/files"
CHBMIT_URL = f"{PHYSIONET}/chbmit/1.0.0/"
SIENA_URL = f"{PHYSIONET}/siena-scalp-eeg/1.0.0/"
TUSZ_RSYNC = "www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/"
ZENODO_API = "https://zenodo.org/api/records/{record}"

CHBMIT_SAMPLE_FILES = [
    "chb01/chb01-summary.txt",
    "chb01/chb01_03.edf",  # contains a seizure
    "chb01/chb01_01.edf",
    "RECORDS",
    "RECORDS-WITH-SEIZURES",
    "SUBJECT-INFO",
]
SIENA_SAMPLE_FILES = [
    "subject_info.csv",
    "PN00/Seizures-list-PN00.txt",
    "PN00/PN00-1.edf",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
                existing = 0  # server ignored the range request
            mode = "ab" if existing else "wb"
            with open(dest, mode) as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        if exc.code == 416:  # already complete
            return
        raise
    _log(f"saved {dest.relative_to(ROOT)}")


def _wget_mirror(url: str, dest: Path, files: Optional[Iterable[str]] = None) -> None:
    """Mirror a PhysioNet folder with wget, or fetch a subset of files."""
    dest.mkdir(parents=True, exist_ok=True)
    if files is not None:
        for rel in files:
            _download_file(url + rel, dest / rel)
        return
    if not _have("wget"):
        sys.exit("wget not found; install it or use --sample (pure-python subset).")
    cut = url.replace("https://physionet.org/", "").count("/")
    _run(["wget", "-r", "-N", "-c", "-np", "-nH", f"--cut-dirs={cut}", "-P", str(dest), url])


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def download_chbmit(sample: bool) -> None:
    _wget_mirror(CHBMIT_URL, DATA / "chbmit", CHBMIT_SAMPLE_FILES if sample else None)


def download_siena(sample: bool) -> None:
    _wget_mirror(SIENA_URL, DATA / "siena", SIENA_SAMPLE_FILES if sample else None)


def download_tusz(sample: bool) -> None:
    user = os.environ.get("TUH_USERNAME")
    pw = os.environ.get("TUH_PASSWORD")
    if not user:
        sys.exit(
            "TUSZ requires registration: fill the form at "
            "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ then "
            "export TUH_USERNAME and TUH_PASSWORD."
        )
    if not _have("rsync"):
        sys.exit("rsync is required for TUSZ downloads.")
    dest = DATA / "tusz"
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{user}@{TUSZ_RSYNC}"
    if sample:
        src += "edf/dev/"
        dest = dest / "edf" / "dev"
        dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-auxvL", "--partial", "--progress", src, str(dest) + "/"]
    if pw and _have("sshpass"):
        cmd = ["sshpass", "-e", *cmd]
        env = dict(os.environ, SSHPASS=pw)
    else:
        env = None
        if pw:
            _log("sshpass not found; rsync will prompt for the password interactively.")
    rc = _run(cmd, env=env)
    if rc != 0:
        sys.exit(f"rsync exited with {rc}")


def download_helsinki(sample: bool) -> None:
    record = os.environ.get("HELSINKI_ZENODO_RECORD")
    if not record:
        sys.exit(
            "Set HELSINKI_ZENODO_RECORD to the numeric Zenodo record id of "
            "'A dataset of neonatal EEG recordings with seizure annotations' "
            "(Stevenson et al., 2019). Search zenodo.org for the title."
        )
    dest = DATA / "helsinki"
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API.format(record=record), timeout=60) as resp:
        meta = json.load(resp)
    files = meta.get("files", [])
    if not files:
        sys.exit("Zenodo record lists no files (is it the concept id? use the version id).")
    # Zenodo records are not paginated for files, but `links.self` may point to
    # a versioned record; follow `latest` if the user gave the concept record.
    if "latest" in meta.get("links", {}) and meta["links"]["latest"] != meta["links"].get("self"):
        _log("Following link to the latest version of the record.")
        with urllib.request.urlopen(meta["links"]["latest"], timeout=60) as resp:
            meta = json.load(resp)
            files = meta.get("files", files)
    edfs = sorted(f for f in files if f["key"].lower().endswith(".edf"))
    others = [f for f in files if not f["key"].lower().endswith(".edf")]
    chosen = others + (edfs[:2] if sample else edfs)
    for f in chosen:
        url = f["links"].get("self") or f["links"].get("download")
        _download_file(url, dest / f["key"])


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def build_manifests() -> None:
    """Create data/manifests/{records,events}.csv from whatever is downloaded."""
    import csv
    import re

    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    records, events = [], []

    # CHB-MIT: parse summary files
    for summ in sorted((DATA / "chbmit").glob("chb*/chb*-summary.txt")):
        subj = summ.parent.name
        fs = 256
        text = summ.read_text(errors="ignore")
        m = re.search(r"Data Sampling Rate:\s*(\d+)", text)
        if m:
            fs = int(m.group(1))
        blocks = re.split(r"\n(?=File Name:)", text)
        for blk in blocks:
            fm = re.search(r"File Name:\s*(\S+)", blk)
            if not fm:
                continue
            fname = fm.group(1)
            path = summ.parent / fname
            records.append(dict(dataset="chbmit", record_id=fname[:-4], subject_id=subj,
                                path=str(path), fs=fs, n_channels="", duration_s="", age_years=""))
            starts = [int(x) for x in re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)", blk)]
            ends = [int(x) for x in re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)", blk)]
            for s, e in zip(starts, ends):
                events.append(dict(dataset="chbmit", record_id=fname[:-4], onset_s=s, offset_s=e,
                                   annotator="chbmit"))

    # Siena: record list only (wall-clock event parsing needs EDF start times; see xseizure.io_edf)
    for edf in sorted((DATA / "siena").glob("PN*/PN*.edf")):
        records.append(dict(dataset="siena", record_id=edf.stem, subject_id=edf.parent.name,
                            path=str(edf), fs=512, n_channels="", duration_s="", age_years=""))

    # TUSZ: csv_bi term annotations
    for csvbi in sorted((DATA / "tusz").rglob("*.csv_bi")):
        parts = csvbi.parts
        try:
            pid = parts[parts.index("edf") + 2]
        except (ValueError, IndexError):
            pid = csvbi.parents[2].name
        rid = csvbi.stem
        records.append(dict(dataset="tusz", record_id=rid, subject_id=pid,
                            path=str(csvbi.with_suffix(".edf")), fs="", n_channels="",
                            duration_s="", age_years=""))
        for line in csvbi.read_text(errors="ignore").splitlines():
            if line.startswith("#") or line.startswith("channel"):
                continue
            cols = line.split(",")
            if len(cols) >= 4 and cols[3].strip() == "seiz":
                events.append(dict(dataset="tusz", record_id=rid, onset_s=float(cols[1]),
                                   offset_s=float(cols[2]), annotator="tusz"))

    # Helsinki: per-second annotation matrices (columns = neonates)
    hel = DATA / "helsinki"
    for ann in sorted(hel.glob("annotations_2017_*.csv")):
        annot = ann.stem[-1]
        with open(ann, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        ncols = len(rows[0])
        for j in range(ncols):
            col = [r[j] for r in rows if j < len(r) and r[j] not in ("", "NaN")]
            vals = [int(float(v)) for v in col]
            rid = f"eeg{j + 1}"
            if annot == "A":
                records.append(dict(dataset="helsinki", record_id=rid, subject_id=rid,
                                    path=str(hel / f"{rid}.edf"), fs=256, n_channels="",
                                    duration_s=len(vals), age_years=0.0))
            on = None
            for t, v in enumerate(vals + [0]):
                if v == 1 and on is None:
                    on = t
                elif v == 0 and on is not None:
                    events.append(dict(dataset="helsinki", record_id=rid, onset_s=on,
                                       offset_s=t, annotator=annot))
                    on = None

    for name, rows, cols in (
        ("records.csv", records, ["dataset", "record_id", "subject_id", "path", "fs",
                                  "n_channels", "duration_s", "age_years"]),
        ("events.csv", events, ["dataset", "record_id", "onset_s", "offset_s", "annotator"]),
    ):
        with open(man / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _log(f"wrote {name}: {len(rows)} rows")


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["chbmit", "siena", "tusz", "helsinki", "all"])
    p.add_argument("--sample", action="store_true", help="download a small subset only")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)

    if args.dataset:
        todo = ["chbmit", "siena", "helsinki", "tusz"] if args.dataset == "all" else [args.dataset]
        for ds in todo:
            {"chbmit": download_chbmit, "siena": download_siena,
             "tusz": download_tusz, "helsinki": download_helsinki}[ds](args.sample)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
