#!/usr/bin/env python3
"""Download the EEG+ECG seizure corpora used by szforecast.

Open corpora (SeizeIT2 on OpenNeuro, Siena and szdb on PhysioNet) are fetched
directly; TUSZ needs credentials read from environment variables (rsync).

Examples
--------
    python scripts/download_data.py --dataset seizeit2 --sample   # metadata + first subject
    python scripts/download_data.py --dataset seizeit2            # full (needs openneuro-py or aws cli)
    python scripts/download_data.py --dataset siena --sample
    python scripts/download_data.py --dataset szdb
    TUH_USERNAME=... TUH_PASSWORD=... python scripts/download_data.py --dataset tusz --sample
    python scripts/download_data.py --build-manifests

No credentials are ever written to disk by this script.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PHYSIONET = "https://physionet.org/files"
SIENA_URL = f"{PHYSIONET}/siena-scalp-eeg/1.0.0/"
SZDB_URL = f"{PHYSIONET}/szdb/1.0.0/"
TUSZ_RSYNC = "www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/"

OPENNEURO_DS = "ds005873"  # SeizeIT2
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"  # public bucket, HTTPS access

SIENA_SAMPLE_FILES = ["subject_info.csv", "PN00/Seizures-list-PN00.txt", "PN00/PN00-1.edf"]


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


def s3_list_keys(prefix: str, max_pages: int = 1000) -> Iterator[str]:
    """Iterate object keys under ``prefix`` in the public OpenNeuro S3 bucket (ListObjectsV2 pagination)."""
    token: Optional[str] = None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    for _ in range(max_pages):
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{OPENNEURO_S3}/?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            tree = ET.fromstring(resp.read())
        for c in tree.iter(f"{ns}Contents"):
            key = c.find(f"{ns}Key")
            if key is not None and key.text:
                yield key.text
        trunc = tree.find(f"{ns}IsTruncated")
        if trunc is None or trunc.text != "true":
            return
        nxt = tree.find(f"{ns}NextContinuationToken")
        token = nxt.text if nxt is not None else None
        if not token:
            return


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def download_seizeit2(sample: bool) -> None:
    dest = DATA / "seizeit2"
    dest.mkdir(parents=True, exist_ok=True)
    if sample:
        # metadata files + everything under the first subject folder
        top = [k for k in s3_list_keys(f"{OPENNEURO_DS}/") if k.count("/") == 1]
        wanted = [k for k in top if k.endswith((".json", ".tsv", ".md", "README", "CHANGES"))]
        subjects = sorted({k.split("/")[1] for k in s3_list_keys(f"{OPENNEURO_DS}/sub-") if "/sub-" in k})
        if subjects:
            wanted += list(s3_list_keys(f"{OPENNEURO_DS}/{subjects[0]}/"))
        for key in wanted:
            rel = key[len(OPENNEURO_DS) + 1:]
            if rel:
                _download_file(f"{OPENNEURO_S3}/{key}", dest / rel)
        _log(f"sample: {len(wanted)} files")
        return
    if _have("openneuro-py"):
        _run(["openneuro-py", "download", f"--dataset={OPENNEURO_DS}", f"--target-dir={dest}"])
    elif _have("aws"):
        _run(["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{OPENNEURO_DS}", str(dest)])
    else:
        sys.exit("Install openneuro-py (pip install openneuro-py) or the AWS CLI for the full download, "
                 "or run with --sample.")


def download_siena(sample: bool) -> None:
    _wget_mirror(SIENA_URL, DATA / "siena", SIENA_SAMPLE_FILES if sample else None)


def download_szdb(sample: bool) -> None:
    files = ["RECORDS", "times.seize", "sz01.hea", "sz01.dat", "sz01.ari"] if sample else None
    _wget_mirror(SZDB_URL, DATA / "szdb", files)


def download_tusz(sample: bool) -> None:
    user = os.environ.get("TUH_USERNAME")
    pw = os.environ.get("TUH_PASSWORD")
    if not user:
        sys.exit("Set TUH_USERNAME (and TUH_PASSWORD) after registering at "
                 "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")
    if not _have("rsync"):
        sys.exit("rsync is required for TUSZ.")
    dest = DATA / "tusz"
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{user}@{TUSZ_RSYNC}" + ("edf/dev/" if sample else "")
    tgt = str(dest / "edf" / "dev") + "/" if sample else str(dest) + "/"
    Path(tgt).mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-auxvL", "--partial", src, tgt]
    if pw and _have("sshpass"):
        cmd = ["sshpass", "-e"] + cmd
        env = dict(os.environ, SSHPASS=pw)
    else:
        env = None
        _log("sshpass not found or TUH_PASSWORD unset: rsync will prompt for the password")
    _run(cmd, env=env)


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def _edf_header(path: Path) -> dict:
    """Parse the fixed EDF header and channel labels without external libraries."""
    with open(path, "rb") as fh:
        hdr = fh.read(256)
        if len(hdr) < 256:
            return {}
        n_rec = int(hdr[236:244].decode(errors="ignore").strip() or 0)
        rec_dur = float(hdr[244:252].decode(errors="ignore").strip() or 0)
        n_ch = int(hdr[252:256].decode(errors="ignore").strip() or 0)
        labels = [fh.read(16).decode(errors="ignore").strip() for _ in range(n_ch)]
        fh.seek(256 + n_ch * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
        nsamp = [int(fh.read(8).decode(errors="ignore").strip() or 0) for _ in range(n_ch)]
    fs = (nsamp[0] / rec_dur) if (nsamp and rec_dur) else ""
    ecg = [l for l in labels if re.search(r"E[CK]G", l, re.I)]
    return dict(start=hdr[168:176].decode(errors="ignore") + " " + hdr[176:184].decode(errors="ignore"),
                duration_s=n_rec * rec_dur, fs=fs, labels=labels, ecg_channel=ecg[0] if ecg else "")


def build_manifests() -> None:
    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    events: List[dict] = []

    # SeizeIT2 (BIDS): events.tsv next to each EDF
    for edf in sorted((DATA / "seizeit2").rglob("*_eeg.edf")):
        h = _edf_header(edf)
        sub = edf.name.split("_")[0]
        rid = edf.name.replace("_eeg.edf", "")
        records.append(dict(dataset="seizeit2", record_id=rid, subject_id=sub, path=str(edf), fs=h.get("fs", ""),
                            has_ecg=bool(h.get("ecg_channel")), ecg_channel=h.get("ecg_channel", ""),
                            start_clock=h.get("start", ""), duration_s=h.get("duration_s", "")))
        ev = edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv"))
        if ev.exists():
            with open(ev, newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    try:
                        on = float(row.get("onset", "nan"))
                        du = float(row.get("duration", "nan"))
                    except ValueError:
                        continue
                    etype = row.get("eventType", row.get("trial_type", "sz"))
                    events.append(dict(dataset="seizeit2", record_id=rid, onset_s=on, offset_s=on + du,
                                       event_type=etype))

    # Siena: header info + record list (event wall-clock conversion happens in szforecast.windows)
    for edf in sorted((DATA / "siena").glob("PN*/PN*.edf")):
        h = _edf_header(edf)
        records.append(dict(dataset="siena", record_id=edf.stem, subject_id=edf.parent.name, path=str(edf),
                            fs=h.get("fs", ""), has_ecg=bool(h.get("ecg_channel")),
                            ecg_channel=h.get("ecg_channel", ""), start_clock=h.get("start", ""),
                            duration_s=h.get("duration_s", "")))

    # TUSZ: csv_bi term annotations + EKG availability from the EDF header
    for csvbi in sorted((DATA / "tusz").rglob("*.csv_bi")):
        edf = csvbi.with_suffix(".edf")
        h = _edf_header(edf) if edf.exists() else {}
        parts = csvbi.parts
        try:
            pid = parts[parts.index("edf") + 2]
        except (ValueError, IndexError):
            pid = csvbi.parents[2].name
        rid = csvbi.stem
        records.append(dict(dataset="tusz", record_id=rid, subject_id=pid, path=str(edf), fs=h.get("fs", ""),
                            has_ecg=bool(h.get("ecg_channel")), ecg_channel=h.get("ecg_channel", ""),
                            start_clock=h.get("start", ""), duration_s=h.get("duration_s", "")))
        for line in csvbi.read_text(errors="ignore").splitlines():
            if line.startswith("#") or line.startswith("channel"):
                continue
            cols = line.split(",")
            if len(cols) >= 4 and cols[3].strip() == "seiz":
                events.append(dict(dataset="tusz", record_id=rid, onset_s=float(cols[1]),
                                   offset_s=float(cols[2]), event_type="seiz"))

    # szdb: times.seize (record, seizure time in seconds)
    ts = DATA / "szdb" / "times.seize"
    if ts.exists():
        for line in ts.read_text(errors="ignore").splitlines():
            toks = line.split()
            if len(toks) >= 2:
                try:
                    events.append(dict(dataset="szdb", record_id=toks[0], onset_s=float(toks[1]),
                                       offset_s="", event_type="seiz"))
                except ValueError:
                    continue

    for name, rows, cols in (
        ("records.csv", records, ["dataset", "record_id", "subject_id", "path", "fs", "has_ecg", "ecg_channel",
                                  "start_clock", "duration_s"]),
        ("events.csv", events, ["dataset", "record_id", "onset_s", "offset_s", "event_type"]),
    ):
        with open(man / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _log(f"wrote {name}: {len(rows)} rows")


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["seizeit2", "siena", "szdb", "tusz", "all"])
    p.add_argument("--sample", action="store_true", help="download a small subset only")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)

    if args.dataset:
        todo = ["seizeit2", "siena", "szdb", "tusz"] if args.dataset == "all" else [args.dataset]
        for ds in todo:
            {"seizeit2": download_seizeit2, "siena": download_siena, "szdb": download_szdb,
             "tusz": download_tusz}[ds](args.sample)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
