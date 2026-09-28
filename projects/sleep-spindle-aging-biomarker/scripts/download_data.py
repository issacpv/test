#!/usr/bin/env python3
"""Download sleep PSG cohorts for spindle_age.

Open: Sleep-EDF Expanded and HMC (PhysioNet), OpenNeuro (openneuro-py).
DUA:  SHHS / MESA / MrOS / CFS from the NSRR via the `nsrr` Ruby gem and NSRR_TOKEN.

Examples
--------
    python scripts/download_data.py --dataset sleep-edf --sample
    python scripts/download_data.py --dataset hmc --sample
    NSRR_TOKEN=... python scripts/download_data.py --dataset shhs --sample
    python scripts/download_data.py --dataset openneuro --openneuro-id ds00xxxx --sample
"""
from __future__ import annotations

import argparse
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
SLEEP_EDF_URL = f"{PHYSIONET}/sleep-edfx/1.0.0/"
HMC_URL = f"{PHYSIONET}/hmc-sleep-staging/1.1/"

SLEEP_EDF_SAMPLE = [
    "SC-subjects.xls", "ST-subjects.xls",
    "sleep-cassette/SC4001E0-PSG.edf", "sleep-cassette/SC4001EC-Hypnogram.edf",
    "sleep-cassette/SC4002E0-PSG.edf", "sleep-cassette/SC4002EC-Hypnogram.edf",
]
HMC_SAMPLE = ["recordings/SN001.edf", "recordings/SN001_sleepscoring.edf"]

# NSRR dataset layouts: (slug, edf folders, annotation folders, sample glob)
NSRR = {
    "shhs": dict(edfs=["shhs/polysomnography/edfs/shhs1", "shhs/polysomnography/edfs/shhs2"],
                 ann=["shhs/polysomnography/annotations-events-nsrr/shhs1",
                      "shhs/polysomnography/annotations-events-nsrr/shhs2"],
                 datasets="shhs/datasets", sample="shhs1-20000*"),
    "mesa": dict(edfs=["mesa/polysomnography/edfs"], ann=["mesa/polysomnography/annotations-events-nsrr"],
                 datasets="mesa/datasets", sample="mesa-sleep-000*"),
    "mros": dict(edfs=["mros/polysomnography/edfs/visit1", "mros/polysomnography/edfs/visit2"],
                 ann=["mros/polysomnography/annotations-events-nsrr/visit1",
                      "mros/polysomnography/annotations-events-nsrr/visit2"],
                 datasets="mros/datasets", sample="mros-visit1-aa000*"),
    "cfs": dict(edfs=["cfs/polysomnography/edfs"], ann=["cfs/polysomnography/annotations-events-nsrr"],
                datasets="cfs/datasets", sample="cfs-visit5-80000*"),
}


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Resumable single-file download with the standard library."""
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


def _physionet(url: str, dest: Path, files: Optional[Iterable[str]]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if files is not None:
        for rel in files:
            _download_file(url + rel, dest / rel)
        return
    if shutil.which("wget") is None:
        sys.exit("wget is required for full PhysioNet mirrors (or use --sample)")
    cut = url.replace("https://physionet.org/", "").count("/")
    subprocess.check_call(["wget", "-r", "-N", "-c", "-np", "-nH", f"--cut-dirs={cut}", "-P", str(dest), url])


def download_sleep_edf(sample: bool) -> None:
    _physionet(SLEEP_EDF_URL, DATA / "sleep-edf", SLEEP_EDF_SAMPLE if sample else None)


def download_hmc(sample: bool) -> None:
    _physionet(HMC_URL, DATA / "hmc", HMC_SAMPLE if sample else None)


def _nsrr(args: List[str], token: str) -> None:
    cmd = ["nsrr", *args, f"--token={token}"]
    _log(" ".join(a if not a.startswith("--token") else "--token=***" for a in cmd))
    rc = subprocess.call(cmd, cwd=str(DATA / "nsrr"))
    if rc != 0:
        sys.exit(f"nsrr exited with {rc}")


def download_nsrr(slug: str, sample: bool) -> None:
    token = os.environ.get("NSRR_TOKEN")
    if not token:
        sys.exit("Set NSRR_TOKEN (https://sleepdata.org/token) after your DUA for "
                 f"'{slug}' is approved at https://sleepdata.org/datasets/{slug}.")
    if shutil.which("nsrr") is None:
        sys.exit("The NSRR downloader is a Ruby gem: `gem install nsrr` "
                 "(https://github.com/nsrr/nsrr-gem). Files can also be browsed at "
                 f"https://sleepdata.org/datasets/{slug}/files.")
    (DATA / "nsrr").mkdir(parents=True, exist_ok=True)
    spec = NSRR[slug]
    _nsrr(["download", spec["datasets"]], token)
    for folder in spec["ann"]:
        _nsrr(["download", folder] + ([f"--file={spec['sample']}"] if sample else []), token)
    for folder in spec["edfs"]:
        _nsrr(["download", folder] + ([f"--file={spec['sample']}"] if sample else []), token)
        if sample:
            break


def download_openneuro(ds_id: Optional[str], sample: bool) -> None:
    if not ds_id:
        sys.exit("--openneuro-id dsXXXXXX is required (search https://openneuro.org for EEG sleep datasets)")
    try:
        import openneuro  # noqa: F401
    except ImportError:
        sys.exit("pip install openneuro-py")
    dest = DATA / "openneuro" / ds_id
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["openneuro-py", "download", f"--dataset={ds_id}", f"--target-dir={dest}"]
    if sample:
        cmd += ["--include=sub-01", "--include=participants.tsv", "--include=dataset_description.json"]
    _log(" ".join(cmd))
    subprocess.check_call(cmd)


def download_dod() -> None:
    if shutil.which("aws") is None:
        sys.exit("pip install awscli; then: aws s3 sync --no-sign-request s3://dreem-dod-h data/dod/dod-h")
    for b in ("dod-h", "dod-o"):
        subprocess.check_call(["aws", "s3", "sync", "--no-sign-request", f"s3://dreem-{b}", str(DATA / "dod" / b)])


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["sleep-edf", "hmc", "shhs", "mesa", "mros", "cfs", "openneuro", "dod"])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--openneuro-id")
    a = p.parse_args(argv)
    if a.dataset == "sleep-edf":
        download_sleep_edf(a.sample)
    elif a.dataset == "hmc":
        download_hmc(a.sample)
    elif a.dataset in NSRR:
        download_nsrr(a.dataset, a.sample)
    elif a.dataset == "openneuro":
        download_openneuro(a.openneuro_id, a.sample)
    elif a.dataset == "dod":
        download_dod()


if __name__ == "__main__":
    main()
