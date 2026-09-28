#!/usr/bin/env python3
"""Download the IED corpora used by iedbench.

Open OpenNeuro datasets are fetched through the public S3 listing API (sample
mode, pure python) or openneuro-py / aws cli (full); TUEV needs TUH credentials
(rsync); the MNI atlas and Bonn sets are manual web downloads (instructions
printed).

Examples
--------
    python scripts/download_data.py --dataset openneuro --accession ds003029 --sample
    python scripts/download_data.py --dataset openneuro --accession ds003876
    TUH_USERNAME=... TUH_PASSWORD=... TUEV_VERSION=v2.0.1 python scripts/download_data.py --dataset tuev --sample
    python scripts/download_data.py --dataset mni
    python scripts/download_data.py --dataset bonn
    python scripts/download_data.py --build-manifests
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"
TUH_HOST = "www.isip.piconepress.com"
TUEV_PATH = "data/tuh_eeg_events/{version}/"
TUEV_LABELS = {"spsw", "gped", "pled", "eyem", "artf", "bckg"}


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd: List[str], env: Optional[dict] = None) -> int:
    _log(" ".join(c if "PASSWORD" not in c else "***" for c in cmd))
    return subprocess.call(cmd, env=env)


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
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


def s3_list_keys(prefix: str, max_pages: int = 1000) -> Iterator[str]:
    """Iterate object keys under ``prefix`` in the public OpenNeuro bucket (ListObjectsV2 pagination)."""
    token: Optional[str] = None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    for _ in range(max_pages):
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{OPENNEURO_S3}/?{urllib.parse.urlencode(q)}", timeout=60) as resp:
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


def download_openneuro(accession: str, sample: bool) -> None:
    dest = DATA / "openneuro" / accession
    dest.mkdir(parents=True, exist_ok=True)
    if sample:
        keys = list(s3_list_keys(f"{accession}/"))
        top = [k for k in keys if k.count("/") == 1 and not k.endswith("/")]
        subjects = sorted({k.split("/")[1] for k in keys if k.split("/")[1].startswith("sub-")})
        wanted = top + ([k for k in keys if k.startswith(f"{accession}/{subjects[0]}/")] if subjects else [])
        for key in wanted:
            rel = key[len(accession) + 1:]
            if rel:
                _download_file(f"{OPENNEURO_S3}/{key}", dest / rel)
        _log(f"{accession}: sample of {len(wanted)} files ({len(subjects)} subjects listed)")
        return
    if _have("openneuro-py"):
        _run(["openneuro-py", "download", f"--dataset={accession}", f"--target-dir={dest}"])
    elif _have("aws"):
        _run(["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{accession}", str(dest)])
    else:
        sys.exit("Install openneuro-py or the AWS CLI for full downloads, or use --sample.")


def download_tuev(sample: bool) -> None:
    user = os.environ.get("TUH_USERNAME")
    pw = os.environ.get("TUH_PASSWORD")
    version = os.environ.get("TUEV_VERSION", "v2.0.1")
    if not user:
        sys.exit("Set TUH_USERNAME (and TUH_PASSWORD) after registering at "
                 "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")
    if not _have("rsync"):
        sys.exit("rsync is required for TUEV.")
    sub = "edf/eval/" if sample else ""
    src = f"{user}@{TUH_HOST}:{TUEV_PATH.format(version=version)}{sub}"
    tgt = DATA / "tuev" / sub
    tgt.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-auxvL", "--partial", src, str(tgt) + "/"]
    env = None
    if pw and _have("sshpass"):
        cmd = ["sshpass", "-e"] + cmd
        env = dict(os.environ, SSHPASS=pw)
    else:
        _log("sshpass not found or TUH_PASSWORD unset: rsync will prompt for the password")
    _run(cmd, env=env)


def download_mni(sample: bool) -> None:
    _log("MNI Open iEEG Atlas is a manual web download: https://mni-open-ieegatlas.research.mcgill.ca/ "
         "(accept terms, download the wake and sleep packages, unpack under data/mni_atlas/).")
    d = DATA / "mni_atlas"
    if d.exists():
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        _log(f"found {n} files under data/mni_atlas")


def download_bonn(sample: bool) -> None:
    _log("Bonn EEG: download Z.zip O.zip N.zip F.zip S.zip from https://www.upf.edu/web/ntsa/downloads "
         "and unpack into data/bonn/<set>/ (100 text files each, 4097 samples at 173.61 Hz).")
    for s in "ZONFS":
        d = DATA / "bonn" / s
        if d.exists():
            _log(f"set {s}: {len(list(d.glob('*.txt')))} files")


# --------------------------------------------------------------------------- #
def _parse_tuev_annotation(path: Path) -> List[dict]:
    """Parse a TUEV per-channel annotation file (comma or whitespace separated: channel, start, stop, label...)."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.startswith("#") or line.lower().startswith(("version", "channel")):
            continue
        toks = [t.strip() for t in line.replace(",", " ").split()]
        if len(toks) < 4:
            continue
        try:
            ch, on, off = toks[0], float(toks[1]), float(toks[2])
        except ValueError:
            continue
        lab = next((t.lower() for t in toks[3:] if t.lower() in TUEV_LABELS), toks[3].lower())
        rows.append(dict(channel=ch, onset_s=on, offset_s=off, label=lab))
    return rows


def build_manifests() -> None:
    man = DATA / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    events: List[dict] = []

    # TUEV
    for edf in sorted((DATA / "tuev").rglob("*.edf")):
        parts = edf.parts
        try:
            pid = parts[parts.index("edf") + 2]
        except (ValueError, IndexError):
            pid = edf.parent.name
        records.append(dict(dataset="tuev", record_id=edf.stem, subject_id=pid, path=str(edf), modality="scalp",
                            fs="", is_normal=0, vigilance_available=0))
        for ann in list(edf.parent.glob(edf.stem + ".rec")) + list(edf.parent.glob(edf.stem + "*.csv")):
            for r in _parse_tuev_annotation(ann):
                events.append(dict(dataset="tuev", record_id=edf.stem, annotator="tuev", **r))

    # OpenNeuro BIDS iEEG datasets
    for ds in sorted((DATA / "openneuro").glob("ds*")):
        for ev in sorted(ds.rglob("*_events.tsv")):
            rid = ev.name.replace("_events.tsv", "")
            sub = rid.split("_")[0]
            records.append(dict(dataset=ds.name, record_id=rid, subject_id=sub, path=str(ev.parent), modality="ieeg",
                                fs="", is_normal=0, vigilance_available=int(ds.name == "ds003876")))
            with open(ev, newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    try:
                        on = float(row.get("onset", "nan"))
                        du = float(row.get("duration", "0") or 0)
                    except ValueError:
                        continue
                    events.append(dict(dataset=ds.name, record_id=rid, channel=row.get("channel", ""), onset_s=on,
                                       offset_s=on + du, label=row.get("trial_type", row.get("eventType", "")),
                                       annotator=row.get("annotator", ds.name)))

    # MNI atlas: any signal file counts as a normal-tissue record
    for f in sorted((DATA / "mni_atlas").rglob("*")):
        if f.is_file() and f.suffix.lower() in (".edf", ".mat", ".npy", ".txt", ".csv"):
            records.append(dict(dataset="mni_atlas", record_id=f.stem, subject_id="", path=str(f), modality="ieeg",
                                fs="", is_normal=1, vigilance_available=int("sleep" in str(f).lower())))

    # Bonn
    for s in "ZONFS":
        for f in sorted((DATA / "bonn" / s).glob("*.txt")):
            records.append(dict(dataset="bonn", record_id=f"{s}_{f.stem}", subject_id="", path=str(f),
                                modality="scalp" if s in "ZO" else "ieeg", fs=173.61, is_normal=int(s in "ZO"),
                                vigilance_available=0))

    for name, rows, cols in (
        ("records.csv", records, ["dataset", "record_id", "subject_id", "path", "modality", "fs", "is_normal",
                                  "vigilance_available"]),
        ("events.csv", events, ["dataset", "record_id", "channel", "onset_s", "offset_s", "label", "annotator"]),
    ):
        with open(man / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _log(f"wrote {name}: {len(rows)} rows")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["openneuro", "tuev", "mni", "bonn", "all"])
    p.add_argument("--accession", default="ds003029", help="OpenNeuro accession for --dataset openneuro")
    p.add_argument("--sample", action="store_true", help="download a small subset only")
    p.add_argument("--build-manifests", action="store_true")
    args = p.parse_args(argv)

    if args.dataset == "all":
        for acc in ("ds003029", "ds003876"):
            download_openneuro(acc, args.sample)
        download_tuev(args.sample)
        download_mni(args.sample)
        download_bonn(args.sample)
    elif args.dataset == "openneuro":
        download_openneuro(args.accession, args.sample)
    elif args.dataset:
        {"tuev": download_tuev, "mni": download_mni, "bonn": download_bonn}[args.dataset](args.sample)
    if args.build_manifests:
        build_manifests()
    if not args.dataset and not args.build_manifests:
        p.print_help()


if __name__ == "__main__":
    main()
