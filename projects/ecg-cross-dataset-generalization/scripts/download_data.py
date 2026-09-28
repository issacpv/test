#!/usr/bin/env python
"""Download the public ECG sources used by ecg_xgen and stage credentialed ones.

Open sources (no login): PTB-XL, ecg-arrhythmia (Chapman/Ningbo), Challenge-2021
training sets (Georgia, CPSC-2018, CPSC-Extra), CODE-15% (Zenodo REST API).
Credentialed sources (PhysioNet CITI + DUA): MIMIC-IV-ECG, MIMIC-IV hosp.
Credentials are read from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Examples
--------
python scripts/download_data.py --dataset ptbxl --out data/ptbxl --sample
python scripts/download_data.py --dataset code15 --out data/code15 --sample
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
ZENODO_CODE15 = "https://zenodo.org/api/records/4916206"

OPEN_SOURCES = {
    "ptbxl": f"{PHYSIONET}/ptb-xl/1.0.3/",
    "ecg-arrhythmia": f"{PHYSIONET}/ecg-arrhythmia/1.0.0/",
    "challenge2021": f"{PHYSIONET}/challenge-2021/1.0.3/",
}
CREDENTIALED_SOURCES = {
    "mimic-iv-ecg": f"{PHYSIONET}/mimic-iv-ecg/1.0/",
    "mimiciv-hosp": f"{PHYSIONET}/mimiciv/3.1/hosp/",
    "mimic-iv-ecg-ext-icd": f"{PHYSIONET}/mimic-iv-ecg-ext-icd-labels/1.0.1/",
}
# Files worth fetching first for a sample / metadata-only run.
SAMPLE_FILES = {
    "ptbxl": ["ptbxl_database.csv", "scp_statements.csv"]
    + [f"records500/00000/{i:05d}_hr.{ext}" for i in range(1, 21) for ext in ("hea", "dat")],
    "ecg-arrhythmia": ["ConditionNames_SNOMED-CT.csv", "RECORDS"]
    + [f"WFDBRecords/01/010/JS{i:05d}.{ext}" for i in range(1, 21) for ext in ("hea", "mat")],
    "challenge2021": ["dx_mapping_scored.csv", "dx_mapping_unscored.csv"]
    + [f"training/georgia/g1/E{i:05d}.{ext}" for i in range(1, 11) for ext in ("hea", "mat")]
    + [f"training/cpsc_2018/g1/A{i:04d}.{ext}" for i in range(1, 11) for ext in ("hea", "mat")],
    "mimic-iv-ecg": ["record_list.csv", "machine_measurements.csv", "waveform_note_links.csv"],
    "mimiciv-hosp": ["patients.csv.gz", "admissions.csv.gz", "d_labitems.csv.gz", "labevents.csv.gz"],
    "mimic-iv-ecg-ext-icd": ["records_w_diag_icd10.csv"],
}


def _need_requests() -> None:
    if requests is None:
        sys.exit("pip install requests")


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None = None, expected_md5: str | None = None) -> bool:
    """Stream ``url`` to ``dest``; skip when present and (if given) MD5 matches."""
    _need_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_md5 is None or md5sum(dest) == expected_md5):
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=120) as r:
        if r.status_code == 401:
            print(f"  [401] {url}: credentials rejected (check PHYSIONET_USERNAME/PASSWORD and DUA)")
            return False
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(dest)
    if expected_md5 and md5sum(dest) != expected_md5:
        print(f"  [md5 mismatch] {dest}")
        return False
    print(f"  [ok] {dest}")
    return True


def wget_recursive(url: str, out: Path, auth: tuple[str, str] | None = None) -> int:
    """Mirror a PhysioNet directory with wget (resumable). Returns the exit code."""
    if shutil.which("wget") is None:
        sys.exit("wget not found; install it or use --sample to fetch individual files via HTTPS")
    cut = len([p for p in url.replace(PHYSIONET + "/", "").split("/") if p])
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", f"--cut-dirs={cut}", "-P", str(out), url]
    if auth:
        cmd[1:1] = ["--user", auth[0], "--password", auth[1]]
    print("  $", " ".join(c if c != (auth[1] if auth else None) else "****" for c in cmd))
    return subprocess.call(cmd)


def physionet_auth() -> tuple[str, str] | None:
    user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    if not user or not pwd:
        return None
    return user, pwd


def download_physionet(name: str, out: Path, sample: bool, waveforms: bool) -> None:
    credentialed = name in CREDENTIALED_SOURCES
    base = CREDENTIALED_SOURCES.get(name) or OPEN_SOURCES[name]
    auth = physionet_auth() if credentialed else None
    if credentialed and auth is None:
        print(
            f"{name} is credentialed. Complete CITI training + DUA at {base.replace('/files/', '/content/')}\n"
            "then `export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=...` and re-run."
        )
        return
    print(f"== {name} -> {out}")
    if sample or (credentialed and not waveforms and name == "mimic-iv-ecg"):
        for rel in SAMPLE_FILES[name]:
            stream_download(base + rel, out / rel, auth=auth)
        return
    if name == "mimiciv-hosp":
        for rel in SAMPLE_FILES[name]:
            stream_download(base + rel, out / rel, auth=auth)
        return
    if name == "challenge2021":
        for rel in ("dx_mapping_scored.csv", "dx_mapping_unscored.csv"):
            stream_download(base + rel, out / rel)
        for src in ("georgia", "cpsc_2018", "cpsc_2018_extra"):
            wget_recursive(base + f"training/{src}/", out / "training" / src, auth=None)
        return
    wget_recursive(base, out, auth=auth)


def zenodo_files(record_api: str) -> Iterable[dict]:
    """Yield file entries {key, size, checksum, url} for a Zenodo record (handles pagination of versions)."""
    _need_requests()
    r = requests.get(record_api, timeout=60)
    r.raise_for_status()
    meta = r.json()
    for f in meta.get("files", []):
        yield {
            "key": f["key"],
            "size": f.get("size"),
            "checksum": f.get("checksum", "").replace("md5:", ""),
            "url": f["links"]["self"],
        }


def download_code15(out: Path, sample: bool) -> None:
    print(f"== CODE-15% (Zenodo 4916206) -> {out}")
    files = list(zenodo_files(ZENODO_CODE15))
    if not files:
        sys.exit("Zenodo API returned no files; check network / record id")
    wanted = [f for f in files if f["key"] == "exams.csv"]
    parts = sorted(f for f in files if f["key"].endswith(".hdf5"))
    wanted += parts[:1] if sample else parts
    total_gb = sum((f["size"] or 0) for f in wanted) / 1e9
    print(f"  {len(wanted)} files, {total_gb:.1f} GB")
    for f in wanted:
        stream_download(f["url"], out / f["key"], expected_md5=f["checksum"] or None)


def verify(root: Path) -> None:
    checks = {
        "ptbxl": ("ptbxl_database.csv", "records500/**/*_hr.hea"),
        "ecg-arrhythmia": ("ConditionNames_SNOMED-CT.csv", "WFDBRecords/**/*.hea"),
        "challenge2021": ("dx_mapping_scored.csv", "training/**/*.hea"),
        "code15": ("exams.csv", "exams_part*.hdf5"),
        "mimic-iv-ecg": ("record_list.csv", "files/**/*.hea"),
    }
    for name, (meta, pattern) in checks.items():
        d = root / name
        n = len(list(d.glob(pattern))) if d.exists() else 0
        has_meta = (d / meta).exists()
        print(f"{name:16s} metadata={'yes' if has_meta else 'NO ':3s} records={n}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(OPEN_SOURCES) + ["code15"] + list(CREDENTIALED_SOURCES))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true", help="download only a handful of records / one HDF5 part")
    ap.add_argument("--waveforms", action="store_true", help="MIMIC-IV-ECG: also mirror the 90 GB WFDB tree")
    ap.add_argument("--verify", action="store_true", help="count records under --out and exit")
    args = ap.parse_args(argv)

    if args.verify:
        verify(args.out)
        return
    if args.dataset is None:
        ap.error("--dataset is required unless --verify")
    if args.dataset == "code15":
        download_code15(args.out, args.sample)
    else:
        download_physionet(args.dataset, args.out, args.sample, args.waveforms)


if __name__ == "__main__":
    main()
