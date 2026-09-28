#!/usr/bin/env python3
"""Download the open parts of the BioShift domains and build fixed split manifests.

The benchmark composes four sibling projects; this script is self-contained
and covers (a) open PhysioNet resources with a ``--sample`` mode, (b) the
CODE-15% Zenodo record, and (c) instructions/stubs for the registration and
credentialed resources.  Credentials are read from the environment only.

Examples
--------
    python scripts/download_data.py --domain ptbxl --sample
    python scripts/download_data.py --domain chbmit --sample
    python scripts/download_data.py --domain siena --sample
    python scripts/download_data.py --domain code15 --sample          # Zenodo API, one HDF5 shard
    PHYSIONET_USER=... PHYSIONET_PASS=... python scripts/download_data.py --domain mimic_iv --sample
    python scripts/download_data.py --make-manifests                  # from data/cache/*.npz
    python scripts/download_data.py --dry-run                         # run the harness on synthetic data
"""
from __future__ import annotations

import argparse
import json
import os
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

PHYSIONET_FILES = "https://physionet.org/files"
OPEN_PHYSIONET = {
    "ptbxl": ("ptb-xl/1.0.3/", ["ptbxl_database.csv", "scp_statements.csv", "records500/00000/00001_hr.hea", "records500/00000/00001_hr.dat"]),
    "chapman": ("ecg-arrhythmia/1.0.0/", ["ConditionNames_SNOMED-CT.csv", "RECORDS", "WFDBRecords/01/010/JS00001.hea", "WFDBRecords/01/010/JS00001.mat"]),
    "ningbo": ("ecg-arrhythmia/1.0.0/", ["ConditionNames_SNOMED-CT.csv", "RECORDS"]),
    "georgia": ("challenge-2021/1.0.3/", ["training/georgia/g1/E00001.hea", "training/georgia/g1/E00001.mat", "training/georgia/RECORDS"]),
    "cpsc": ("challenge-2021/1.0.3/", ["training/cpsc_2018/g1/A0001.hea", "training/cpsc_2018/g1/A0001.mat"]),
    "chbmit": ("chbmit/1.0.0/", ["RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO", "chb01/chb01-summary.txt", "chb01/chb01_03.edf"]),
    "siena": ("siena-scalp-eeg/1.0.0/", ["subject_info.csv", "PN00/Seizures-list-PN00.txt", "PN00/PN00-1.edf"]),
}
CREDENTIALED_PHYSIONET = {
    "mimic_iv": ("mimiciv/3.1/", ["icu/icustays.csv.gz", "hosp/patients.csv.gz"]),
    "eicu": ("eicu-crd/2.0/", ["patient.csv.gz"]),
    "hirid": ("hirid/1.1.1/", ["reference_data.tar.gz"]),
}
ZENODO_CODE15 = "4916206"
ZENODO_HELSINKI_QUERY = "A dataset of neonatal EEG recordings with seizure annotations"


def _log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def _download(url: str, dest: Path, auth: Optional[tuple[str, str]] = None, chunk: int = 1 << 20) -> None:
    """Resumable HTTP download with optional basic auth (stdlib only)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    if auth:
        import base64

        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if existing and resp.status != 206:
                existing = 0
            with open(dest, "ab" if existing else "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            return
        raise
    _log(f"saved {dest.relative_to(ROOT)}")


def _wget_mirror(url: str, dest: Path, auth: Optional[tuple[str, str]] = None) -> None:
    if shutil.which("wget") is None:
        sys.exit("wget not found; install it or use --sample")
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", f"--cut-dirs={url.replace('https://physionet.org/', '').count('/')}", "-P", str(dest)]
    if auth:
        cmd += ["--user", auth[0], "--password", auth[1]]
    cmd.append(url)
    _log(" ".join(c if c != (auth[1] if auth else None) else "***" for c in cmd))
    subprocess.call(cmd)


def physionet(domain: str, sample: bool) -> None:
    if domain in OPEN_PHYSIONET:
        rel, files = OPEN_PHYSIONET[domain]
        auth = None
    else:
        rel, files = CREDENTIALED_PHYSIONET[domain]
        user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
        if not (user and pw):
            sys.exit(f"{domain} is credentialed: export PHYSIONET_USER and PHYSIONET_PASS after completing CITI training and signing the DUA.")
        auth = (user, pw)
    base = f"{PHYSIONET_FILES}/{rel}"
    dest = DATA / domain
    if sample:
        for f in files:
            _download(base + f, dest / f, auth)
    else:
        _wget_mirror(base, dest, auth)


def zenodo(domain: str, sample: bool) -> None:
    api = "https://zenodo.org/api/records/"
    if domain == "code15":
        url = api + ZENODO_CODE15
    else:
        url = api + "?q=" + urllib.request.quote(f'title:"{ZENODO_HELSINKI_QUERY}"') + "&size=1"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    rec = payload if domain == "code15" else payload["hits"]["hits"][0]
    files = rec.get("files", [])
    _log(f"Zenodo record {rec.get('id')}: {len(files)} files")
    dest = DATA / domain
    picked = files
    if sample:
        keep = [f for f in files if f["key"].endswith((".csv", ".txt", ".md"))] + [f for f in files if f["key"].endswith((".hdf5", ".h5", ".edf", ".zip"))][:1]
        picked = keep
    for f in picked:
        link = f["links"].get("self") or f["links"].get("download")
        _download(link, dest / f["key"])


def notes(domain: str) -> None:
    msgs = {
        "aumcdb": "AmsterdamUMCdb: request access and sign the end-user licence at https://amsterdammedicaldatascience.nl/amsterdamumcdb/ ; place the CSVs under data/aumcdb/.",
        "tusz": "TUSZ v2.0.3: fill the TUH EEG data-use form at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ ; then: rsync -auxvL nedc-eeg@www.isip.piconepress.com:data/tuh_eeg_seizure/v2.0.3/ data/tusz/ (password from e-mail).",
        "echonet_dynamic": "EchoNet-Dynamic: accept the Stanford AIMI research use agreement at https://echonet.github.io/dynamic/ ; download the archive and extract to data/echonet_dynamic/ (Videos/, FileList.csv, VolumeTracings.csv).",
        "echonet_pediatric": "EchoNet-Pediatric: same route at https://echonet.github.io/pediatric/ ; extract to data/echonet_pediatric/ (A4C/, PSAX/ with FileList.csv each).",
        "camus": "CAMUS: register at https://www.creatis.insa-lyon.fr/Challenge/camus/ and download training + testing sets to data/camus/.",
    }
    _log(msgs[domain])


def make_manifests() -> None:
    import numpy as np

    from bioshift.adapters import write_split_manifest
    from bioshift.spec import default_benchmark

    b = default_benchmark()
    n = 0
    for task in b.tasks.values():
        for dom in [d for d in b.domains.values() if d.modality == task.modality]:
            npz = DATA / "cache" / task.id / f"{dom.id}.npz"
            if not npz.exists():
                continue
            groups = np.load(npz, allow_pickle=False)["groups"]
            out = ROOT / "manifests" / task.id / f"{dom.id}_splits.csv"
            write_split_manifest(groups, out, seed=0)
            n += 1
            _log(f"manifest {out.relative_to(ROOT)} ({len(np.unique(groups))} groups)")
    _log(f"{n} manifests written (official splits should overwrite these where they exist; see manifests/README.md)")


def dry_run() -> None:
    from bioshift import SyntheticAdapter, default_benchmark, leaderboard, run_benchmark, shift_loss_regression

    b = default_benchmark()
    ad = SyntheticAdapter(b, n_per_domain=500, n_groups=50, d=10)
    cells = [b.cells_for(task=t)[0] for t in b.tasks]
    res = run_benchmark(b, ad, cells=cells, n_boot=20, verbose=True)
    out = ROOT / "outputs" / "dry_run"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_long.csv", index=False)
    leaderboard(res).to_csv(out / "leaderboard.csv", index=False)
    print(leaderboard(res).to_string())
    print(shift_loss_regression(res).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=sorted({*OPEN_PHYSIONET, *CREDENTIALED_PHYSIONET, "code15", "helsinki", "aumcdb", "tusz", "echonet_dynamic", "echonet_pediatric", "camus"}))
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--make-manifests", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.domain in OPEN_PHYSIONET or args.domain in CREDENTIALED_PHYSIONET:
        physionet(args.domain, args.sample)
    elif args.domain in ("code15", "helsinki"):
        zenodo(args.domain, args.sample)
    elif args.domain:
        notes(args.domain)
    if args.make_manifests:
        make_manifests()
    if args.dry_run:
        dry_run()
    if not (args.domain or args.make_manifests or args.dry_run):
        ap.print_help()


if __name__ == "__main__":
    main()
