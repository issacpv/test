#!/usr/bin/env python
"""Stage data for pva_detect: simulated labelled waveforms (open) and the credentialed charted ICU
databases (HiRID, eICU-CRD, MIMIC-IV icu) with credentials from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Examples
--------
python scripts/download_data.py --simulate --out data/simulated --duration 600 --seeds 0 1 2
python scripts/download_data.py --dataset hirid --out data/hirid/1.1.1 --sample
python scripts/download_data.py --dataset eicu --out data/eicu/2.0
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
CREDENTIALED = {
    "hirid": {
        "base": f"{PHYSIONET}/hirid/1.1.1/",
        "files": ["reference_data.tar.gz", "raw_stage/observation_tables_parquet.tar.gz", "raw_stage/pharma_records_parquet.tar.gz"],
        "sample": ["reference_data.tar.gz"],
    },
    "eicu": {
        "base": f"{PHYSIONET}/eicu-crd/2.0/",
        "files": ["patient.csv.gz", "apachePatientResult.csv.gz", "respiratoryCharting.csv.gz", "respiratoryCare.csv.gz",
                  "vitalPeriodic.csv.gz", "infusionDrug.csv.gz", "hospital.csv.gz"],
        "sample": ["patient.csv.gz", "hospital.csv.gz"],
    },
    "mimiciv-icu": {
        "base": f"{PHYSIONET}/mimiciv/3.1/icu/",
        "files": ["d_items.csv.gz", "icustays.csv.gz", "chartevents.csv.gz", "procedureevents.csv.gz", "inputevents.csv.gz"],
        "sample": ["d_items.csv.gz", "icustays.csv.gz"],
    },
}


def physionet_auth() -> tuple[str, str] | None:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (u, p) if u and p else None


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None = None) -> bool:
    if requests is None:
        sys.exit("pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=300) as r:
        if r.status_code in (401, 403):
            print(f"  [{r.status_code}] {url}: credentials rejected or DUA not signed")
            return False
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(dest)
    print(f"  [ok] {dest}")
    return True


def simulate(out: Path, duration: float, seeds: list[int]) -> None:
    from pva_detect import lungsim

    out.mkdir(parents=True, exist_ok=True)
    for name, base in lungsim.SCENARIOS.items():
        for seed in seeds:
            cfg = lungsim.SimConfig(**{**base.__dict__, "duration_s": duration, "seed": seed})
            r = lungsim.simulate(cfg)
            np.savez(out / f"{name}_seed{seed}.npz", t=r.t, flow=r.flow, paw=r.paw, volume=r.volume, pmus=r.pmus, fs=r.fs,
                     breaths=r.breaths.to_records(index=False), efforts=r.efforts.to_records(index=False),
                     asynchrony_index=r.asynchrony_index(), scenario=name)
            print(f"  [ok] {name} seed={seed}: {len(r.breaths)} breaths, AI={r.asynchrony_index():.1f}%")


def download(name: str, out: Path, sample: bool) -> None:
    src = CREDENTIALED[name]
    auth = physionet_auth()
    if auth is None:
        print(f"{name} is credentialed. Complete CITI training + DUA at "
              f"{src['base'].replace('/files/', '/content/')} then export PHYSIONET_USERNAME/PASSWORD.")
        return
    for rel in (src["sample"] if sample else src["files"]):
        stream_download(src["base"] + rel, out / rel, auth)
    if name == "hirid":
        print("  extract with: tar -xzf data/hirid/1.1.1/reference_data.tar.gz -C data/hirid/1.1.1/ (and the parquet tarballs)")


def verify(root: Path) -> None:
    sim = list((root / "simulated").glob("*.npz")) if (root / "simulated").exists() else []
    print(f"simulated scenarios: {len(sim)}")
    for name, sub in {"hirid": "hirid/1.1.1", "eicu": "eicu/2.0", "mimiciv-icu": "mimiciv/3.1/icu"}.items():
        d = root / sub
        for rel in CREDENTIALED[name]["files"]:
            print(f"{name:12s} {rel:45s} {'present' if (d / rel).exists() else 'MISSING'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(CREDENTIALED))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
    elif args.simulate:
        simulate(args.out, args.duration, args.seeds)
    elif args.dataset:
        download(args.dataset, args.out, args.sample)
    else:
        ap.error("one of --dataset, --simulate, --verify is required")


if __name__ == "__main__":
    main()
