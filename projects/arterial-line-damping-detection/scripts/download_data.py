#!/usr/bin/env python
"""Stage data for art_damping: synthetic records with embedded flush tests (open), VitalDB tracks
(open, REST API) and the credentialed MIMIC sources (PHYSIONET_USERNAME / PHYSIONET_PASSWORD).

Examples
--------
python scripts/download_data.py --simulate --out data/synthetic --n-records 20
python scripts/download_data.py --dataset vitaldb --out data/vitaldb --tracks --sample
python scripts/download_data.py --dataset mimic3wdb-matched --out data/mimic3wdb-matched/1.0 --sample
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

VITALDB_API = "https://api.vitaldb.net"
VITALDB_TRACKS = ["SNUADC/ART", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP", "Solar8000/ART_SBP",
                  "Solar8000/ART_DBP", "Solar8000/ART_MBP", "Solar8000/HR"]
PHYSIONET = "https://physionet.org/files"
CREDENTIALED = {
    "mimic3wdb-matched": {"base": f"{PHYSIONET}/mimic3wdb-matched/1.0/",
                          "sample": ["RECORDS", "RECORDS-waveforms", "RECORDS-numerics"], "sample_dir": "p00/p000020/"},
    "mimiciii": {"base": f"{PHYSIONET}/mimiciii/1.4/",
                 "files": ["D_ITEMS.csv.gz", "ICUSTAYS.csv.gz", "CHARTEVENTS.csv.gz", "INPUTEVENTS_MV.csv.gz", "PROCEDUREEVENTS_MV.csv.gz"],
                 "sample": ["D_ITEMS.csv.gz", "ICUSTAYS.csv.gz"]},
    "mimic4wdb": {"base": f"{PHYSIONET}/mimic4wdb/0.1.0/", "sample": ["RECORDS", "RECORDS-waveforms"]},
    "mimiciv-icu": {"base": f"{PHYSIONET}/mimiciv/3.1/icu/",
                    "files": ["d_items.csv.gz", "icustays.csv.gz", "chartevents.csv.gz", "inputevents.csv.gz", "procedureevents.csv.gz"],
                    "sample": ["d_items.csv.gz", "icustays.csv.gz"]},
}


def _need_requests() -> None:
    if requests is None:
        sys.exit("pip install requests")


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None = None) -> bool:
    _need_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
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


def physionet_auth() -> tuple[str, str] | None:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (u, p) if u and p else None


def simulate(out: Path, n_records: int, seed: int) -> None:
    from art_damping import transfer

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    fs = 125.0
    for k in range(n_records):
        fn, zeta = float(rng.uniform(5, 40)), float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
        s = transfer.synthetic_true_abp(fs=fs, n_beats=int(rng.integers(120, 200)), hr=float(rng.uniform(55, 110)), seed=int(rng.integers(1e6)))
        measured = transfer.apply_catheter_system(s.abp, fs, fn, zeta)
        # embed two flush tests through the same system (replace segments of the measured record)
        flush_idx = []
        for frac in (0.3, 0.7):
            i0 = int(frac * len(measured))
            seg_len = int(3.6 * fs)
            y = transfer.synthetic_flush(fs, fn, zeta, baseline=s.abp[i0:i0 + seg_len])
            measured[i0:i0 + len(y)] = y
            flush_idx.append(i0 + int(1.6 * fs))
        np.savez(out / f"rec{k:03d}.npz", abp=measured.astype(np.float32), fs=fs, fn=fn, zeta=zeta,
                 adequacy=transfer.gardner_adequacy(fn, zeta), flush_release_idx=np.asarray(flush_idx), true_abp=s.abp.astype(np.float32))
        print(f"  [ok] rec{k:03d}: fn={fn:.1f} Hz zeta={zeta:.2f} ({transfer.gardner_adequacy(fn, zeta)})")


def download_vitaldb(out: Path, tracks: bool, sample: bool, max_cases: int | None) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    stream_download(f"{VITALDB_API}/cases", out / "cases.csv")
    stream_download(f"{VITALDB_API}/trks", out / "trks.csv")
    if not tracks:
        return
    trks = pd.read_csv(out / "trks.csv")
    has_art = set(trks[trks["tname"] == "SNUADC/ART"]["caseid"])
    has_nibp = set(trks[trks["tname"] == "Solar8000/NIBP_SBP"]["caseid"])
    eligible = sorted(has_art & has_nibp)
    print(f"  {len(eligible)} cases with ART waveform and NIBP")
    eligible = eligible[:5] if sample else (eligible[:max_cases] if max_cases else eligible)
    for cid in eligible:
        sub = trks[(trks["caseid"] == cid) & (trks["tname"].isin(VITALDB_TRACKS))]
        for _, row in sub.iterrows():
            stream_download(f"{VITALDB_API}/{row['tid']}", out / "tracks" / str(cid) / (row["tname"].replace("/", "_") + ".csv"))


def download_physionet(name: str, out: Path, sample: bool) -> None:
    src = CREDENTIALED[name]
    auth = physionet_auth()
    if auth is None:
        print(f"{name} is credentialed. Complete CITI training + DUA at "
              f"{src['base'].replace('/files/', '/content/')} then export PHYSIONET_USERNAME/PASSWORD.")
        return
    files = src.get("sample", []) if sample or "files" not in src else src["files"]
    for rel in files:
        stream_download(src["base"] + rel, out / rel, auth)
    if sample and src.get("sample_dir"):
        print(f"  sample patient folder: mirror {src['base']}{src['sample_dir']} with wget (see data/README.md)")


def verify(root: Path) -> None:
    syn = list((root / "synthetic").glob("*.npz")) if (root / "synthetic").exists() else []
    print(f"synthetic records: {len(syn)}")
    v = root / "vitaldb"
    n_tracks = len(list((v / "tracks").glob("*/*.csv"))) if (v / "tracks").exists() else 0
    print(f"vitaldb: trks.csv={'yes' if (v / 'trks.csv').exists() else 'NO'} track files={n_tracks}")
    for name, sub in {"mimic3wdb-matched": "mimic3wdb-matched/1.0", "mimiciii": "mimiciii/1.4",
                      "mimic4wdb": "mimic4wdb/0.1.0", "mimiciv-icu": "mimiciv/3.1/icu"}.items():
        d = root / sub
        print(f"{name:20s} files={len(list(d.rglob('*'))) if d.exists() else 0}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["vitaldb"] + list(CREDENTIALED))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--n-records", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracks", action="store_true", help="vitaldb: also download per-case tracks")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
    elif args.simulate:
        simulate(args.out, args.n_records, args.seed)
    elif args.dataset == "vitaldb":
        download_vitaldb(args.out, args.tracks, args.sample, args.max_cases)
    elif args.dataset:
        download_physionet(args.dataset, args.out, args.sample)
    else:
        ap.error("one of --dataset, --simulate, --verify is required")


if __name__ == "__main__":
    main()
