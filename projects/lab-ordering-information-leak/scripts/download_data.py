#!/usr/bin/env python
"""Stage the data for lab_leak: generate the open semi-synthetic cohorts and download the
credentialed MIMIC-IV / eICU-CRD files (credentials from PHYSIONET_USERNAME / PHYSIONET_PASSWORD).

Examples
--------
python scripts/download_data.py --simulate --out data/synthetic --n-stays 2000 --gammas 0 1 2
python scripts/download_data.py --dataset mimiciv --out data/mimiciv/3.1
python scripts/download_data.py --dataset eicu --out data/eicu/2.0 --sample   # small tables only
python scripts/download_data.py --verify --out data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
SOURCES = {
    "mimiciv": {
        "base": f"{PHYSIONET}/mimiciv/3.1/",
        "files": ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/d_labitems.csv.gz",
                  "hosp/labevents.csv.gz", "hosp/poe.csv.gz", "hosp/poe_detail.csv.gz",
                  "icu/icustays.csv.gz", "icu/d_items.csv.gz"],
        "sample": ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/d_labitems.csv.gz",
                   "icu/icustays.csv.gz", "icu/d_items.csv.gz"],
    },
    "eicu": {
        "base": f"{PHYSIONET}/eicu-crd/2.0/",
        "files": ["patient.csv.gz", "hospital.csv.gz", "lab.csv.gz", "apachePatientResult.csv.gz"],
        "sample": ["patient.csv.gz", "hospital.csv.gz"],
    },
}


def physionet_auth() -> tuple[str, str] | None:
    user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    return (user, pwd) if user and pwd else None


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None) -> bool:
    if requests is None:
        sys.exit("pip install requests")
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


def download(name: str, out: Path, sample: bool) -> None:
    src = SOURCES[name]
    auth = physionet_auth()
    if auth is None:
        print(f"{name} is credentialed. Complete CITI training + DUA at "
              f"{src['base'].replace('/files/', '/content/')} then export PHYSIONET_USERNAME/PASSWORD.")
        return
    for rel in (src["sample"] if sample else src["files"]):
        stream_download(src["base"] + rel, out / rel, auth)


def simulate(out: Path, n_stays: int, gammas: list[float], site_shifts: list[float], seed: int) -> None:
    from lab_leak.simulate import SimConfig, simulate_cohort

    out.mkdir(parents=True, exist_ok=True)
    for g in gammas:
        for s in site_shifts:
            long, stays = simulate_cohort(SimConfig(n_stays=n_stays, gamma=g, site_shift=s, seed=seed))
            stem = f"gamma{g:g}_site{s:g}"
            for df, kind in ((long, "long"), (stays, "stays")):
                try:
                    df.to_parquet(out / f"{kind}_{stem}.parquet", index=False)
                except (ImportError, ValueError):
                    df.to_csv(out / f"{kind}_{stem}.csv", index=False)
            print(f"  [ok] {stem}: {len(long)} measurements, {len(stays)} stays, prevalence {stays['y'].mean():.2f}")


def verify(root: Path) -> None:
    for name, src in SOURCES.items():
        d = root / {"mimiciv": "mimiciv/3.1", "eicu": "eicu/2.0"}[name]
        for rel in src["files"]:
            p = d / rel
            print(f"{name:8s} {rel:32s} {'present' if p.exists() else 'MISSING'}"
                  + (f"  {p.stat().st_size / 1e9:.2f} GB" if p.exists() else ""))
    syn = list((root / "synthetic").glob("long_*")) if (root / "synthetic").exists() else []
    print(f"synthetic cohorts: {len(syn)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(SOURCES))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample", action="store_true", help="only the small dictionary/patient tables")
    ap.add_argument("--simulate", action="store_true", help="generate semi-synthetic cohorts (open)")
    ap.add_argument("--n-stays", type=int, default=2000)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--site-shifts", type=float, nargs="+", default=[0.0, -1.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.out)
    elif args.simulate:
        simulate(args.out, args.n_stays, args.gammas, args.site_shifts, args.seed)
    elif args.dataset:
        download(args.dataset, args.out, args.sample)
    else:
        ap.error("one of --dataset, --simulate or --verify is required")


if __name__ == "__main__":
    main()
