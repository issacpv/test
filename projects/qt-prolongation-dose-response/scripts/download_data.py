#!/usr/bin/env python
"""Stage the data used by qt_dose.

Credentialed (PhysioNet CITI + DUA): MIMIC-IV-ECG, MIMIC-IV hosp/icu tables.
Credentials are read from PHYSIONET_USERNAME / PHYSIONET_PASSWORD.

Open: FAERS torsades-de-pointes counts via the openFDA API (no key needed for
low volumes; set OPENFDA_API_KEY for higher rate limits).

Examples
--------
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --tables
python scripts/download_data.py --dataset mimic-iv-ecg --out data/mimic-iv-ecg --sample
python scripts/download_data.py --dataset mimiciv-hosp  --out data/mimiciv-hosp --tables
python scripts/download_data.py --dataset faers --out data/external
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

PHYSIONET = "https://physionet.org/files"
OPENFDA_EVENT = "https://api.fda.gov/drug/event.json"

# Minimal tabular file lists (paths relative to each PhysioNet project root).
TABLES = {
    "mimic-iv-ecg": [
        "record_list.csv",
        "machine_measurements.csv",
        "waveform_note_links.csv",
    ],
    "mimiciv-hosp": [
        "hosp/emar.csv.gz",
        "hosp/emar_detail.csv.gz",
        "hosp/prescriptions.csv.gz",
        "hosp/pharmacy.csv.gz",
        "hosp/labevents.csv.gz",
        "hosp/d_labitems.csv.gz",
        "hosp/patients.csv.gz",
        "hosp/admissions.csv.gz",
        "hosp/diagnoses_icd.csv.gz",
    ],
    "mimiciv-icu": ["icu/inputevents.csv.gz", "icu/d_items.csv.gz"],
}
ROOTS = {
    "mimic-iv-ecg": f"{PHYSIONET}/mimic-iv-ecg/1.0/",
    "mimiciv-hosp": f"{PHYSIONET}/mimiciv/3.1/",
    "mimiciv-icu": f"{PHYSIONET}/mimiciv/3.1/",
}
# A handful of sample WFDB records for exercising the delineation code path.
SAMPLE_WFDB = [f"files/p1000/p10000032/s40689238/40689238.{ext}" for ext in ("hea", "dat")]

# Drugs whose torsades reporting we pull from FAERS as an external comparator.
QT_DRUGS_FAERS = [
    "sotalol", "dofetilide", "amiodarone", "haloperidol", "methadone",
    "azithromycin", "ondansetron", "citalopram", "quetiapine", "ciprofloxacin",
    # negative controls
    "aspirin", "metoprolol", "acetaminophen",
]


def _need_requests() -> None:
    if requests is None:
        sys.exit("pip install requests")


def _auth() -> tuple[str, str]:
    u, p = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
    if not u or not p:
        sys.exit("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD for credentialed downloads.")
    return u, p


def stream_download(url: str, dest: Path, auth: tuple[str, str] | None) -> bool:
    _need_requests()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest}")
        return True
    with requests.get(url, stream=True, auth=auth, timeout=180) as r:
        if r.status_code != 200:
            print(f"  [fail {r.status_code}] {url}")
            return False
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
    print(f"  [ok] {dest}")
    return True


def fetch_physionet(dataset: str, out: Path, tables: bool, sample: bool) -> None:
    root = ROOTS[dataset]
    auth = _auth()
    files: list[str] = []
    if tables:
        files += TABLES[dataset]
    if sample and dataset == "mimic-iv-ecg":
        files += SAMPLE_WFDB
    if not files:
        print("Nothing selected; pass --tables and/or --sample.")
        return
    for rel in files:
        stream_download(root + rel, out / Path(rel).name, auth)


def fetch_faers(out: Path) -> None:
    """Pull torsades / long-QT report counts per drug from openFDA (disproportionality)."""
    _need_requests()
    import json

    key = os.environ.get("OPENFDA_API_KEY")
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for drug in QT_DRUGS_FAERS:
        params = {
            "search": (
                f'patient.drug.medicinalproduct:"{drug}"'
                ' AND patient.reaction.reactionmeddrapt:"torsade de pointes"'
            ),
            "count": "patient.reaction.reactionmeddrapt.exact",
        }
        if key:
            params["api_key"] = key
        try:
            r = requests.get(OPENFDA_EVENT, params=params, timeout=60)
            n_tdp = 0
            if r.status_code == 200:
                for row in r.json().get("results", []):
                    if row.get("term", "").lower() == "torsade de pointes":
                        n_tdp = int(row.get("count", 0))
            # total reports for this drug (denominator)
            tot_params = {"search": f'patient.drug.medicinalproduct:"{drug}"', "limit": "1"}
            if key:
                tot_params["api_key"] = key
            rt = requests.get(OPENFDA_EVENT, params=tot_params, timeout=60)
            n_tot = int(rt.json().get("meta", {}).get("results", {}).get("total", 0)) if rt.status_code == 200 else 0
            results[drug] = {"tdp": n_tdp, "total": n_tot}
            print(f"  {drug}: tdp={n_tdp} total={n_tot}")
        except requests.RequestException as e:  # pragma: no cover - network
            print(f"  [warn] {drug}: {e}")
            results[drug] = {"tdp": None, "total": None}
    (out / "faers_tdp_counts.json").write_text(json.dumps(results, indent=2))
    print(f"  [ok] {out / 'faers_tdp_counts.json'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["mimic-iv-ecg", "mimiciv-hosp", "mimiciv-icu", "faers"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tables", action="store_true", help="fetch the tabular files")
    ap.add_argument("--sample", action="store_true", help="fetch a few sample WFDB records")
    args = ap.parse_args(argv)

    if args.dataset == "faers":
        fetch_faers(args.out)
    else:
        fetch_physionet(args.dataset, args.out, args.tables, args.sample)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
