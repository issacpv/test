#!/usr/bin/env python
"""Downloader for radiology-delay-ed-disposition.

Credentialed PhysioNet resources use wget with PHYSIONET_USERNAME /
PHYSIONET_PASSWORD from the environment.  Open demos are fetched without
credentials, and --synthetic writes schema-matching synthetic tables.

Examples
--------
python scripts/download_data.py --synthetic
python scripts/download_data.py --sample
python scripts/download_data.py --note --ed --hosp --cxr-labels   # credentialed
python scripts/download_data.py --check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PN = "https://physionet.org/files"


def _wget(url: str, dest_dir: Path, credentialed: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-N", "-c", "-q", "--show-progress", "-P", str(dest_dir)]
    if credentialed:
        user, pwd = os.environ.get("PHYSIONET_USERNAME"), os.environ.get("PHYSIONET_PASSWORD")
        if not user or not pwd:
            print("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD for credentialed resources.", file=sys.stderr)
            return 2
        cmd += ["--user", user, "--password", pwd]
    cmd.append(url)
    return subprocess.call(cmd)


def _fetch(base: str, files: list[str], dest: Path, credentialed: bool) -> None:
    for f in files:
        if _wget(f"{PN}/{base}/{f}", dest / Path(f).parent, credentialed) != 0:
            print(f"failed: {base}/{f}", file=sys.stderr)
            return


def fetch_sample() -> None:
    _fetch("mimic-iv-ed-demo/2.2", ["ed/edstays.csv.gz", "ed/triage.csv.gz", "ed/diagnosis.csv.gz"], DATA / "mimic-iv-ed-demo", False)
    _fetch("mimic-iv-demo/2.2", ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz"], DATA / "mimic-iv-demo", False)


def write_synthetic(n_subjects: int = 300, seed: int = 0) -> None:
    """Schema-matching synthetic radiology / radiology_detail / edstays / triage tables."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    out = DATA / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    exams = ["CHEST (PORTABLE AP)", "CT HEAD W/O CONTRAST", "CT ABD & PELVIS WITH CONTRAST", "CTA CHEST W&W/O C&RECONS, NON-CORONARY", "US ABD LIMIT, SINGLE ORGAN", "MR HEAD W & W/O CONTRAST", "ANKLE (AP, MORTISE & LAT) RIGHT"]
    t0 = pd.Timestamp("2150-01-01")
    ed_rows, tri_rows, rad_rows, det_rows = [], [], [], []
    stay_id, note_no = 30000000, 0
    for s in range(10000000, 10000000 + n_subjects):
        for _ in range(int(rng.integers(1, 3))):
            intime = t0 + pd.Timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
            los_h = float(rng.gamma(3, 2))
            outtime = intime + pd.Timedelta(hours=los_h)
            disp = "ADMITTED" if rng.random() < 0.35 else "HOME"
            ed_rows.append({"subject_id": s, "hadm_id": s * 10 if disp == "ADMITTED" else "", "stay_id": stay_id, "intime": intime, "outtime": outtime, "gender": "F", "race": "WHITE", "arrival_transport": "WALK IN", "disposition": disp})
            tri_rows.append({"subject_id": s, "stay_id": stay_id, "temperature": 98.4, "heartrate": 82, "resprate": 16, "o2sat": 97, "sbp": 124, "dbp": 78, "pain": 4, "acuity": int(rng.integers(1, 6)), "chiefcomplaint": "abd pain"})
            for k in range(int(rng.integers(0, 3))):
                exam = exams[int(rng.integers(0, len(exams)))]
                chart = intime + pd.Timedelta(hours=float(rng.uniform(0.3, max(0.4, los_h * 0.8))))
                tat_h = float(rng.gamma(2, 0.8)) * (1.6 if chart.hour >= 23 or chart.hour < 7 else 1.0)
                store = chart + pd.Timedelta(hours=tat_h)
                note_id = f"{s}-RR-{note_no}"
                rad_rows.append({"note_id": note_id, "subject_id": s, "hadm_id": s * 10 if disp == "ADMITTED" else "", "note_type": "RR", "note_seq": note_no, "charttime": chart, "storetime": store, "text": f"EXAMINATION: {exam}\nFINDINGS: no acute process. " + ("Wet read discussed with ED team." if rng.random() < 0.1 else "")})
                det_rows += [{"note_id": note_id, "subject_id": s, "field_name": "exam_name", "field_value": exam, "field_ordinal": 1}, {"note_id": note_id, "subject_id": s, "field_name": "cpt_code", "field_value": "71046", "field_ordinal": 1}]
                if rng.random() < 0.05:
                    add_id = f"{s}-AR-{note_no}"
                    rad_rows.append({"note_id": add_id, "subject_id": s, "hadm_id": "", "note_type": "AR", "note_seq": note_no, "charttime": chart, "storetime": store + pd.Timedelta(hours=float(rng.gamma(2, 6))), "text": "ADDENDUM: additional finding."})
                    det_rows.append({"note_id": add_id, "subject_id": s, "field_name": "parent_note_id", "field_value": note_id, "field_ordinal": 1})
                note_no += 1
            stay_id += 1
    pd.DataFrame(ed_rows).to_csv(out / "edstays.csv", index=False)
    pd.DataFrame(tri_rows).to_csv(out / "triage.csv", index=False)
    pd.DataFrame(rad_rows).to_csv(out / "radiology.csv", index=False)
    pd.DataFrame(det_rows).to_csv(out / "radiology_detail.csv", index=False)
    print(f"synthetic tables written to {out}: {len(ed_rows)} stays, {len(rad_rows)} reports")


def check() -> None:
    parts = {
        "note radiology": DATA / "mimic-iv-note" / "note" / "radiology.csv.gz",
        "note radiology_detail": DATA / "mimic-iv-note" / "note" / "radiology_detail.csv.gz",
        "ED edstays": DATA / "mimic-iv-ed" / "ed" / "edstays.csv.gz",
        "hosp patients": DATA / "mimiciv" / "hosp" / "patients.csv.gz",
        "cxr metadata": DATA / "mimic-cxr-jpg" / "mimic-cxr-2.0.0-metadata.csv.gz",
        "synthetic radiology": DATA / "synthetic" / "radiology.csv",
    }
    for k, p in parts.items():
        print(f"{'OK ' if p.exists() else '-- '} {k}: {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--note", action="store_true")
    ap.add_argument("--ed", action="store_true")
    ap.add_argument("--hosp", action="store_true")
    ap.add_argument("--cxr-labels", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.synthetic:
        write_synthetic()
    if a.sample:
        fetch_sample()
    if a.note:
        _fetch("mimic-iv-note/2.2", ["note/radiology.csv.gz", "note/radiology_detail.csv.gz"], DATA / "mimic-iv-note", True)
    if a.ed:
        _fetch("mimic-iv-ed/2.2", ["ed/edstays.csv.gz", "ed/triage.csv.gz", "ed/diagnosis.csv.gz"], DATA / "mimic-iv-ed", True)
    if a.hosp:
        _fetch("mimiciv/3.1", ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz"], DATA / "mimiciv", True)
    if a.cxr_labels:
        _fetch("mimic-cxr-jpg/2.1.0", ["mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-chexpert.csv.gz"], DATA / "mimic-cxr-jpg", True)
    if a.check or not any([a.synthetic, a.sample, a.note, a.ed, a.hosp, a.cxr_labels]):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
