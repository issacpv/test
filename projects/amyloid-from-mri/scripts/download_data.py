#!/usr/bin/env python3
"""Downloader for the open-registration part of this project (OASIS-3 on NITRC-IR/XNAT)
and instructions/stubs for the application-only parts (ADNI and A4 via LONI IDA).

OASIS-3 lives in an XNAT server (https://www.nitrc.org/ir, project OASIS3). Access requires a
NITRC account that has accepted the OASIS Data Use Agreement. Credentials are read from
OASIS_USER / OASIS_PASSWORD (aliases NITRC_USER / NITRC_PASSWORD); if absent you are prompted.

Examples
--------
List subjects and experiments (writes CSVs, no imaging download):
    python scripts/download_data.py oasis-list --out data/oasis3/tables --sample 20

Download FreeSurfer outputs (stats files are enough for the ROI models):
    python scripts/download_data.py oasis-freesurfer --ids data/oasis3/tables/freesurfer_ids.csv \
        --out data/oasis3/freesurfer --sample 5

Download MR sessions (T1 for the CNN arm) or PUP outputs (Centiloid tables):
    python scripts/download_data.py oasis-mr  --ids data/oasis3/tables/mr_ids.csv  --out data/oasis3/mr --sample 2
    python scripts/download_data.py oasis-pup --ids data/oasis3/tables/pup_ids.csv --out data/oasis3/pup --sample 2

Print instructions and check the expected ADNI / A4 files:
    python scripts/download_data.py adni --root data/adni
    python scripts/download_data.py a4   --root data/a4

The endpoints mirror the official scripts at https://github.com/NrgXnat/oasis-scripts
(download_oasis_scans.sh, download_oasis_freesurfer.sh, download_oasis_pup.sh).
"""
from __future__ import annotations

import argparse
import csv
import getpass
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

XNAT_BASE = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"


# ---------------------------------------------------------------------------
# XNAT helpers
# ---------------------------------------------------------------------------
def _credentials() -> tuple[str, str]:
    user = os.environ.get("OASIS_USER") or os.environ.get("NITRC_USER")
    pwd = os.environ.get("OASIS_PASSWORD") or os.environ.get("NITRC_PASSWORD")
    if not user:
        user = input("NITRC username: ").strip()
    if not pwd:
        pwd = getpass.getpass("NITRC password: ")
    return user, pwd


def xnat_session(base: str = XNAT_BASE) -> requests.Session:
    """Open an authenticated XNAT session (POST /data/JSESSION returns a JSESSIONID)."""
    user, pwd = _credentials()
    sess = requests.Session()
    r = sess.post(f"{base}/data/JSESSION", auth=(user, pwd), timeout=60)
    if r.status_code != 200:
        sys.exit(f"XNAT login failed ({r.status_code}). Check credentials / DUA approval for {PROJECT}.")
    sess.cookies.set("JSESSIONID", r.text.strip())
    return sess


def xnat_get_json(sess: requests.Session, url: str) -> list[dict]:
    r = sess.get(url, params={"format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json().get("ResultSet", {}).get("Result", [])


def list_subjects(sess: requests.Session, project: str = PROJECT, base: str = XNAT_BASE) -> list[dict]:
    return xnat_get_json(sess, f"{base}/data/projects/{project}/subjects")


def list_experiments(sess: requests.Session, subject: str, project: str = PROJECT,
                     base: str = XNAT_BASE) -> list[dict]:
    return xnat_get_json(sess, f"{base}/data/projects/{project}/subjects/{subject}/experiments")


def list_assessors(sess: requests.Session, subject: str, experiment: str, project: str = PROJECT,
                   base: str = XNAT_BASE) -> list[dict]:
    return xnat_get_json(
        sess, f"{base}/data/projects/{project}/subjects/{subject}/experiments/{experiment}/assessors")


def download_zip(sess: requests.Session, url: str, out_dir: Path, extract: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / "download.zip"
    with sess.get(url, params={"format": "zip"}, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with open(zpath, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    if extract:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
        zpath.unlink()
    return out_dir


def _subject_of(oasis_id: str) -> str:
    return oasis_id.split("_")[0]


def _read_ids(path: Path, column_candidates: Iterable[str]) -> list[str]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        col = next((c for c in column_candidates if c in cols), None)
        if col is None:
            # single-column CSV without header, or first column
            fh.seek(0)
            return [row[0].strip() for row in csv.reader(fh) if row and row[0].strip()]
        return [row[col].strip() for row in reader if row.get(col, "").strip()]


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_oasis_list(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = xnat_session()
    subjects = list_subjects(sess)
    if args.sample:
        subjects = subjects[: args.sample]
    print(f"{len(subjects)} subjects listed from {PROJECT}")
    rows_mr, rows_pet, rows_fs, rows_pup = [], [], [], []
    for s in subjects:
        label = s.get("label") or s.get("ID")
        for e in list_experiments(sess, label):
            elabel = e.get("label", "")
            xsi = e.get("xsiType", "")
            if "_MR_" in elabel:
                rows_mr.append({"subject": label, "experiment_id": elabel, "xsiType": xsi})
                for a in list_assessors(sess, label, elabel):
                    if "Freesurfer" in a.get("label", ""):
                        rows_fs.append({"subject": label, "experiment_id": elabel,
                                        "freesurfer_id": a["label"]})
            elif any(t in elabel for t in ("_PIB_", "_AV45_", "_FDG_")):
                rows_pet.append({"subject": label, "experiment_id": elabel, "xsiType": xsi})
                for a in list_assessors(sess, label, elabel):
                    if "PUP" in a.get("label", ""):
                        rows_pup.append({"subject": label, "experiment_id": elabel, "pup_id": a["label"]})
    for name, rows in (("mr_ids.csv", rows_mr), ("pet_ids.csv", rows_pet),
                       ("freesurfer_ids.csv", rows_fs), ("pup_ids.csv", rows_pup)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {out / name} ({len(rows)} rows)")
    print("Spreadsheets (clinical, PUP Centiloid tables) are downloaded from the XNAT project page "
          "('OASIS3_data_files'); see data/README.md.")


def _bulk(args: argparse.Namespace, kind: str) -> None:
    ids = _read_ids(Path(args.ids), {"mr": ["experiment_id", "MR ID", "mr_id"],
                                     "freesurfer": ["freesurfer_id", "FS ID", "fs_id"],
                                     "pup": ["pup_id", "PUP ID"]}[kind])
    if args.sample:
        ids = ids[: args.sample]
    sess = xnat_session()
    out = Path(args.out)
    for i, oid in enumerate(ids, 1):
        subj = _subject_of(oid)
        if kind == "mr":
            url = (f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{subj}/experiments/{oid}"
                   f"/scans/{args.scan_type}/files")
        elif kind == "freesurfer":
            # Freesurfer assessor sits on the MR session with the same day index
            exp = oid.replace("_Freesurfer53_", "_MR_").replace("_Freesurfer_", "_MR_")
            url = (f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{subj}/experiments/{exp}"
                   f"/assessors/{oid}/files")
        else:  # pup: OAS30001_AV45_PUPTIMECOURSE_d2430 -> PET session OAS30001_AV45_d2430
            exp = oid.replace("_PUPTIMECOURSE", "")
            url = (f"{XNAT_BASE}/data/projects/{PROJECT}/subjects/{subj}/experiments/{exp}"
                   f"/assessors/{oid}/files")
        dest = out / oid
        if dest.exists() and any(dest.iterdir()):
            print(f"[{i}/{len(ids)}] {oid}: exists, skipping")
            continue
        print(f"[{i}/{len(ids)}] {oid}: downloading")
        try:
            download_zip(sess, url, dest)
        except requests.HTTPError as exc:
            print(f"    failed: {exc}")


def cmd_oasis_mr(args: argparse.Namespace) -> None:
    _bulk(args, "mr")


def cmd_oasis_freesurfer(args: argparse.Namespace) -> None:
    _bulk(args, "freesurfer")


def cmd_oasis_pup(args: argparse.Namespace) -> None:
    _bulk(args, "pup")


ADNI_EXPECTED = ["ADNIMERGE.csv", "UCBERKELEY_AMY_6MM.csv", "UCSFFSX.csv"]
A4_EXPECTED = ["SUBJINFO.csv", "PTDEMOG.csv", "SPINFO.csv"]


def _check(root: Path, expected: list[str], label: str, instructions: str) -> None:
    print(instructions)
    tables = root / "tables"
    missing = [f for f in expected if not list(tables.glob(f.replace(".csv", "*.csv")))]
    if missing:
        print(f"\n{label}: expected tables not found under {tables}: {missing}")
        print("Download them manually from LONI IDA (application required), then re-run.")
    else:
        print(f"\n{label}: all expected tables present under {tables}.")


def cmd_adni(args: argparse.Namespace) -> None:
    _check(Path(args.root), ADNI_EXPECTED, "ADNI",
           "ADNI requires an approved application at https://adni.loni.usc.edu/data-samples/access-data/.\n"
           "After approval: https://ida.loni.usc.edu -> ADNI -> Download -> Study Data ->\n"
           "  ADNIMERGE.csv, UCBERKELEY_AMY_6MM*.csv (Centiloid), UCSFFSX*.csv (FreeSurfer ROIs).\n"
           "Place them in <root>/tables/. MRI NIfTI (optional CNN arm) go to <root>/mri/<PTID>/.")


def cmd_a4(args: argparse.Namespace) -> None:
    _check(Path(args.root), A4_EXPECTED, "A4",
           "A4/LEARN pre-randomization data require access to the 'A4' project on https://ida.loni.usc.edu.\n"
           "Download the screening tables (SUBJINFO, PTDEMOG, SPINFO with amyloid PET SUVr/Centiloid,\n"
           "APOE, PACC) and T1 MRI of screen-eligible participants. Place tables in <root>/tables/.")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("oasis-list", help="list OASIS-3 subjects/experiments/assessors to CSV")
    s.add_argument("--out", default="data/oasis3/tables")
    s.add_argument("--sample", type=int, default=0, help="only the first N subjects")
    s.set_defaults(func=cmd_oasis_list)

    for name, func, ids_help in (("oasis-mr", cmd_oasis_mr, "CSV with experiment_id column"),
                                 ("oasis-freesurfer", cmd_oasis_freesurfer, "CSV with freesurfer_id column"),
                                 ("oasis-pup", cmd_oasis_pup, "CSV with pup_id column")):
        s = sub.add_parser(name)
        s.add_argument("--ids", required=True, help=ids_help)
        s.add_argument("--out", required=True)
        s.add_argument("--sample", type=int, default=0)
        if name == "oasis-mr":
            s.add_argument("--scan-type", default="ALL", help="XNAT scan filter, e.g. ALL or T1w")
        s.set_defaults(func=func)

    for name, func in (("adni", cmd_adni), ("a4", cmd_a4)):
        s = sub.add_parser(name)
        s.add_argument("--root", default=f"data/{name}")
        s.set_defaults(func=func)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
