#!/usr/bin/env python
"""Download helpers for tau-proxy-from-mri.

Open parts: none (all cohorts are under DUA). This script therefore
(1) talks to XNAT Central for OASIS-3 / OASIS-3_AV1451 with credentials
    from the environment (XNAT_USER / XNAT_PASS),
(2) checks that manually downloaded ADNI tables are in place, and
(3) writes a synthetic sample table set with ``--sample`` so the pipeline
    can be run without any DUA data.

Examples
--------
python scripts/download_data.py --sample
python scripts/download_data.py --list-resources --project OASIS3
python scripts/download_data.py --tables --project OASIS3_AV1451
python scripts/download_data.py --freesurfer --project OASIS3 --subjects-from data/oasis3/tau_subjects.txt
python scripts/download_data.py --adni-check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
XNAT_BASE = os.environ.get("XNAT_BASE", "https://central.xnat.org")


def _session():
    """Return an authenticated ``requests.Session`` for XNAT Central."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install requests") from exc
    user, pw = os.environ.get("XNAT_USER"), os.environ.get("XNAT_PASS")
    if not user or not pw:
        raise SystemExit("Set XNAT_USER and XNAT_PASS (NITRC credentials with OASIS-3 DUA approval).")
    s = requests.Session()
    s.auth = (user, pw)
    r = s.get(f"{XNAT_BASE}/data/JSESSION", timeout=60)
    r.raise_for_status()
    return s


def list_resources(project: str) -> None:
    s = _session()
    r = s.get(f"{XNAT_BASE}/data/projects/{project}/resources", params={"format": "json"}, timeout=120)
    r.raise_for_status()
    for row in r.json()["ResultSet"]["Result"]:
        print(row.get("label"), row.get("xnat_abstractresource_id"))
        rr = s.get(
            f"{XNAT_BASE}/data/projects/{project}/resources/{row['label']}/files",
            params={"format": "json"},
            timeout=120,
        )
        if rr.ok:
            for f in rr.json()["ResultSet"]["Result"]:
                print("   ", f.get("Name"), f.get("Size"))


def download_tables(project: str, out: Path) -> None:
    """Download every file in project-level resources (tabular data files)."""
    s = _session()
    out.mkdir(parents=True, exist_ok=True)
    r = s.get(f"{XNAT_BASE}/data/projects/{project}/resources", params={"format": "json"}, timeout=120)
    r.raise_for_status()
    for row in r.json()["ResultSet"]["Result"]:
        label = row["label"]
        rr = s.get(f"{XNAT_BASE}/data/projects/{project}/resources/{label}/files", params={"format": "json"}, timeout=120)
        rr.raise_for_status()
        for f in rr.json()["ResultSet"]["Result"]:
            name = f["Name"]
            if not name.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".pdf")):
                continue
            dest = out / name
            if dest.exists():
                continue
            with s.get(f"{XNAT_BASE}{f['URI']}", stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            print("downloaded", dest)


def _experiments(s, project: str, subject: str, xsi: str):
    r = s.get(
        f"{XNAT_BASE}/data/projects/{project}/subjects/{subject}/experiments",
        params={"format": "json", "xsiType": xsi},
        timeout=120,
    )
    r.raise_for_status()
    return [row["label"] for row in r.json()["ResultSet"]["Result"]]


def download_freesurfer(project: str, subjects: list[str], out: Path) -> None:
    """Fetch FreeSurfer stats files for each MR session (assessor resources)."""
    s = _session()
    for subj in subjects:
        for exp in _experiments(s, project, subj, "xnat:mrSessionData"):
            r = s.get(f"{XNAT_BASE}/data/experiments/{exp}/assessors", params={"format": "json"}, timeout=120)
            if not r.ok:
                continue
            for a in r.json()["ResultSet"]["Result"]:
                if "freesurfer" not in a.get("xsiType", "").lower() and "FS" not in a.get("label", ""):
                    continue
                dest = out / exp / "stats"
                dest.mkdir(parents=True, exist_ok=True)
                rr = s.get(f"{XNAT_BASE}/data/experiments/{exp}/assessors/{a['ID']}/files", params={"format": "json"}, timeout=120)
                if not rr.ok:
                    continue
                for f in rr.json()["ResultSet"]["Result"]:
                    if f["Name"] in {"aseg.stats", "lh.aparc.stats", "rh.aparc.stats", "wmparc.stats"}:
                        target = dest / f["Name"]
                        if target.exists():
                            continue
                        resp = s.get(f"{XNAT_BASE}{f['URI']}", timeout=600)
                        if resp.ok:
                            target.write_bytes(resp.content)
                            print("downloaded", target)


def download_flair(project: str, subjects: list[str], out: Path) -> None:
    """Fetch FLAIR NIfTI scans for each MR session."""
    s = _session()
    for subj in subjects:
        for exp in _experiments(s, project, subj, "xnat:mrSessionData"):
            r = s.get(f"{XNAT_BASE}/data/experiments/{exp}/scans", params={"format": "json"}, timeout=120)
            if not r.ok:
                continue
            for sc in r.json()["ResultSet"]["Result"]:
                if "flair" not in (sc.get("type", "") + sc.get("series_description", "")).lower():
                    continue
                rr = s.get(f"{XNAT_BASE}/data/experiments/{exp}/scans/{sc['ID']}/resources/NIFTI/files", params={"format": "json"}, timeout=120)
                if not rr.ok:
                    continue
                for f in rr.json()["ResultSet"]["Result"]:
                    dest = out / exp / f["Name"]
                    if dest.exists():
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with s.get(f"{XNAT_BASE}{f['URI']}", stream=True, timeout=1200) as resp:
                        resp.raise_for_status()
                        with open(dest, "wb") as fh:
                            for chunk in resp.iter_content(1 << 20):
                                fh.write(chunk)
                    print("downloaded", dest)


def adni_check() -> int:
    expected = ["ADNIMERGE.csv"]
    tdir = DATA / "adni" / "tables"
    ok = True
    for name in expected:
        p = tdir / name
        print(("found   " if p.exists() else "MISSING ") + str(p))
        ok &= p.exists()
    tau = sorted(tdir.glob("UCBERKELEY*TAU*.csv")) + sorted(tdir.glob("UCBERKELEYAV1451*.csv"))
    fs = sorted(tdir.glob("UCSFFSX*.csv"))
    print(f"tau tables: {[p.name for p in tau]}")
    print(f"FreeSurfer tables: {[p.name for p in fs]}")
    return 0 if ok and tau and fs else 1


def write_sample() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from tau_proxy.tables import write_synthetic_oasis3_tables

    out = DATA / "sample"
    write_synthetic_oasis3_tables(out, n_subjects=120, seed=0)
    print("sample tables written to", out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="OASIS3")
    ap.add_argument("--list-resources", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--freesurfer", action="store_true")
    ap.add_argument("--flair", action="store_true")
    ap.add_argument("--subjects-from", type=Path, help="text file with one OASIS subject ID per line")
    ap.add_argument("--adni-check", action="store_true")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args(argv)

    if a.sample:
        write_sample()
        return 0
    if a.adni_check:
        return adni_check()
    if a.list_resources:
        list_resources(a.project)
        return 0
    if a.tables:
        download_tables(a.project, DATA / a.project.lower() / "tables")
        return 0
    if a.freesurfer or a.flair:
        if not a.subjects_from:
            raise SystemExit("--subjects-from is required for imaging downloads")
        subjects = [ln.strip() for ln in a.subjects_from.read_text().splitlines() if ln.strip()]
        if a.freesurfer:
            download_freesurfer(a.project, subjects, DATA / a.project.lower() / "freesurfer")
        if a.flair:
            download_flair(a.project, subjects, DATA / a.project.lower() / "flair")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
