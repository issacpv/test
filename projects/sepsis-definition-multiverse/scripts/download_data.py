#!/usr/bin/env python
"""Download inputs for sepsis-definition-multiverse.

Open parts (no credentials): the MIMIC-IV and eICU-CRD *demo* databases and the mimic-code
reference SQL. Credentialed parts (MIMIC-IV v3.1, eICU-CRD v2.0) are fetched with HTTP basic
auth using ``PHYSIONET_USER`` / ``PHYSIONET_PASS`` from the environment; they are never stored.

Examples
--------
    python scripts/download_data.py --sample                      # demos + reference SQL
    python scripts/download_data.py --mimic-iv --tables hosp/labevents icu/icustays
    python scripts/download_data.py --build-duckdb --db data/mimiciv_demo.duckdb --source data/mimic-iv-demo/2.2
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = logging.getLogger("download_data")

PHYSIONET = "https://physionet.org/files"
DEMOS = {"mimic-iv-demo/2.2": None, "eicu-crd-demo/2.0.1": None}
FULL = {"mimic-iv": "mimiciv/3.1", "eicu": "eicu-crd/2.0"}
MIMIC_TABLES = [
    "hosp/admissions", "hosp/patients", "hosp/labevents", "hosp/d_labitems", "hosp/prescriptions", "hosp/emar",
    "hosp/emar_detail", "hosp/microbiologyevents",
    "icu/icustays", "icu/chartevents", "icu/d_items", "icu/inputevents", "icu/outputevents", "icu/procedureevents",
]
EICU_TABLES = ["patient", "lab", "vitalPeriodic", "vitalAperiodic", "infusionDrug", "medication", "microLab",
               "treatment", "respiratoryCare", "nurseCharting", "intakeOutput", "apacheApsVar", "hospital"]
REFERENCE_SQL = {
    "sepsis3.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/sepsis/sepsis3.sql",
    "suspicion_of_infection.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/sepsis/suspicion_of_infection.sql",
    "sofa.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/score/sofa.sql",
}


def credentials() -> Optional[tuple]:
    u, p = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    return (u, p) if u and p else None


def fetch(url: str, dest: Path, auth: Optional[tuple] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, auth=auth, timeout=600) as r:
        if r.status_code in (401, 403):
            raise PermissionError(f"{url}: credentialed access required (set PHYSIONET_USER/PHYSIONET_PASS and sign the DUA)")
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    LOG.info("%s (%.1f MB)", dest.relative_to(ROOT), dest.stat().st_size / 1e6)
    return dest


def list_directory(url: str, auth: Optional[tuple] = None) -> List[str]:
    """Parse the PhysioNet directory index for file links (csv.gz, csv, sql, txt)."""
    r = requests.get(url, auth=auth, timeout=120)
    r.raise_for_status()
    links = re.findall(r'href="([^"?]+)"', r.text)
    return [l for l in links if re.search(r"\.(csv\.gz|csv|sql|txt|md)$", l)]


def download_tree(project: str, subdirs: Iterable[str], auth: Optional[tuple], name_filter: Optional[List[str]] = None) -> None:
    for sub in subdirs:
        url = f"{PHYSIONET}/{project}/{sub}/" if sub else f"{PHYSIONET}/{project}/"
        for fname in list_directory(url, auth):
            base = fname.split("/")[-1]
            if name_filter and not any(base.startswith(t) for t in name_filter):
                continue
            fetch(url + base, DATA / project / sub / base, auth)


def download_demos() -> None:
    for proj in DEMOS:
        subdirs = ["hosp", "icu"] if proj.startswith("mimic") else [""]
        download_tree(proj, subdirs, auth=None)


def download_reference_sql() -> None:
    for name, url in REFERENCE_SQL.items():
        fetch(url, DATA / "reference" / name)


def download_full(which: str, tables: Optional[List[str]]) -> None:
    auth = credentials()
    if auth is None:
        raise SystemExit("PHYSIONET_USER / PHYSIONET_PASS not set; credentialed download refused")
    project = FULL[which]
    if which == "mimic-iv":
        wanted = tables or MIMIC_TABLES
        for sub in ("hosp", "icu"):
            names = [t.split("/")[1] for t in wanted if t.startswith(sub + "/")]
            if names:
                download_tree(project, [sub], auth, name_filter=names)
    else:
        download_tree(project, [""], auth, name_filter=tables or EICU_TABLES)


def build_duckdb(db: Path, source: Path) -> None:
    import duckdb

    con = duckdb.connect(str(db))
    for f in sorted(source.rglob("*.csv*")):
        table = f.relative_to(source).with_suffix("").with_suffix("").as_posix().replace("/", "_").lower()
        table = re.sub(r"[^a-z0-9_]", "_", table)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{f.as_posix()}', header=true)")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        LOG.info("%s: %d rows", table, n)
    con.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="demo databases + reference SQL (open)")
    p.add_argument("--mimic-iv", action="store_true", help="MIMIC-IV v3.1 (credentialed)")
    p.add_argument("--eicu", action="store_true", help="eICU-CRD v2.0 (credentialed)")
    p.add_argument("--tables", nargs="*", default=None, help="subset of tables (e.g. hosp/labevents icu/icustays)")
    p.add_argument("--build-duckdb", action="store_true")
    p.add_argument("--db", type=Path, default=DATA / "mimiciv_demo.duckdb")
    p.add_argument("--source", type=Path, default=DATA / "mimic-iv-demo" / "2.2")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.sample:
        download_demos()
        download_reference_sql()
    if args.mimic_iv:
        download_full("mimic-iv", args.tables)
    if args.eicu:
        download_full("eicu", args.tables)
    if args.build_duckdb:
        build_duckdb(args.db, args.source)
    if not any([args.sample, args.mimic_iv, args.eicu, args.build_duckdb]):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
