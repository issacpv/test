#!/usr/bin/env python
"""Download inputs for circadian-label-bias-icu.

Open parts: MIMIC-IV demo, eICU-CRD demo and mimic-code reference SQL. Credentialed parts
(MIMIC-IV v3.1, eICU-CRD v2.0, MIMIC-IV-ED v2.2) use HTTP basic auth with
``PHYSIONET_USER`` / ``PHYSIONET_PASS`` from the environment; nothing is stored.

Examples
--------
    python scripts/download_data.py --sample
    python scripts/download_data.py --mimic-iv
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

DEMOS = {"mimic-iv-demo/2.2": ["hosp", "icu"], "eicu-crd-demo/2.0.1": [""]}
FULL = {"mimic-iv": ("mimiciv/3.1", ["hosp", "icu"]), "eicu": ("eicu-crd/2.0", [""]), "mimic-iv-ed": ("mimic-iv-ed/2.2", ["ed"])}
MIMIC_TABLES = ["admissions", "patients", "transfers", "labevents", "d_labitems", "prescriptions", "emar",
                "microbiologyevents", "icustays", "chartevents", "d_items", "inputevents", "outputevents", "procedureevents"]
EICU_TABLES = ["patient", "hospital", "lab", "nurseCharting", "vitalPeriodic", "vitalAperiodic", "infusionDrug",
               "medication", "microLab", "treatment", "respiratoryCare", "intakeOutput", "apacheApsVar"]
ED_TABLES = ["edstays", "triage"]
REFERENCE_SQL = {
    "sepsis3.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/sepsis/sepsis3.sql",
    "kdigo_stages.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/organfailure/kdigo_stages.sql",
    "ventilation.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/treatment/ventilation.sql",
    "vasoactive_agent.sql": "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/medication/vasoactive_agent.sql",
    # code_status.sql has moved between mimic-code releases: located at runtime via the GitHub tree API
    "code_status.sql": None,
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
            raise PermissionError(f"{url}: credentialed access required (set PHYSIONET_USER/PHYSIONET_PASS, sign the DUA)")
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    LOG.info("%s (%.1f MB)", dest.relative_to(ROOT), dest.stat().st_size / 1e6)
    return dest


def list_directory(url: str, auth: Optional[tuple] = None) -> List[str]:
    r = requests.get(url, auth=auth, timeout=120)
    r.raise_for_status()
    return [l for l in re.findall(r'href="([^"?]+)"', r.text) if re.search(r"\.(csv\.gz|csv|txt|md)$", l)]


def download_tree(project: str, subdirs: Iterable[str], auth: Optional[tuple], name_filter: Optional[List[str]] = None) -> None:
    for sub in subdirs:
        url = f"{PHYSIONET}/{project}/{sub}/" if sub else f"{PHYSIONET}/{project}/"
        for fname in list_directory(url, auth):
            base = fname.split("/")[-1]
            if name_filter and not any(base.lower().startswith(t.lower()) for t in name_filter):
                continue
            fetch(url + base, DATA / project / sub / base, auth)


def fetch_reference_sql() -> None:
    for name, url in REFERENCE_SQL.items():
        if url is None:
            # locate the file via the GitHub contents API (path has changed across releases)
            r = requests.get("https://api.github.com/repos/MIT-LCP/mimic-code/git/trees/main?recursive=1", timeout=120)
            if r.ok:
                paths = [t["path"] for t in r.json().get("tree", []) if t["path"].endswith("code_status.sql") and "mimic-iv" in t["path"]]
                if paths:
                    url = f"https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/{paths[0]}"
                else:
                    LOG.warning("code_status.sql not found in mimic-code tree; skipping")
                    continue
        try:
            fetch(url, DATA / "reference" / name)
        except requests.HTTPError as exc:  # pragma: no cover - network
            LOG.warning("reference %s not fetched: %s", name, exc)


def download_full(which: str, tables: Optional[List[str]]) -> None:
    auth = credentials()
    if auth is None:
        raise SystemExit("PHYSIONET_USER / PHYSIONET_PASS not set; credentialed download refused")
    project, subdirs = FULL[which]
    default = {"mimic-iv": MIMIC_TABLES, "eicu": EICU_TABLES, "mimic-iv-ed": ED_TABLES}[which]
    download_tree(project, subdirs, auth, name_filter=tables or default)


def build_duckdb(db: Path, source: Path) -> None:
    import duckdb

    con = duckdb.connect(str(db))
    for f in sorted(source.rglob("*.csv*")):
        table = re.sub(r"[^a-z0-9_]", "_", f.relative_to(source).with_suffix("").with_suffix("").as_posix().replace("/", "_").lower())
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{f.as_posix()}', header=true)")
        LOG.info("%s: %d rows", table, con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    con.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="demo databases + reference SQL (open)")
    p.add_argument("--mimic-iv", action="store_true")
    p.add_argument("--eicu", action="store_true")
    p.add_argument("--mimic-iv-ed", action="store_true")
    p.add_argument("--tables", nargs="*", default=None)
    p.add_argument("--build-duckdb", action="store_true")
    p.add_argument("--db", type=Path, default=DATA / "mimiciv_demo.duckdb")
    p.add_argument("--source", type=Path, default=DATA / "mimic-iv-demo" / "2.2")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.sample:
        for proj, subs in DEMOS.items():
            download_tree(proj, subs, auth=None)
        fetch_reference_sql()
    if args.mimic_iv:
        download_full("mimic-iv", args.tables)
    if args.eicu:
        download_full("eicu", args.tables)
    if args.mimic_iv_ed:
        download_full("mimic-iv-ed", args.tables)
    if args.build_duckdb:
        build_duckdb(args.db, args.source)
    if not any([args.sample, args.mimic_iv, args.eicu, args.mimic_iv_ed, args.build_duckdb]):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
