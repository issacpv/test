#!/usr/bin/env python
"""Download the open inputs for cross-species-imaging-transcriptomics.

Everything is open access: the Allen Brain Cell (ABC) Atlas MERFISH data live in a
public S3 bucket served over HTTPS, the Allen Mouse Brain ISH data come from the
Allen Brain Map RMA API, and orthologs come from Ensembl BioMart.

Examples
--------
Smoke test (ABC metadata + gene tables, 3 ISH genes, orthologs)::

    python scripts/download_data.py --sample

Full MERFISH expression (large) and imputed values::

    python scripts/download_data.py --merfish --imputed

ISH expression energy for a gene list::

    python scripts/download_data.py --ish --genes Pvalb Sst Vip Gad1 Slc17a7

The script is resumable: files that already exist with the expected size are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = logging.getLogger("download_data")

ABC_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
ABC_RELEASE = "20230830"
ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
BIOMART = "https://www.ensembl.org/biomart/martservice"

MERFISH_DIRS = ["MERFISH-C57BL6J-638850", "MERFISH-C57BL6J-638850-CCF", "Allen-CCF-2020"]
ZHUANG_DIRS = [f"Zhuang-ABCA-{i}" for i in range(1, 5)] + [f"Zhuang-ABCA-{i}-CCF" for i in range(1, 5)]


# --------------------------------------------------------------------------- helpers
def stream_download(url: str, dest: Path, expected_size: Optional[int] = None, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` with resume-by-size and simple retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        LOG.debug("skip existing %s", dest)
        return dest
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            LOG.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except requests.RequestException as exc:  # pragma: no cover - network
            LOG.warning("attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


# --------------------------------------------------------------------------- ABC atlas
def fetch_abc_manifest(release: str = ABC_RELEASE) -> Dict:
    """Fetch and cache the ABC Atlas release manifest (JSON)."""
    dest = DATA / "abc" / f"manifest_{release}.json"
    if not dest.exists():
        stream_download(f"{ABC_BASE}/releases/{release}/manifest.json", dest)
    with dest.open() as fh:
        return json.load(fh)


def iter_manifest_files(manifest: Dict) -> Iterator[Dict]:
    """Walk the nested ``file_listing`` of an ABC manifest and yield file records.

    Each yielded dict has keys ``directory``, ``kind`` (``metadata`` or
    ``expression_matrices``), ``relative_path``, ``url`` and ``size``. The manifest
    layout is ``file_listing[directory][kind][...]{"files": {name: {relative_path,
    url, size}}}``; this walker is tolerant to the extra nesting levels used for
    expression matrices (``{matrix}/{"log2"|"raw"}``).
    """
    listing = manifest.get("file_listing", {})
    for directory, kinds in listing.items():
        for kind, node in kinds.items():
            stack = [node]
            while stack:
                cur = stack.pop()
                if not isinstance(cur, dict):
                    continue
                files = cur.get("files")
                if isinstance(files, dict):
                    for name, rec in files.items():
                        if isinstance(rec, dict) and "relative_path" in rec:
                            yield {
                                "directory": directory,
                                "kind": kind,
                                "name": name,
                                "relative_path": rec["relative_path"],
                                "url": rec.get("url") or f"{ABC_BASE}/{rec['relative_path']}",
                                "size": rec.get("size"),
                            }
                for key, val in cur.items():
                    if key != "files" and isinstance(val, dict):
                        stack.append(val)


def download_abc(directories: Iterable[str], kinds: Iterable[str], name_filter: Optional[List[str]] = None,
                 release: str = ABC_RELEASE) -> List[Path]:
    """Download ABC files whose directory and kind match; optional substring filter on file name."""
    manifest = fetch_abc_manifest(release)
    directories = set(directories)
    kinds = set(kinds)
    out: List[Path] = []
    for rec in iter_manifest_files(manifest):
        if rec["directory"] not in directories or rec["kind"] not in kinds:
            continue
        if name_filter and not any(f in rec["relative_path"] for f in name_filter):
            continue
        dest = DATA / "abc" / rec["relative_path"]
        out.append(stream_download(rec["url"], dest, expected_size=rec.get("size")))
    LOG.info("ABC: %d files in place", len(out))
    return out


# --------------------------------------------------------------------------- Allen ISH
def rma_query(criteria: str, num_rows: int = 2000) -> List[Dict]:
    """Run a paginated RMA query against the Allen Brain Map API."""
    rows: List[Dict] = []
    start = 0
    while True:
        url = f"{ALLEN_API}?criteria={criteria},rma::options[start_row$eq{start}][num_rows$eq{num_rows}]"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success", False):
            raise RuntimeError(payload.get("msg", "RMA query failed"))
        msg = payload["msg"]
        rows.extend(msg)
        if len(msg) < num_rows:
            return rows
        start += num_rows


def ish_section_datasets(gene: str) -> List[Dict]:
    crit = (
        "model::SectionDataSet,rma::criteria,[failed$eq'false'],products[abbreviation$eq'Mouse'],"
        f"genes[acronym$eq'{gene}'],rma::include,genes,plane_of_section"
    )
    return rma_query(crit)


def ish_unionize(section_data_set_id: int) -> List[Dict]:
    crit = f"model::StructureUnionize,rma::criteria,[section_data_set_id$eq{section_data_set_id}]"
    return rma_query(crit)


def download_ish(genes: List[str]) -> Path:
    """Fetch expression energy per structure for each gene; write a gene x structure CSV."""
    import pandas as pd

    out_dir = DATA / "ish"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for gene in genes:
        sd_path = out_dir / f"section_datasets_{gene}.json"
        if sd_path.exists():
            datasets = json.loads(sd_path.read_text())
        else:
            datasets = ish_section_datasets(gene)
            sd_path.write_text(json.dumps(datasets))
        for ds in datasets:
            plane = (ds.get("plane_of_section") or {}).get("name", "unknown")
            u_path = out_dir / f"unionize_{ds['id']}.json"
            if u_path.exists():
                unions = json.loads(u_path.read_text())
            else:
                unions = ish_unionize(ds["id"])
                u_path.write_text(json.dumps(unions))
            for u in unions:
                records.append({"gene": gene, "section_data_set_id": ds["id"], "plane": plane,
                                "structure_id": u["structure_id"], "expression_energy": u["expression_energy"],
                                "expression_density": u.get("expression_density"), "sum_pixels": u.get("sum_pixels")})
        LOG.info("ISH %s: %d section data sets", gene, len(datasets))
    df = pd.DataFrame(records)
    dest = out_dir / "ish_expression_energy.csv"
    df.to_csv(dest, index=False)
    return dest


# --------------------------------------------------------------------------- orthologs
def download_orthologs() -> Path:
    """One-to-one mouse-human orthologs from Ensembl BioMart (TSV)."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        '<Filter name="with_hsapiens_homolog" excluded="0"/>'
        '<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>'
        '<Attribute name="hsapiens_homolog_ensembl_gene"/><Attribute name="hsapiens_homolog_associated_gene_name"/>'
        '<Attribute name="hsapiens_homolog_orthology_type"/><Attribute name="hsapiens_homolog_orthology_confidence"/>'
        "</Dataset></Query>"
    )
    dest = DATA / "orthologs" / "mouse_human_all.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.post(BIOMART, data={"query": xml}, timeout=600)
        r.raise_for_status()
        dest.write_text(r.text)
    import pandas as pd

    df = pd.read_csv(dest, sep="\t")
    one2one = df[df["Human homology type"] == "ortholog_one2one"] if "Human homology type" in df else df
    out = DATA / "orthologs" / "mouse_human_one2one.csv"
    one2one.to_csv(out, index=False)
    LOG.info("orthologs: %d rows, %d one-to-one", len(df), len(one2one))
    return out


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", action="store_true", help="metadata/gene tables + 3 ISH genes + orthologs")
    p.add_argument("--metadata", action="store_true", help="ABC MERFISH + CCF metadata (no expression)")
    p.add_argument("--merfish", action="store_true", help="ABC MERFISH cell-by-gene matrices (large)")
    p.add_argument("--imputed", action="store_true", help="ABC imputed whole-transcriptome matrices (very large)")
    p.add_argument("--zhuang", action="store_true", help="Zhuang-ABCA-1..4 metadata + expression")
    p.add_argument("--ish", action="store_true", help="Allen ISH expression energy per structure")
    p.add_argument("--genes", nargs="*", default=None, help="gene symbols for --ish")
    p.add_argument("--orthologs", action="store_true")
    p.add_argument("--release", default=ABC_RELEASE)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.sample:
        download_abc(MERFISH_DIRS, ["metadata"], release=args.release)
        download_ish(args.genes or ["Pvalb", "Sst", "Vip"])
        download_orthologs()
        return 0
    if args.metadata:
        download_abc(MERFISH_DIRS, ["metadata"], release=args.release)
    if args.merfish:
        download_abc(["MERFISH-C57BL6J-638850"], ["expression_matrices"], release=args.release)
    if args.imputed:
        download_abc(["MERFISH-C57BL6J-638850-imputed"], ["expression_matrices"], release=args.release)
    if args.zhuang:
        download_abc(ZHUANG_DIRS, ["metadata", "expression_matrices"], release=args.release)
    if args.ish:
        if not args.genes:
            p.error("--ish requires --genes")
        download_ish(args.genes)
    if args.orthologs:
        download_orthologs()
    if not any([args.metadata, args.merfish, args.imputed, args.zhuang, args.ish, args.orthologs]):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
