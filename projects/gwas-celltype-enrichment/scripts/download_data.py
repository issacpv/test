#!/usr/bin/env python3
"""Download the open inputs for gwas-celltype-enrichment.

Sub-commands
------------
abc            Allen Brain Cell Atlas via its public S3 manifest (no credentials).
siletti        Siletti et al. 2023 human brain atlas from CZ CELLxGENE (curation REST API).
big40          UK Biobank BIG40 imaging GWAS summary statistics (open web server).
gwas-catalog   GWAS Catalog summary statistics by GCST accession (REST + FTP layout).
pgc            Prints the PGC files you must click-through-download manually.

Examples
--------
python scripts/download_data.py abc --sample
python scripts/download_data.py abc --merfish --wmb-10x TH HY
python scripts/download_data.py big40 --idps 0001 0002 --sample
python scripts/download_data.py gwas-catalog --accessions GCST90027158
python scripts/download_data.py siletti --list

Only standard-library + `gwas_ct.abc_loader` are required. `--sample` keeps
everything small (metadata only / first N rows) so it can run in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gwas_ct.abc_loader import AbcManifest, DEFAULT_RELEASE  # noqa: E402

DATA = ROOT / "data"

# Siletti et al. 2023 CELLxGENE collection ("Transcriptomic diversity of cell types across the adult
# human brain"). If CZI rotates ids, pass --collection explicitly.
SILETTI_COLLECTION = "283d65eb-dd53-496d-adb7-7570c7caa443"
CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"

BIG40_BASE = "https://open.oxcin.ox.ac.uk/ukbiobank/big40/release2/stats33k"
GWAS_CATALOG_FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
GWAS_CATALOG_REST = "https://www.ebi.ac.uk/gwas/rest/api/studies"

PGC_EXPECTED = {
    "SCZ3 (Trubetskoy 2022)": "PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv.gz",
    "BD (Mullins 2021)": "pgc-bip2021-all.vcf.tsv.gz",
    "MDD (Howard 2019 / Als 2023)": "see PGC MDD page",
    "ADHD (Demontis 2023)": "ADHD2022_iPSYCH_deCODE_PGC.meta.gz",
    "ASD (Grove 2019)": "iPSYCH-PGC_ASD_Nov2017.gz",
}


def _stream_download(url: str, dest: Path, max_bytes: Optional[int] = None, chunk: int = 1 << 20) -> Path:
    """Stream `url` to `dest` (resumable-ish: skips if size matches). Optionally truncate at max_bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "gwas_ct-downloader/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total else None
        if dest.exists() and total_i is not None and dest.stat().st_size == total_i and max_bytes is None:
            print(f"  [skip] {dest} (complete)")
            return dest
        written = 0
        with open(dest, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                written += len(buf)
                if max_bytes is not None and written >= max_bytes:
                    break
    print(f"  [ok] {dest} ({written/1e6:.1f} MB)")
    return dest


# --------------------------------------------------------------------------------------- ABC Atlas
def cmd_abc(args: argparse.Namespace) -> None:
    base = DATA / "abc_atlas"
    man = AbcManifest.fetch(release=args.release, cache_dir=base)
    print(f"ABC manifest {man.version}: {len(man.datasets())} datasets")

    wanted = [
        ("WMB-10X", "metadata", "cell_metadata_with_cluster_annotation"),
        ("WMB-10X", "metadata", "gene"),
        ("WMB-taxonomy", "metadata", "cluster_to_cluster_annotation_membership"),
        ("WMB-taxonomy", "metadata", "cluster_annotation_term"),
        ("MERFISH-C57BL6J-638850", "metadata", "cell_metadata_with_cluster_annotation"),
        ("MERFISH-C57BL6J-638850", "metadata", "gene"),
        ("MERFISH-C57BL6J-638850-CCF", "metadata", "cell_metadata_with_parcellation_annotation"),
        ("Allen-CCF-2020", "metadata", "parcellation_to_parcellation_term_membership"),
        ("WHB-10Xv3", "metadata", "cell_metadata"),
        ("WHB-10Xv3", "metadata", "gene"),
        ("WHB-taxonomy", "metadata", "cluster_annotation_term"),
        ("WHB-taxonomy", "metadata", "cluster_to_cluster_annotation_membership"),
    ]
    max_bytes = 5_000_000 if args.sample else None
    for ds, kind, name in wanted:
        try:
            f = man.get(ds, kind, name)
        except KeyError as e:
            print(f"  [warn] {e}")
            continue
        man.download(f, base, max_bytes=max_bytes)

    if args.sample:
        print("--sample: skipping expression matrices.")
        return
    if args.merfish:
        man.download(man.get("MERFISH-C57BL6J-638850", "expression_matrices", "C57BL6J-638850", "log2"), base)
    for region in args.wmb_10x or []:
        name = f"WMB-10Xv3-{region}"
        man.download(man.get("WMB-10Xv3", "expression_matrices", name, "log2"), base)
    if args.human:
        for name in ("WHB-10Xv3-Neurons", "WHB-10Xv3-Nonneurons"):
            man.download(man.get("WHB-10Xv3", "expression_matrices", name, "log2"), base)


# ------------------------------------------------------------------------------------- CELLxGENE
def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "gwas_ct/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def cmd_siletti(args: argparse.Namespace) -> None:
    coll = _get_json(f"{CELLXGENE_API}/collections/{args.collection}")
    print(coll.get("name"))
    datasets = coll.get("datasets", [])
    for d in datasets:
        print(f"  {d.get('dataset_id')}  cells={d.get('cell_count')}  {d.get('title')}")
    if args.list:
        return
    out = DATA / "cellxgene" / "siletti"
    for d in datasets:
        if args.dataset and d.get("dataset_id") != args.dataset:
            continue
        for asset in d.get("assets", []):
            if asset.get("filetype", "").upper() == "H5AD":
                _stream_download(asset["url"], out / f"{d['dataset_id']}.h5ad",
                                 max_bytes=2_000_000 if args.sample else None)


# ----------------------------------------------------------------------------------------- BIG40
def cmd_big40(args: argparse.Namespace) -> None:
    out = DATA / "gwas" / "big40"
    out.mkdir(parents=True, exist_ok=True)
    idps: List[str] = [str(i).zfill(4) for i in args.idps]
    for idp in idps:
        url = f"{args.base_url}/{idp}.txt.gz"
        try:
            _stream_download(url, out / f"{idp}.txt.gz", max_bytes=2_000_000 if args.sample else None)
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {url}: {e}\n         Check the current path on https://open.oxcin.ox.ac.uk/ukbiobank/big40/")


# ---------------------------------------------------------------------------------- GWAS Catalog
def gcst_ftp_dir(accession: str) -> str:
    """GWAS Catalog FTP directories are bucketed in blocks of 1000 accessions."""
    n = int(accession.replace("GCST", ""))
    lo = ((n - 1) // 1000) * 1000 + 1
    hi = lo + 999
    width = len(accession) - 4
    return f"{GWAS_CATALOG_FTP}/GCST{lo:0{width}d}-GCST{hi:0{width}d}/{accession}"


def cmd_gwas_catalog(args: argparse.Namespace) -> None:
    out = DATA / "gwas" / "gwas_catalog"
    for acc in args.accessions:
        try:
            meta = _get_json(f"{GWAS_CATALOG_REST}/{acc}")
            print(f"{acc}: {meta.get('diseaseTrait', {}).get('trait')} | {meta.get('publicationInfo', {}).get('title')}")
        except Exception as e:  # noqa: BLE001
            print(f"{acc}: metadata lookup failed ({e})")
        d = gcst_ftp_dir(acc)
        print(f"  FTP dir: {d}/  (look for harmonised/*.h.tsv.gz)")
        (out / acc).mkdir(parents=True, exist_ok=True)
        (out / acc / "SOURCE.txt").write_text(d + "\n")
        if args.file:
            _stream_download(f"{d}/{args.file}", out / acc / args.file,
                             max_bytes=2_000_000 if args.sample else None)


def cmd_pgc(_: argparse.Namespace) -> None:
    print("PGC summary statistics require a click-through agreement; download manually from")
    print("https://pgc.unc.edu/for-researchers/download-results/ into data/gwas/pgc/ :")
    for k, v in PGC_EXPECTED.items():
        print(f"  - {k}: {v}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("abc")
    a.add_argument("--release", default=DEFAULT_RELEASE)
    a.add_argument("--sample", action="store_true", help="metadata only, truncated to 5 MB per file")
    a.add_argument("--merfish", action="store_true")
    a.add_argument("--wmb-10x", nargs="*", metavar="REGION", help="e.g. TH HY Isocortex-1")
    a.add_argument("--human", action="store_true")
    a.set_defaults(func=cmd_abc)

    s = sub.add_parser("siletti")
    s.add_argument("--collection", default=SILETTI_COLLECTION)
    s.add_argument("--list", action="store_true")
    s.add_argument("--dataset", default=None)
    s.add_argument("--sample", action="store_true")
    s.set_defaults(func=cmd_siletti)

    b = sub.add_parser("big40")
    b.add_argument("--idps", nargs="+", required=True)
    b.add_argument("--base-url", default=BIG40_BASE)
    b.add_argument("--sample", action="store_true")
    b.set_defaults(func=cmd_big40)

    g = sub.add_parser("gwas-catalog")
    g.add_argument("--accessions", nargs="+", required=True)
    g.add_argument("--file", default=None, help="file name inside the FTP dir to fetch")
    g.add_argument("--sample", action="store_true")
    g.set_defaults(func=cmd_gwas_catalog)

    sub.add_parser("pgc").set_defaults(func=cmd_pgc)

    args = p.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
