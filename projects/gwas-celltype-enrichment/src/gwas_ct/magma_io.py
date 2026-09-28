"""Writers for MAGMA and S-LDSC inputs, parsers for their outputs, and command builders.

MAGMA (de Leeuw et al. 2015) gene-set file format: one set per line, `SET_NAME GENE1 GENE2 ...`.
MAGMA gene-covariate file: header `GENE <covar1> <covar2> ...`, one gene per row.
S-LDSC (Finucane et al. 2018) needs a gene-coordinate file (`GENE CHR START END`) and one gene-set
file with one gene id per line for `make_annot.py --gene-set-file`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

MAGMA_GENES_OUT_COLS = ["GENE", "CHR", "START", "STOP", "NSNPS", "NPARAM", "N", "ZSTAT", "P"]


# ----------------------------------------------------------------------------- id mapping
def symbols_to_ids(symbols: Iterable[str], mapping: Mapping[str, str]) -> List[str]:
    """Map gene symbols to MAGMA/LDSC gene ids (Entrez or Ensembl); unmapped symbols are dropped."""
    return [mapping[s] for s in symbols if s in mapping]


def mapping_from_table(tab: pd.DataFrame, symbol_col: str = "symbol", id_col: str = "entrez") -> Dict[str, str]:
    """Build a symbol->id dict from a two-column table, keeping the first id per symbol."""
    t = tab[[symbol_col, id_col]].dropna().drop_duplicates(subset=[symbol_col])
    return dict(zip(t[symbol_col].astype(str), t[id_col].astype(str)))


# ----------------------------------------------------------------------------- MAGMA writers
def write_magma_geneset_file(gene_sets: Mapping[str, Iterable[str]], path: PathLike,
                             id_map: Optional[Mapping[str, str]] = None, min_genes: int = 10) -> Path:
    """Write MAGMA `--set-annot` file. Set names must not contain whitespace (replaced with '_')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for name, genes in gene_sets.items():
            ids = symbols_to_ids(genes, id_map) if id_map else [str(g) for g in genes]
            ids = list(dict.fromkeys(ids))
            if len(ids) < min_genes:
                continue
            fh.write(str(name).replace(" ", "_") + " " + " ".join(ids) + "\n")
            n += 1
    if n == 0:
        raise ValueError("No gene set had >= min_genes mapped genes")
    return path


def write_magma_covar_file(covar: pd.DataFrame, path: PathLike, id_map: Optional[Mapping[str, str]] = None,
                           add_average: bool = True) -> Path:
    """Write MAGMA `--gene-covar` file from a (genes x labels) frame (e.g. specificity quantiles).

    `add_average` appends an `Average` column (mean over labels) so you can run
    `--model condition-hide=Average direction=greater` as in MAGMA_Celltyping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = covar.copy()
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    if add_average:
        df["Average"] = df.mean(axis=1)
    if id_map:
        keep = [g for g in df.index if g in id_map]
        df = df.loc[keep]
        df.index = [id_map[g] for g in keep]
        df = df[~df.index.duplicated()]
    df.index.name = "GENE"
    df.to_csv(path, sep="\t")
    return path


def magma_commands(magma_bin: str, sumstats: PathLike, out_prefix: PathLike, gene_loc: PathLike,
                   bfile: PathLike, n_col_or_value: Union[str, int], snp_col: str = "SNP", p_col: str = "P",
                   window: str = "35,10", geneset_file: Optional[PathLike] = None,
                   covar_file: Optional[PathLike] = None) -> List[str]:
    """Shell commands for the standard MAGMA pipeline (annotate -> gene analysis -> gene-set/property)."""
    out_prefix = str(out_prefix)
    n_arg = f"ncol={n_col_or_value}" if isinstance(n_col_or_value, str) else f"N={n_col_or_value}"
    cmds = [
        f"{magma_bin} --annotate window={window} --snp-loc {sumstats}.snploc --gene-loc {gene_loc} --out {out_prefix}",
        f"{magma_bin} --bfile {bfile} --pval {sumstats} use={snp_col},{p_col} {n_arg} "
        f"--gene-annot {out_prefix}.genes.annot --gene-model snp-wise=mean --out {out_prefix}",
    ]
    if geneset_file is not None:
        cmds.append(f"{magma_bin} --gene-results {out_prefix}.genes.raw --set-annot {geneset_file} --out {out_prefix}.sets")
    if covar_file is not None:
        cmds.append(f"{magma_bin} --gene-results {out_prefix}.genes.raw --gene-covar {covar_file} "
                    f"--model condition-hide=Average direction=greater --out {out_prefix}.covar")
    return cmds


# ----------------------------------------------------------------------------- MAGMA parsers
def read_magma_genes_out(path: PathLike) -> pd.DataFrame:
    """Parse `<prefix>.genes.out` (gene-level MAGMA results) into a DataFrame indexed by GENE."""
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    df["GENE"] = df["GENE"].astype(str)
    return df.set_index("GENE")


def read_magma_gsa_out(path: PathLike) -> pd.DataFrame:
    """Parse `<prefix>.gsa.out` (gene-set or gene-property results), skipping '#' header lines."""
    rows = []
    header: Optional[List[str]] = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if header is None:
                header = parts
                continue
            rows.append(parts)
    if header is None:
        raise ValueError(f"No header found in {path}")
    df = pd.DataFrame(rows, columns=header)
    for c in ("NGENES", "BETA", "BETA_STD", "SE", "P"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def gene_z_from_magma(genes_out: pd.DataFrame, id_to_symbol: Optional[Mapping[str, str]] = None) -> pd.Series:
    """Gene-level Z statistics (ZSTAT) keyed by symbol if a mapping is given, else by id."""
    z = genes_out["ZSTAT"].astype(float)
    if id_to_symbol:
        z = z.rename(index=lambda g: id_to_symbol.get(str(g), str(g)))
        z = z[~z.index.duplicated()]
    return z


# ----------------------------------------------------------------------------- S-LDSC writers
def write_ldsc_gene_coord(coords: pd.DataFrame, path: PathLike, gene_col: str = "GENE", chr_col: str = "CHR",
                          start_col: str = "START", end_col: str = "END") -> Path:
    """Write the gene coordinate file for `make_annot.py --gene-coord-file` (tab-separated)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = coords[[gene_col, chr_col, start_col, end_col]].copy()
    df.columns = ["GENE", "CHR", "START", "END"]
    df["CHR"] = df["CHR"].astype(str).str.replace("chr", "", regex=False)
    df.to_csv(path, sep="\t", index=False)
    return path


def write_ldsc_genesets(gene_sets: Mapping[str, Iterable[str]], out_dir: PathLike,
                        id_map: Optional[Mapping[str, str]] = None, min_genes: int = 10) -> Dict[str, Path]:
    """One file per gene set (one gene id per line) for `make_annot.py --gene-set-file`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, genes in gene_sets.items():
        ids = symbols_to_ids(genes, id_map) if id_map else [str(g) for g in genes]
        ids = list(dict.fromkeys(ids))
        if len(ids) < min_genes:
            continue
        p = out_dir / (str(name).replace(" ", "_").replace("/", "_") + ".GeneSet")
        p.write_text("\n".join(ids) + "\n")
        paths[name] = p
    return paths


def ldsc_commands(ldsc_dir: PathLike, geneset_file: PathLike, gene_coord: PathLike, bfile_prefix: PathLike,
                  annot_prefix: PathLike, sumstats: PathLike, baseline_prefix: PathLike, weights_prefix: PathLike,
                  frq_prefix: PathLike, out_prefix: PathLike, window_kb: int = 100,
                  chromosomes: Sequence[int] = tuple(range(1, 23))) -> List[str]:
    """Shell commands for make_annot -> ldsc l2 -> ldsc h2 (partitioned heritability)."""
    ldsc_dir = str(ldsc_dir)
    cmds: List[str] = []
    for c in chromosomes:
        cmds.append(
            f"python {ldsc_dir}/make_annot.py --gene-set-file {geneset_file} --gene-coord-file {gene_coord} "
            f"--windowsize {window_kb * 1000} --bimfile {bfile_prefix}.{c}.bim --annot-file {annot_prefix}.{c}.annot.gz")
        cmds.append(
            f"python {ldsc_dir}/ldsc.py --l2 --bfile {bfile_prefix}.{c} --ld-wind-cm 1 --annot {annot_prefix}.{c}.annot.gz "
            f"--thin-annot --out {annot_prefix}.{c} --print-snps {ldsc_dir}/hapmap3_snps/hm.{c}.snp")
    cmds.append(
        f"python {ldsc_dir}/ldsc.py --h2 {sumstats} --ref-ld-chr {annot_prefix}.,{baseline_prefix}. "
        f"--w-ld-chr {weights_prefix}. --frqfile-chr {frq_prefix}. --overlap-annot --print-coefficients --out {out_prefix}")
    return cmds


def read_ldsc_results(path: PathLike) -> pd.DataFrame:
    """Parse `<out>.results` from `ldsc.py --h2 --print-coefficients`."""
    df = pd.read_csv(path, sep="\t")
    if "Coefficient_z-score" not in df.columns and {"Coefficient", "Coefficient_std_error"} <= set(df.columns):
        df["Coefficient_z-score"] = df["Coefficient"] / df["Coefficient_std_error"]
    return df
