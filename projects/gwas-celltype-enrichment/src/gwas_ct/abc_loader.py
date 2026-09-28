"""Allen Brain Cell Atlas (ABC Atlas) loader.

The ABC Atlas is served from a public S3 bucket described by a JSON manifest
(`releases/<version>/manifest.json`).  This module mirrors the essential behaviour of
`abc_atlas_access.AbcProjectCache` without the dependency: parse the manifest, resolve
files, download with size caps, read metadata, and build pseudobulk matrices from
(backed) anndata objects.

Manifest structure (verified against release 20241115)::

    file_listing[<dataset>][<kind>][<name>] -> {"files": {<ext>: {url, size, file_hash, ...}}}
    file_listing[<dataset>]["expression_matrices"][<name>][<variant>] -> {"files": {"h5ad": {...}}}

where <kind> in {metadata, expression_matrices, image_volumes, mapmycells} and
<variant> in {raw, log2}.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

ABC_S3_HTTP = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
DEFAULT_RELEASE = "20241115"

MOUSE_HIERARCHY = ("class", "subclass", "supertype", "cluster")
HUMAN_HIERARCHY = ("supercluster_term_label", "cluster_term_label")  # WHB naming in cell_metadata
MERFISH_DOMAIN_LEVELS = ("parcellation_division", "parcellation_structure", "parcellation_substructure")

ArrayLike = Union[np.ndarray, "scipy.sparse.spmatrix"]  # noqa: F821


@dataclass(frozen=True)
class ManifestFile:
    """One downloadable file in the ABC manifest."""

    dataset: str
    kind: str
    name: str
    variant: Optional[str]
    ext: str
    url: str
    relative_path: str
    size: int
    file_hash: str

    @property
    def local_relpath(self) -> Path:
        return Path(self.relative_path)


class AbcManifest:
    """Parsed ABC Atlas manifest with download helpers."""

    def __init__(self, manifest: dict):
        self._m = manifest
        self.version: str = manifest.get("version", "unknown")
        self._files: List[ManifestFile] = list(self._iter_files())

    # ---------------------------------------------------------------- construction
    @classmethod
    def fetch(cls, release: str = DEFAULT_RELEASE, cache_dir: Optional[Path] = None,
              timeout: int = 60) -> "AbcManifest":
        """Download (or reuse cached) manifest for `release`."""
        url = f"{ABC_S3_HTTP}/releases/{release}/manifest.json"
        if cache_dir is not None:
            local = Path(cache_dir) / "releases" / release / "manifest.json"
            if local.exists():
                return cls(json.loads(local.read_text()))
        req = urllib.request.Request(url, headers={"User-Agent": "gwas_ct/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        if cache_dir is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(data))
        return cls(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AbcManifest":
        return cls(json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------- parsing
    def _iter_files(self) -> Iterable[ManifestFile]:
        fl = self._m.get("file_listing", {})
        for dataset, kinds in fl.items():
            for kind, names in kinds.items():
                for name, entry in names.items():
                    if "files" in entry:  # metadata / image volumes / mapmycells
                        for ext, f in entry["files"].items():
                            yield ManifestFile(dataset, kind, name, None, ext, f["url"],
                                               f["relative_path"], int(f.get("size", 0)), f.get("file_hash", ""))
                    else:  # expression matrices: variant -> files
                        for variant, sub in entry.items():
                            for ext, f in sub.get("files", {}).items():
                                yield ManifestFile(dataset, kind, name, variant, ext, f["url"],
                                                   f["relative_path"], int(f.get("size", 0)), f.get("file_hash", ""))

    def datasets(self) -> List[str]:
        return sorted(self._m.get("directory_listing", {}).keys())

    def files(self, dataset: Optional[str] = None, kind: Optional[str] = None) -> List[ManifestFile]:
        return [f for f in self._files
                if (dataset is None or f.dataset == dataset) and (kind is None or f.kind == kind)]

    def get(self, dataset: str, kind: str, name: str, variant: Optional[str] = None,
            ext: Optional[str] = None) -> ManifestFile:
        """Resolve a single file; raises KeyError with a helpful message."""
        cands = [f for f in self._files if f.dataset == dataset and f.kind == kind and f.name == name
                 and (variant is None or f.variant == variant) and (ext is None or f.ext == ext)]
        if not cands:
            names = sorted({f.name for f in self._files if f.dataset == dataset and f.kind == kind})
            raise KeyError(f"No file {dataset}/{kind}/{name}(variant={variant}); available names: {names[:20]}")
        if len(cands) > 1 and variant is None:
            # prefer log2 for expression matrices
            for c in cands:
                if c.variant == "log2":
                    return c
        return cands[0]

    # ---------------------------------------------------------------- download
    @staticmethod
    def md5(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for buf in iter(lambda: fh.read(chunk), b""):
                h.update(buf)
        return h.hexdigest()

    def download(self, f: ManifestFile, download_base: Union[str, Path], max_bytes: Optional[int] = None,
                 verify_md5: bool = False, chunk: int = 1 << 20) -> Path:
        """Download `f` under `download_base/<relative_path>`; skip if complete."""
        dest = Path(download_base) / f.local_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and max_bytes is None and f.size and dest.stat().st_size == f.size:
            return dest
        req = urllib.request.Request(f.url, headers={"User-Agent": "gwas_ct/0.1"})
        written = 0
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
                if max_bytes is not None and written >= max_bytes:
                    break
        if verify_md5 and max_bytes is None and f.file_hash:
            got = self.md5(dest)
            if got != f.file_hash:
                raise IOError(f"MD5 mismatch for {dest}: {got} != {f.file_hash}")
        return dest


# ------------------------------------------------------------------------------------ metadata
def load_cell_metadata(path: Union[str, Path], columns: Optional[Sequence[str]] = None,
                       nrows: Optional[int] = None) -> pd.DataFrame:
    """Read an ABC `cell_metadata*.csv` indexed by `cell_label` with categorical label columns."""
    df = pd.read_csv(path, usecols=list(columns) + ["cell_label"] if columns else None, nrows=nrows)
    df = df.set_index("cell_label")
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() < max(1000, len(df) // 10):
            df[c] = df[c].astype("category")
    return df


def hierarchy_table(cell_meta: pd.DataFrame, levels: Sequence[str] = MOUSE_HIERARCHY) -> pd.DataFrame:
    """Unique mapping from the finest level to all coarser levels plus cell counts."""
    levels = [l for l in levels if l in cell_meta.columns]
    if not levels:
        raise ValueError(f"None of {levels} in metadata columns {list(cell_meta.columns)[:20]}")
    fine = levels[-1]
    counts = cell_meta.groupby(fine, observed=True).size().rename("n_cells")
    tab = cell_meta[levels].drop_duplicates(subset=[fine]).set_index(fine)
    return tab.join(counts)


# ------------------------------------------------------------------------------------ expression
def read_expression(path: Union[str, Path], backed: bool = True):
    """Open an ABC h5ad (cells x genes). Requires `anndata`. Backed mode avoids loading 30 GB."""
    try:
        import anndata as ad
    except ImportError as e:  # pragma: no cover
        raise ImportError("anndata is required to read ABC expression matrices: pip install anndata") from e
    return ad.read_h5ad(path, backed="r" if backed else None)


def pseudobulk(X: ArrayLike, labels: Union[Sequence, np.ndarray, pd.Series], gene_names: Sequence[str],
               min_cells: int = 10, agg: str = "mean") -> pd.DataFrame:
    """Aggregate a (cells x genes) matrix into (labels x genes) mean or sum expression.

    Works for dense numpy and scipy.sparse matrices via an indicator-matrix product, so it is
    O(nnz) and never densifies X.

    Parameters
    ----------
    X : (n_cells, n_genes) array or sparse matrix (log2 or raw counts, your choice).
    labels : per-cell label (cell type, spatial domain, or a combined "subclass|division" string).
    gene_names : column names for the output.
    min_cells : labels with fewer cells are dropped.
    agg : "mean" or "sum".
    """
    from scipy import sparse

    lab = pd.Categorical(np.asarray(labels))
    codes = lab.codes
    n_lab = len(lab.categories)
    n_cells = X.shape[0]
    ind = sparse.csr_matrix((np.ones(n_cells), (codes, np.arange(n_cells))), shape=(n_lab, n_cells))
    sums = ind @ X
    sums = np.asarray(sums.todense()) if sparse.issparse(sums) else np.asarray(sums)
    counts = np.bincount(codes, minlength=n_lab).astype(float)
    if agg == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = sums / counts[:, None]
    elif agg == "sum":
        out = sums
    else:
        raise ValueError("agg must be 'mean' or 'sum'")
    df = pd.DataFrame(out, index=lab.categories, columns=list(gene_names))
    df.index.name = "label"
    keep = counts >= min_cells
    return df.loc[keep]


def pseudobulk_backed(adata, labels: pd.Series, chunk: int = 50_000, min_cells: int = 10) -> pd.DataFrame:
    """Chunked pseudobulk *sum* over a backed anndata; returns mean per label.

    `labels` must be indexed by `adata.obs_names` (missing cells are ignored).
    """
    lab = labels.reindex(adata.obs_names)
    mask = lab.notna().to_numpy()
    cats = pd.Categorical(lab[mask])
    n_lab = len(cats.categories)
    n_genes = adata.n_vars
    sums = np.zeros((n_lab, n_genes))
    counts = np.zeros(n_lab)
    idx_all = np.flatnonzero(mask)
    code_all = cats.codes
    for start in range(0, len(idx_all), chunk):
        rows = idx_all[start:start + chunk]
        Xc = adata[rows].X
        Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
        codes = code_all[start:start + chunk]
        for k in np.unique(codes):
            sel = codes == k
            sums[k] += Xc[sel].sum(axis=0)
            counts[k] += sel.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]
    df = pd.DataFrame(means, index=cats.categories, columns=list(adata.var_names))
    return df.loc[counts >= min_cells]


def merfish_domain_labels(cell_meta_ccf: pd.DataFrame, level: str = "parcellation_division",
                          celltype_level: Optional[str] = "subclass", sep: str = "|") -> pd.Series:
    """Per-cell spatial-domain labels from the MERFISH `*_with_parcellation_annotation.csv` table.

    Returns `division` labels, or `celltype|division` combined labels when `celltype_level` is set.
    Cells outside the brain parcellation (NaN / 'unassigned') are dropped.
    """
    if level not in cell_meta_ccf.columns:
        raise KeyError(f"{level} not in columns; expected one of {MERFISH_DOMAIN_LEVELS}")
    dom = cell_meta_ccf[level].astype(str)
    ok = ~dom.isin(["nan", "unassigned", "brain-unassigned", ""])
    if celltype_level is None:
        return dom[ok]
    ct = cell_meta_ccf[celltype_level].astype(str)
    return (ct + sep + dom)[ok]


def split_combined_labels(index: pd.Index, sep: str = "|") -> pd.DataFrame:
    """Split 'celltype|domain' labels into two columns."""
    parts = [s.split(sep, 1) for s in index.astype(str)]
    return pd.DataFrame(parts, index=index, columns=["celltype", "domain"])
