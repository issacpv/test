"""HCP normative connectome loading and seed-based connectivity maps.

The normative-connectome logic of stimulation network mapping (Fox et al. 2012,
*Biol Psychiatry*; Siddiqi et al. 2022, *Nat Rev Neurosci*) is: take a group-average
functional connectome from a large healthy cohort, seed it at the stimulation
site, and use the resulting map to predict clinical effect. This module provides
the connectome side of that pipeline in a form that composes with the E-field side
(:mod:`tms_target.efield`).

Two representations are supported:

* **Parcellated** (``n_parcels x n_parcels``), e.g. HCP-MMP1 (Glasser et al. 2016,
  *Nature*) or Schaefer parcellations. This is what the statistics run on, and
  what the spatial nulls in :mod:`tms_target.nulls` require, because a spin test
  needs parcel centroid coordinates.
* **Dense** (grayordinate ``.dconn`` CIFTI, ~91k x 91k). Only ever touched one row
  at a time -- the full S1200 group-average dense connectome is ~33 GB and must
  not be loaded into memory. Requires ``nibabel``.

Functional connectivity is averaged and compared in Fisher-z space throughout.
Averaging raw Pearson r across subjects is biased toward zero, and the bias is
not uniform across the connectome, so it distorts exactly the spatial pattern
these analyses depend on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

__all__ = [
    "Connectome",
    "load_parcellated_connectome",
    "load_parcel_coords",
    "fisher_z",
    "inverse_fisher_z",
    "average_connectomes",
    "seed_connectivity",
    "dense_seed_connectivity",
    "structural_to_weights",
]

LOG = logging.getLogger(__name__)

ConnKind = Literal["functional", "structural"]


@dataclass
class Connectome:
    """A parcellated connectivity matrix with the geometry needed for nulls.

    Attributes:
        matrix: (p, p) symmetric connectivity. For functional connectomes this is
            Fisher-z transformed correlation; for structural it is a monotone
            transform of streamline count (see :func:`structural_to_weights`).
        labels: (p,) parcel names, in matrix order.
        coords: (p, 3) parcel centroid coordinates in mm (MNI or fsaverage).
            Required for variogram nulls and for distance-controlled analyses.
        kind: ``"functional"`` or ``"structural"``.
        hemisphere: (p,) array of ``"L"``/``"R"``/``"subcortical"``, used to keep
            spin tests within hemisphere.
        metadata: Provenance (source file, HCP release, parcellation, n subjects).
    """

    matrix: np.ndarray
    labels: np.ndarray
    coords: np.ndarray | None = None
    kind: ConnKind = "functional"
    hemisphere: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"matrix must be square, got {self.matrix.shape}")
        self.labels = np.asarray(self.labels)
        if self.labels.shape != (self.n_parcels,):
            raise ValueError(
                f"labels must have length {self.n_parcels}, got {self.labels.shape}"
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.coords.shape != (self.n_parcels, 3):
                raise ValueError(f"coords must be ({self.n_parcels}, 3), got {self.coords.shape}")
        if self.hemisphere is not None:
            self.hemisphere = np.asarray(self.hemisphere)

    @property
    def n_parcels(self) -> int:
        return int(self.matrix.shape[0])

    def index_of(self, label: str) -> int:
        """Return the matrix index of a parcel by name.

        Args:
            label: Parcel name, matched exactly then case-insensitively.

        Returns:
            Row/column index.

        Raises:
            KeyError: If the label is not present.
        """
        exact = np.flatnonzero(self.labels == label)
        if exact.size:
            return int(exact[0])
        lowered = np.char.lower(self.labels.astype(str))
        loose = np.flatnonzero(lowered == label.lower())
        if loose.size:
            return int(loose[0])
        raise KeyError(f"parcel {label!r} not found; first labels: {list(self.labels[:5])}")

    def distance_matrix(self) -> np.ndarray:
        """Euclidean parcel-centroid distance matrix in mm.

        Returns:
            (p, p) distances.

        Raises:
            ValueError: If ``coords`` was not provided.
        """
        if self.coords is None:
            raise ValueError("coords are required for a distance matrix")
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def symmetrized(self) -> "Connectome":
        """Return a copy with ``(M + M.T) / 2`` and a zero diagonal."""
        m = 0.5 * (self.matrix + self.matrix.T)
        np.fill_diagonal(m, 0.0)
        return Connectome(m, self.labels, self.coords, self.kind, self.hemisphere, dict(self.metadata))


def fisher_z(r: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Fisher z transform with clipping so ``r = +/-1`` stays finite.

    Args:
        r: Correlation values.
        eps: Clip distance from +/-1.

    Returns:
        ``arctanh`` of the clipped input.
    """
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -1.0 + eps, 1.0 - eps))


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher z transform (``tanh``)."""
    return np.tanh(np.asarray(z, dtype=float))


def load_parcellated_connectome(
    matrix_path: str | Path,
    labels_path: str | Path | None = None,
    coords_path: str | Path | None = None,
    *,
    kind: ConnKind = "functional",
    already_fisher_z: bool = False,
    **metadata: object,
) -> Connectome:
    """Load a parcellated connectome from ``.npy``, ``.csv`` or whitespace text.

    Args:
        matrix_path: Square matrix file.
        labels_path: One parcel name per line; defaults to ``parcel_0000`` style.
        coords_path: (p, 3) centroid coordinates, same formats as the matrix.
        kind: ``"functional"`` or ``"structural"``.
        already_fisher_z: If ``False`` and ``kind == "functional"``, the matrix is
            assumed to hold Pearson r and is Fisher-z transformed on load.
        **metadata: Recorded on the returned object.

    Returns:
        A :class:`Connectome`.

    Raises:
        FileNotFoundError: If ``matrix_path`` does not exist.
        ValueError: If shapes are inconsistent.
    """
    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    matrix = _load_array(matrix_path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_path} is not a square matrix: {matrix.shape}")

    if kind == "functional" and not already_fisher_z:
        finite = matrix[np.isfinite(matrix)]
        if finite.size and np.abs(finite).max() <= 1.0 + 1e-9:
            matrix = fisher_z(matrix)
        else:
            LOG.warning("%s has |values| > 1; assuming it is already z-scored", matrix_path)

    p = matrix.shape[0]
    if labels_path is not None:
        labels = np.array(
            [ln.strip() for ln in Path(labels_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        )
        if labels.size != p:
            raise ValueError(f"{labels_path} has {labels.size} labels for a {p}-parcel matrix")
    else:
        labels = np.array([f"parcel_{i:04d}" for i in range(p)])

    coords = None
    if coords_path is not None:
        coords = _load_array(Path(coords_path))
        if coords.shape != (p, 3):
            raise ValueError(f"{coords_path} must be ({p}, 3), got {coords.shape}")

    hemi = np.array(
        ["L" if str(lab).startswith(("L_", "lh")) else "R" if str(lab).startswith(("R_", "rh")) else "other"
         for lab in labels]
    )
    meta = {"matrix_path": str(matrix_path), **metadata}
    return Connectome(matrix, labels, coords, kind, hemi, meta)


def _load_array(path: Path) -> np.ndarray:
    """Load a numeric array from ``.npy``, ``.csv``/``.tsv`` or whitespace text."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if suffix == ".npz":
        with np.load(path) as bundle:
            return np.asarray(bundle[list(bundle.files)[0]], dtype=float)
    delimiter = "," if suffix == ".csv" else "\t" if suffix in (".tsv", ".tab") else None
    return np.loadtxt(path, delimiter=delimiter, dtype=float)


def load_parcel_coords(path: str | Path) -> np.ndarray:
    """Load (p, 3) parcel centroid coordinates."""
    coords = _load_array(Path(path))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected (p, 3) coordinates, got {coords.shape}")
    return coords


def average_connectomes(connectomes: Sequence[Connectome]) -> Connectome:
    """Average several connectomes in z space, preserving labels and coordinates.

    Use this to build a normative group connectome from per-subject matrices, or
    to compare a normative connectome against the mean of an individualized set.

    Args:
        connectomes: Two or more connectomes with identical labels.

    Returns:
        The element-wise mean connectome, with ``n_inputs`` in its metadata.

    Raises:
        ValueError: If fewer than one connectome is given, or labels disagree.
    """
    if not connectomes:
        raise ValueError("no connectomes to average")
    ref = connectomes[0]
    for c in connectomes[1:]:
        if not np.array_equal(c.labels, ref.labels):
            raise ValueError("connectomes must share the same parcel labels")
    stack = np.stack([c.matrix for c in connectomes], axis=0)
    mean = np.nanmean(stack, axis=0)
    meta = dict(ref.metadata)
    meta["n_inputs"] = len(connectomes)
    return Connectome(mean, ref.labels, ref.coords, ref.kind, ref.hemisphere, meta)


def seed_connectivity(
    conn: Connectome,
    seed: int | str | Iterable[int],
    *,
    weights: np.ndarray | None = None,
    exclude_seed: bool = True,
) -> np.ndarray:
    """Seed-based connectivity map from a parcellated connectome.

    With a single seed this is just the corresponding row. With several seeds it
    is the weighted mean of their rows -- which is how an **E-field-weighted seed**
    is built: pass every parcel as a seed and the normalised per-parcel field
    magnitude as ``weights``. That single generalization is what turns a
    point-seed network map into a dose-aware one.

    Args:
        conn: Parcellated connectome.
        seed: Parcel index, parcel name, or an iterable of indices.
        weights: (k,) weights for multiple seeds, or (p,) when ``seed`` is every
            parcel. Normalised to sum to 1 internally.
        exclude_seed: Set the seed parcels' own entries to ``nan`` so they cannot
            drive a correlation with a map that also peaks there.

    Returns:
        (p,) connectivity map in the connectome's units (Fisher z for functional).

    Raises:
        ValueError: If ``weights`` has the wrong length or sums to zero.
    """
    if isinstance(seed, str):
        indices = np.array([conn.index_of(seed)])
    elif isinstance(seed, (int, np.integer)):
        indices = np.array([int(seed)])
    else:
        indices = np.asarray(list(seed), dtype=int)
    if indices.size == 0:
        raise ValueError("no seed parcels given")
    if indices.min() < 0 or indices.max() >= conn.n_parcels:
        raise ValueError(f"seed indices out of range for {conn.n_parcels} parcels")

    rows = conn.matrix[indices]
    if weights is None:
        w = np.ones(indices.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != indices.shape:
            raise ValueError(f"weights must be {indices.shape}, got {w.shape}")
    total = np.nansum(w)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("seed weights must be positive and sum to a finite value")
    w = w / total

    out = np.nansum(rows * w[:, None], axis=0)
    if exclude_seed:
        out = out.astype(float)
        out[indices] = np.nan
    return out


def dense_seed_connectivity(
    dconn_path: str | Path, seed_indices: Sequence[int], *, chunk: int = 1
) -> np.ndarray:
    """Extract and average rows of a dense CIFTI connectome without loading it.

    The HCP S1200 MSM-All group-average dense connectome is tens of gigabytes.
    ``nibabel`` memory-maps it, so reading a handful of rows is cheap; reading it
    all is not. Never call ``np.asarray`` on the whole dataobj.

    Args:
        dconn_path: Path to a ``.dconn.nii`` CIFTI file.
        seed_indices: Grayordinate indices to average.
        chunk: Rows read per slice operation.

    Returns:
        (n_grayordinates,) mean seed map, Fisher-z transformed.

    Raises:
        ImportError: If ``nibabel`` is not installed.
        ValueError: If no seed indices are given.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "dense_seed_connectivity requires nibabel: pip install nibabel"
        ) from exc
    idx = np.asarray(list(seed_indices), dtype=int)
    if idx.size == 0:
        raise ValueError("no seed indices given")

    img = nib.load(str(dconn_path))
    dataobj = img.dataobj
    acc = np.zeros(dataobj.shape[1], dtype=np.float64)
    for start in range(0, idx.size, max(chunk, 1)):
        block = idx[start : start + max(chunk, 1)]
        acc += np.asarray(dataobj[block, :], dtype=np.float64).sum(axis=0)
    return fisher_z(acc / idx.size)


def structural_to_weights(
    streamlines: np.ndarray,
    *,
    transform: Literal["log", "sqrt", "none"] = "log",
    normalize: bool = True,
) -> np.ndarray:
    """Convert a streamline-count matrix into analysis-ready weights.

    Raw streamline counts from probabilistic tractography span several orders of
    magnitude and are strongly distance-biased, so a log transform is standard
    before any correlation with a spatially smooth map such as an E-field.

    Args:
        streamlines: (p, p) non-negative counts.
        transform: ``"log"`` applies ``log1p``, ``"sqrt"`` the square root.
        normalize: Scale the result to a maximum of 1.

    Returns:
        (p, p) symmetric weights with a zero diagonal.

    Raises:
        ValueError: If the input is not square or contains negative values.
    """
    m = np.asarray(streamlines, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"streamlines must be square, got {m.shape}")
    if np.nanmin(m) < 0:
        raise ValueError("streamline counts must be non-negative")
    if transform == "log":
        m = np.log1p(m)
    elif transform == "sqrt":
        m = np.sqrt(m)
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    peak = np.nanmax(m)
    if normalize and peak > 0:
        m = m / peak
    return m
