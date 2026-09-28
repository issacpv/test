"""E-field map loading and ROI dosing metrics.

This is the half of the pipeline that network-mapping papers almost always omit.
A seed-based connectivity map says *where the stimulation site is connected*; it
says nothing about **how much field actually reached that site**, or how much
reached everywhere else. Two coil placements with the same nominal target can
deliver peak field centimetres apart, and stimulator output is set by a motor
threshold measured in a different cortical region with different conductivity
geometry (see "Electric-field-based dosing for TMS", *Imaging Neuroscience* 2024,
doi:10.1162/imag_a_00106).

Supported inputs:

* **NIfTI** volumes from ROAST (Huang et al. 2019, *J Neural Eng*) or from
  SimNIBS ``--map-to-mni`` output: 3D scalar ``|E|``, or 4D vector field.
* **Gmsh ``.msh``** meshes from SimNIBS (Thielscher et al. 2015, *EMBC*;
  Puonti et al. 2020, *NeuroImage*), reading the ``normE`` element field or the
  vector ``E`` field, via ``meshio``.
* **Plain arrays**, so the metrics are testable and usable without either
  optional dependency.

Both loaders are optional-dependency-gated; everything below
:class:`EFieldMap` works on numpy arrays alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

__all__ = [
    "EFieldMap",
    "load_efield_nifti",
    "load_efield_msh",
    "roi_dose_metrics",
    "parcellate_field",
    "focality",
    "field_weighted_centroid",
    "normalize_to_motor_threshold",
]

LOG = logging.getLogger(__name__)


@dataclass
class EFieldMap:
    """A sampled electric-field magnitude map.

    Attributes:
        magnitude: (n,) field magnitude in V/m at each sample.
        coords: (n, 3) sample coordinates in mm, or ``None`` for voxel grids whose
            geometry is carried by ``affine`` instead.
        vectors: (n, 3) field vectors in V/m, when available. Needed for the
            normal-component analyses that matter for tDCS polarity.
        volumes: (n,) sample volumes in mm^3 for element-weighted integrals. Mesh
            elements differ in size by more than an order of magnitude, so
            unweighted means over elements are biased toward small elements.
        affine: 4x4 voxel-to-world affine for volumetric sources.
        space: Free-text coordinate space, e.g. ``"MNI152"`` or ``"subject"``.
        metadata: Provenance (source file, montage, stimulator output).
    """

    magnitude: np.ndarray
    coords: np.ndarray | None = None
    vectors: np.ndarray | None = None
    volumes: np.ndarray | None = None
    affine: np.ndarray | None = None
    space: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.magnitude = np.asarray(self.magnitude, dtype=float).ravel()
        n = self.magnitude.size
        if n == 0:
            raise ValueError("magnitude is empty")
        for name in ("coords", "vectors"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if arr.shape != (n, 3):
                    raise ValueError(f"{name} must be ({n}, 3), got {arr.shape}")
                setattr(self, name, arr)
        if self.volumes is not None:
            vol = np.asarray(self.volumes, dtype=float).ravel()
            if vol.size != n:
                raise ValueError(f"volumes must have length {n}, got {vol.size}")
            if np.nanmin(vol) < 0:
                raise ValueError("volumes must be non-negative")
            self.volumes = vol

    @property
    def n_samples(self) -> int:
        return int(self.magnitude.size)

    @property
    def weights(self) -> np.ndarray:
        """Integration weights: element volumes if present, else uniform."""
        if self.volumes is not None:
            return self.volumes
        return np.ones(self.n_samples, dtype=float)

    def peak_location(self) -> np.ndarray:
        """Coordinates of the maximum-magnitude sample.

        Returns:
            (3,) coordinates in mm.

        Raises:
            ValueError: If the map has no coordinates.
        """
        if self.coords is None:
            raise ValueError("peak_location requires coords")
        return self.coords[int(np.nanargmax(self.magnitude))]

    def scaled(self, factor: float) -> "EFieldMap":
        """Return a copy with magnitude (and vectors) multiplied by ``factor``.

        Field solutions are linear in stimulator output, so rescaling is exact --
        which is what makes dose normalization a cheap post-hoc operation.
        """
        return EFieldMap(
            magnitude=self.magnitude * factor,
            coords=self.coords,
            vectors=None if self.vectors is None else self.vectors * factor,
            volumes=self.volumes,
            affine=self.affine,
            space=self.space,
            metadata={**self.metadata, "scale_factor": factor},
        )


def load_efield_nifti(
    path: str | Path, *, mask_path: str | Path | None = None, space: str = "MNI152"
) -> EFieldMap:
    """Load an E-field from a NIfTI volume (ROAST, or SimNIBS mapped to a volume).

    A 3D volume is read as ``|E|``; a 4D volume with three components in the last
    axis is read as the vector field and its norm taken.

    Args:
        path: NIfTI file.
        mask_path: Optional NIfTI mask; only non-zero voxels are kept. Strongly
            recommended -- a grey-matter mask removes the scalp and skull voxels
            that otherwise dominate the peak.
        space: Coordinate space label.

    Returns:
        An :class:`EFieldMap` with per-voxel coordinates in world mm and voxel
        volumes as integration weights.

    Raises:
        ImportError: If ``nibabel`` is not installed.
        ValueError: If the volume has an unsupported shape or the mask mismatches.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("load_efield_nifti requires nibabel: pip install nibabel") from exc

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=float)
    affine = np.asarray(img.affine, dtype=float)

    if data.ndim == 4 and data.shape[-1] == 3:
        vectors_grid = data
        magnitude_grid = np.linalg.norm(data, axis=-1)
    elif data.ndim == 3:
        vectors_grid = None
        magnitude_grid = data
    elif data.ndim == 4 and data.shape[-1] == 1:
        vectors_grid = None
        magnitude_grid = data[..., 0]
    else:
        raise ValueError(f"unsupported E-field shape {data.shape}; expected 3D or 4D with 3 components")

    keep = np.isfinite(magnitude_grid) & (magnitude_grid != 0)
    if mask_path is not None:
        mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0
        if mask.shape != magnitude_grid.shape:
            raise ValueError(f"mask shape {mask.shape} != field shape {magnitude_grid.shape}")
        keep &= mask

    ijk = np.argwhere(keep)
    homogeneous = np.c_[ijk, np.ones(len(ijk))]
    coords = (homogeneous @ affine.T)[:, :3]
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    return EFieldMap(
        magnitude=magnitude_grid[keep],
        coords=coords,
        vectors=None if vectors_grid is None else vectors_grid[keep],
        volumes=np.full(len(ijk), voxel_mm3),
        affine=affine,
        space=space,
        metadata={"source": str(path), "mask": str(mask_path) if mask_path else None},
    )


def load_efield_msh(
    path: str | Path,
    *,
    field_name: str | None = None,
    tissue_tags: Sequence[int] | None = (2,),
    space: str = "subject",
) -> EFieldMap:
    """Load an E-field from a SimNIBS Gmsh ``.msh`` result file.

    SimNIBS writes tetrahedral element data named ``normE`` (magnitude) and ``E``
    (vector), tagged by tissue. Tag 2 is grey matter in the SimNIBS convention,
    which is nearly always the right restriction: field in skull and scalp is
    irrelevant to neural effect but would otherwise set the map's peak.

    Args:
        path: ``.msh`` file.
        field_name: Cell-data name to read. ``None`` tries ``normE`` then ``E``.
        tissue_tags: Element tags to keep; ``None`` keeps everything.
        space: Coordinate space label.

    Returns:
        An :class:`EFieldMap` sampled at element centroids, with element volumes
        as integration weights.

    Raises:
        ImportError: If ``meshio`` is not installed.
        ValueError: If no suitable field is found in the file.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("load_efield_msh requires meshio: pip install meshio") from exc

    mesh = meshio.read(str(path))
    candidates = [field_name] if field_name else ["normE", "E", "magnE"]
    chosen: str | None = None
    for name in candidates:
        if name and name in mesh.cell_data_dict:
            chosen = name
            break
    if chosen is None:
        raise ValueError(
            f"no E-field data in {path}; available cell data: {sorted(mesh.cell_data_dict)}"
        )

    mags: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    volumes: list[np.ndarray] = []
    points = np.asarray(mesh.points, dtype=float)

    for block_idx, block in enumerate(mesh.cells):
        if block.type not in ("tetra", "triangle"):
            continue
        values = np.asarray(mesh.cell_data_dict[chosen][block.type], dtype=float)
        conn = np.asarray(block.data, dtype=int)
        if values.shape[0] != conn.shape[0]:
            continue

        keep = np.ones(conn.shape[0], dtype=bool)
        if tissue_tags is not None:
            tags = _element_tags(mesh, block.type, block_idx)
            if tags is not None:
                keep = np.isin(tags, np.asarray(tissue_tags))
        if not keep.any():
            continue

        conn = conn[keep]
        values = values[keep]
        centroids.append(points[conn].mean(axis=1))
        if values.ndim == 2 and values.shape[1] == 3:
            vecs.append(values)
            mags.append(np.linalg.norm(values, axis=1))
        else:
            mags.append(values.ravel())
        volumes.append(
            _tetra_volumes(points, conn) if block.type == "tetra" else _triangle_areas(points, conn)
        )

    if not mags:
        raise ValueError(f"no elements survived tissue_tags={tissue_tags} in {path}")

    return EFieldMap(
        magnitude=np.concatenate(mags),
        coords=np.concatenate(centroids, axis=0),
        vectors=np.concatenate(vecs, axis=0) if len(vecs) == len(mags) else None,
        volumes=np.concatenate(volumes),
        space=space,
        metadata={"source": str(path), "field": chosen, "tissue_tags": list(tissue_tags or [])},
    )


def _element_tags(mesh: object, cell_type: str, block_idx: int) -> np.ndarray | None:
    """Best-effort extraction of Gmsh physical tags for one cell block."""
    for key in ("gmsh:physical", "medit:ref", "cell_tags"):
        data = getattr(mesh, "cell_data", {}).get(key)  # type: ignore[union-attr]
        if data is not None and block_idx < len(data):
            return np.asarray(data[block_idx]).ravel()
    dict_data = getattr(mesh, "cell_data_dict", {})
    for key in ("gmsh:physical", "medit:ref"):
        if key in dict_data and cell_type in dict_data[key]:
            return np.asarray(dict_data[key][cell_type]).ravel()
    return None


def _tetra_volumes(points: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Signed-absolute volumes of tetrahedra in mm^3."""
    a, b, c, d = (points[conn[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _triangle_areas(points: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Areas of triangles in mm^2, used as weights for surface fields."""
    a, b, c = (points[conn[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def roi_dose_metrics(
    field: EFieldMap,
    roi_mask: np.ndarray | None = None,
    *,
    percentiles: Sequence[float] = (50.0, 95.0, 99.0),
) -> dict[str, float]:
    """Volume-weighted dosing metrics inside an ROI.

    Reports both the mean and high percentiles, because they answer different
    questions: the mean is the right summary for a diffuse network effect, while
    the 99th percentile tracks the focal hotspot that drives suprathreshold
    activation. Papers that report only one of them are not comparable.

    Args:
        field: The field map.
        roi_mask: (n,) boolean mask over samples; ``None`` uses all samples.
        percentiles: Percentiles of ``|E|`` to report.

    Returns:
        Dict with ``n_samples``, ``volume_mm3``, ``mean_Vpm``, ``sd_Vpm``,
        ``max_Vpm``, ``p50_Vpm`` etc., and ``fraction_above_half_max``.

    Raises:
        ValueError: If the mask has the wrong length or selects nothing.
    """
    n = field.n_samples
    if roi_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(roi_mask, dtype=bool).ravel()
        if mask.size != n:
            raise ValueError(f"roi_mask must have length {n}, got {mask.size}")
    if not mask.any():
        raise ValueError("roi_mask selects no samples")

    mag = field.magnitude[mask]
    w = field.weights[mask]
    finite = np.isfinite(mag)
    mag, w = mag[finite], w[finite]
    if mag.size == 0:
        raise ValueError("no finite field values inside the ROI")

    total_w = w.sum()
    mean = float((mag * w).sum() / total_w)
    var = float((w * (mag - mean) ** 2).sum() / total_w)
    global_max = float(np.nanmax(field.magnitude))

    out: dict[str, float] = {
        "n_samples": float(mag.size),
        "volume_mm3": float(total_w),
        "mean_Vpm": mean,
        "sd_Vpm": float(np.sqrt(max(var, 0.0))),
        "max_Vpm": float(mag.max()),
        "fraction_above_half_max": float(w[mag >= 0.5 * global_max].sum() / total_w),
    }
    for q in percentiles:
        out[f"p{int(q)}_Vpm"] = float(_weighted_percentile(mag, w, q))
    return out


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Volume-weighted percentile of ``values``."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(q / 100.0, cum, v))


def focality(field: EFieldMap, *, threshold_fraction: float = 0.5) -> dict[str, float]:
    """Focality of a field map: the volume receiving at least a fraction of peak.

    A smaller half-max volume means a more focal montage or coil. This is the
    standard focality measure in the tES literature and the right covariate when
    comparing montages whose peak magnitudes have been matched.

    Args:
        field: The field map.
        threshold_fraction: Fraction of the peak defining the volume.

    Returns:
        Dict with ``peak_Vpm``, ``threshold_Vpm``, ``volume_mm3`` above threshold
        and ``fraction_of_total_volume``.

    Raises:
        ValueError: If ``threshold_fraction`` is outside (0, 1].
    """
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in (0, 1]")
    mag = field.magnitude
    w = field.weights
    peak = float(np.nanmax(mag))
    cut = threshold_fraction * peak
    above = np.isfinite(mag) & (mag >= cut)
    return {
        "peak_Vpm": peak,
        "threshold_Vpm": cut,
        "volume_mm3": float(w[above].sum()),
        "fraction_of_total_volume": float(w[above].sum() / w.sum()),
    }


def field_weighted_centroid(field: EFieldMap, *, power: float = 1.0) -> np.ndarray:
    """Field-magnitude-weighted centroid, a robust alternative to the peak voxel.

    The single peak voxel is noisy and mesh-dependent; the weighted centroid is
    stable and is the better "effective stimulation site" for seeding a
    connectivity map. Raising ``power`` concentrates the estimate toward the peak.

    Args:
        field: The field map.
        power: Exponent applied to magnitude before weighting.

    Returns:
        (3,) coordinates in mm.

    Raises:
        ValueError: If the map has no coordinates or zero total weight.
    """
    if field.coords is None:
        raise ValueError("field_weighted_centroid requires coords")
    mag = np.nan_to_num(field.magnitude, nan=0.0)
    w = (np.maximum(mag, 0.0) ** power) * field.weights
    total = w.sum()
    if total <= 0:
        raise ValueError("total field weight is zero")
    return (field.coords * w[:, None]).sum(axis=0) / total


def parcellate_field(
    field: EFieldMap,
    parcel_labels: np.ndarray,
    n_parcels: int | None = None,
    *,
    statistic: Literal["mean", "max", "p95"] = "mean",
) -> np.ndarray:
    """Reduce a sample-wise field map to one value per parcel.

    This is the bridge between the field and the connectome: both must live on the
    same parcellation before any joint model or spatial null can be run.

    Args:
        field: The field map.
        parcel_labels: (n,) integer parcel index per sample; negative values are
            treated as unassigned and ignored.
        n_parcels: Number of parcels; inferred from the labels when ``None``.
        statistic: Within-parcel summary. ``"mean"`` is volume-weighted.

    Returns:
        (n_parcels,) array, ``nan`` for parcels with no samples.

    Raises:
        ValueError: If lengths mismatch or the statistic is unknown.
    """
    labels = np.asarray(parcel_labels, dtype=int).ravel()
    if labels.size != field.n_samples:
        raise ValueError(f"parcel_labels must have length {field.n_samples}, got {labels.size}")
    p = int(n_parcels if n_parcels is not None else (labels.max() + 1 if labels.size else 0))
    if p <= 0:
        raise ValueError("could not determine a positive number of parcels")

    out = np.full(p, np.nan, dtype=float)
    mag = field.magnitude
    w = field.weights
    for parcel in range(p):
        sel = (labels == parcel) & np.isfinite(mag)
        if not sel.any():
            continue
        if statistic == "mean":
            out[parcel] = float((mag[sel] * w[sel]).sum() / w[sel].sum())
        elif statistic == "max":
            out[parcel] = float(mag[sel].max())
        elif statistic == "p95":
            out[parcel] = _weighted_percentile(mag[sel], w[sel], 95.0)
        else:
            raise ValueError(f"unknown statistic {statistic!r}")
    return out


def normalize_to_motor_threshold(
    field: EFieldMap,
    m1_mask: np.ndarray,
    *,
    target_Vpm: float = 100.0,
    statistic: Literal["p99", "mean", "max"] = "p99",
) -> EFieldMap:
    """Rescale a field so that the field in M1 equals a reference value.

    This implements E-field-based dosing: instead of expressing stimulator output
    as "% of resting motor threshold", express it as the field magnitude actually
    delivered to M1 at motor threshold, then scale every other target to match.
    It removes the coil-to-cortex-distance confound that plagues %RMT dosing and
    is the correction that connectivity-only target scoring silently omits.

    Args:
        field: The field map.
        m1_mask: (n,) boolean mask of the M1 ROI.
        target_Vpm: Field magnitude that M1 should receive after scaling.
        statistic: Which M1 summary to normalise.

    Returns:
        A rescaled copy of the field map.

    Raises:
        ValueError: If the M1 statistic is non-positive.
    """
    metrics = roi_dose_metrics(field, m1_mask, percentiles=(99.0,))
    key = {"p99": "p99_Vpm", "mean": "mean_Vpm", "max": "max_Vpm"}[statistic]
    current = metrics[key]
    if not np.isfinite(current) or current <= 0:
        raise ValueError(f"M1 {statistic} is {current}; cannot normalize")
    scaled = field.scaled(target_Vpm / current)
    scaled.metadata["dose_normalization"] = {
        "statistic": statistic,
        "m1_value_Vpm": current,
        "target_Vpm": target_Vpm,
    }
    return scaled
