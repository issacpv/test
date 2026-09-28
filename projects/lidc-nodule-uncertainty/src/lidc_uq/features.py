"""Image feature extraction for malignancy and disagreement prediction.

Two backends:

* **numpy/scipy fallback** (:func:`extract_simple_features`) -- always available.
  Computes shape, attenuation and boundary-sharpness descriptors chosen
  specifically because they are the plausible *drivers of reader disagreement*, not
  just of malignancy. A nodule with a blurred, low-gradient boundary and
  heterogeneous attenuation near a management threshold is exactly the sort that
  four radiologists will score differently, so the feature set is built around
  boundary-gradient statistics, attenuation heterogeneity and threshold proximity.
* **pyradiomics** (:func:`extract_pyradiomics_features`) -- optional
  (van Griethuysen et al. 2017, *Cancer Research*). Use it for the published
  analysis; use the fallback for development, CI and any environment where the
  dependency chain will not install.

All features are computed **inside the consensus mask** and in a dilated boundary
shell around it. Spacing is always honoured, so features are physical rather than
voxel-count quantities -- essential when combining LIDC (mixed slice thickness,
0.6-5 mm) with a screening cohort standardised near 1 mm.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = [
    "extract_simple_features",
    "extract_pyradiomics_features",
    "SIMPLE_FEATURE_NAMES",
    "feature_frame",
    "resample_to_isotropic",
]

LOG = logging.getLogger(__name__)

#: Feature names produced by :func:`extract_simple_features`, in a stable order.
SIMPLE_FEATURE_NAMES: tuple[str, ...] = (
    # shape
    "volume_mm3",
    "surface_area_mm2",
    "equivalent_diameter_mm",
    "sphericity",
    "surface_to_volume",
    "elongation",
    "flatness",
    "extent",
    # attenuation inside the nodule
    "hu_mean",
    "hu_sd",
    "hu_p10",
    "hu_p50",
    "hu_p90",
    "hu_iqr",
    "hu_skew",
    "hu_kurtosis",
    "hu_entropy",
    # boundary sharpness - the disagreement-relevant block
    "boundary_gradient_mean",
    "boundary_gradient_sd",
    "boundary_gradient_p90",
    "inner_outer_hu_contrast",
    "boundary_hu_sd",
    "rim_transition_width_mm",
    # texture (co-occurrence, one offset per axis, averaged)
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_entropy",
    # threshold proximity
    "diameter_distance_to_6mm",
    "diameter_distance_to_8mm",
    "diameter_distance_to_nearest_threshold",
)

#: Lung-RADS solid-nodule size thresholds in mm, used for the proximity features.
LUNG_RADS_THRESHOLDS_MM: tuple[float, ...] = (6.0, 8.0, 15.0)


def resample_to_isotropic(
    image: np.ndarray,
    spacing_mm: Sequence[float],
    *,
    target_mm: float = 1.0,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample a volume to isotropic spacing.

    LIDC slice thickness ranges from 0.6 to 5 mm. Texture features computed at
    native spacing are therefore not comparable across scans, and the variation is
    correlated with scanner and era -- a textbook route to a model that learns
    acquisition rather than biology. Resample before extracting texture, and record
    the original spacing as a covariate regardless.

    Args:
        image: Input volume.
        spacing_mm: Spacing per axis, same order as ``image``.
        target_mm: Desired isotropic spacing.
        order: Spline interpolation order; use 0 for masks, 1 for images.

    Returns:
        ``(resampled, new_spacing)``.

    Raises:
        ValueError: If spacing length mismatches or contains non-positive values.
    """
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.size != image.ndim:
        raise ValueError(f"spacing has {spacing.size} entries for a {image.ndim}D image")
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError(f"spacing must be positive and finite, got {spacing}")
    zoom = spacing / float(target_mm)
    out = ndimage.zoom(image, zoom, order=order)
    return out, (target_mm,) * image.ndim  # type: ignore[return-value]


def extract_simple_features(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    n_gray_levels: int = 32,
    hu_window: tuple[float, float] = (-1000.0, 400.0),
) -> dict[str, float]:
    """Extract dependency-free shape, intensity, boundary and texture features.

    Args:
        image: CT volume in Hounsfield units.
        mask: Boolean nodule mask, same shape as ``image``.
        spacing_mm: Voxel spacing per axis, same order as the arrays.
        n_gray_levels: Quantisation levels for the co-occurrence texture features.
        hu_window: HU range used for quantisation. The default spans lung
            parenchyma to soft tissue/calcification.

    Returns:
        Dict keyed by :data:`SIMPLE_FEATURE_NAMES`. Features that cannot be
        computed (e.g. a mask touching the volume edge) are ``nan`` rather than
        silently zero.

    Raises:
        ValueError: If shapes disagree, the mask is empty, or parameters are invalid.
    """
    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask).astype(bool)
    if image.shape != mask.shape:
        raise ValueError(f"image {image.shape} and mask {mask.shape} must match")
    if not mask.any():
        raise ValueError("mask is empty")
    if n_gray_levels < 2:
        raise ValueError("n_gray_levels must be >= 2")
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.size != image.ndim:
        raise ValueError(f"spacing has {spacing.size} entries for a {image.ndim}D image")
    if np.any(spacing <= 0):
        raise ValueError("spacing must be positive")

    out: dict[str, float] = {name: float("nan") for name in SIMPLE_FEATURE_NAMES}
    voxel_mm3 = float(np.prod(spacing))

    # ------------------------------------------------------------------- shape
    n_vox = int(mask.sum())
    volume = n_vox * voxel_mm3
    area = _surface_area_mm2(mask, spacing)
    out["volume_mm3"] = volume
    out["surface_area_mm2"] = area
    eq_diam = float((6.0 * volume / np.pi) ** (1.0 / 3.0))
    out["equivalent_diameter_mm"] = eq_diam
    if area > 0:
        # sphericity = (36 pi V^2)^(1/3) / A, equal to 1 for a perfect sphere
        out["sphericity"] = float((36.0 * np.pi * volume**2) ** (1.0 / 3.0) / area)
        out["surface_to_volume"] = float(area / volume) if volume > 0 else float("nan")

    axes = _principal_axis_lengths(mask, spacing)
    if axes is not None and axes[0] > 0:
        major, mid, minor = axes
        out["elongation"] = float(mid / major)
        out["flatness"] = float(minor / major)
    bbox = ndimage.find_objects(mask.astype(np.uint8))
    if bbox and bbox[0] is not None:
        bbox_vox = int(np.prod([s.stop - s.start for s in bbox[0]]))
        out["extent"] = float(n_vox / bbox_vox) if bbox_vox > 0 else float("nan")

    # --------------------------------------------------------------- intensity
    inside = image[mask]
    inside = inside[np.isfinite(inside)]
    if inside.size:
        out["hu_mean"] = float(inside.mean())
        out["hu_sd"] = float(inside.std(ddof=1)) if inside.size > 1 else 0.0
        p10, p25, p50, p75, p90 = np.percentile(inside, [10, 25, 50, 75, 90])
        out["hu_p10"], out["hu_p50"], out["hu_p90"] = float(p10), float(p50), float(p90)
        out["hu_iqr"] = float(p75 - p25)
        out["hu_skew"] = _moment(inside, 3)
        out["hu_kurtosis"] = _moment(inside, 4)
        out["hu_entropy"] = _histogram_entropy(inside, n_gray_levels, hu_window)

    # ------------------------------------------------- boundary sharpness block
    boundary = _boundary_shell(mask)
    outer = _outer_shell(mask, thickness=2)
    gradient = _gradient_magnitude(image, spacing)
    if boundary.any():
        gb = gradient[boundary]
        gb = gb[np.isfinite(gb)]
        if gb.size:
            out["boundary_gradient_mean"] = float(gb.mean())
            out["boundary_gradient_sd"] = float(gb.std(ddof=1)) if gb.size > 1 else 0.0
            out["boundary_gradient_p90"] = float(np.percentile(gb, 90))
        hb = image[boundary]
        hb = hb[np.isfinite(hb)]
        if hb.size > 1:
            out["boundary_hu_sd"] = float(hb.std(ddof=1))
    if outer.any() and inside.size:
        outside = image[outer]
        outside = outside[np.isfinite(outside)]
        if outside.size:
            out["inner_outer_hu_contrast"] = float(inside.mean() - outside.mean())
            # A sharp margin traverses the inside/outside HU gap over a short
            # distance; width = contrast / peak gradient, in mm.
            peak_grad = out["boundary_gradient_p90"]
            if np.isfinite(peak_grad) and peak_grad > 0:
                out["rim_transition_width_mm"] = float(
                    abs(inside.mean() - outside.mean()) / peak_grad
                )

    # ----------------------------------------------------------------- texture
    texture = _glcm_features(image, mask, n_gray_levels, hu_window)
    out.update(texture)

    # ------------------------------------------------------ threshold proximity
    out["diameter_distance_to_6mm"] = float(eq_diam - 6.0)
    out["diameter_distance_to_8mm"] = float(eq_diam - 8.0)
    out["diameter_distance_to_nearest_threshold"] = float(
        min(abs(eq_diam - t) for t in LUNG_RADS_THRESHOLDS_MM)
    )
    return out


def _moment(values: np.ndarray, order: int) -> float:
    """Standardised central moment; 3 = skewness, 4 = kurtosis (not excess)."""
    if values.size < 2:
        return float("nan")
    sd = values.std()
    if sd <= 0:
        return 0.0
    return float((((values - values.mean()) / sd) ** order).mean())


def _histogram_entropy(
    values: np.ndarray, n_levels: int, window: tuple[float, float]
) -> float:
    """Shannon entropy of a windowed intensity histogram, in bits."""
    lo, hi = window
    clipped = np.clip(values, lo, hi)
    hist, _ = np.histogram(clipped, bins=n_levels, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return float("nan")
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def _surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Surface area by counting exposed voxel faces, weighted by face area.

    A face-counting estimate overestimates a smooth sphere's area by ~50%, so
    ``sphericity`` built on it is not on the same scale as pyradiomics'
    mesh-based value. It is monotone in the same quantity and adequate for a
    fallback; the published analysis should use the pyradiomics backend.
    """
    area = 0.0
    for axis in range(mask.ndim):
        face = float(np.prod(np.delete(spacing, axis)))
        shifted = np.roll(mask, 1, axis=axis)
        idx: list[slice | int] = [slice(None)] * mask.ndim
        idx[axis] = 0
        shifted[tuple(idx)] = False
        area += float((mask & ~shifted).sum()) * face

        shifted = np.roll(mask, -1, axis=axis)
        idx[axis] = -1
        shifted[tuple(idx)] = False
        area += float((mask & ~shifted).sum()) * face
    return area


def _principal_axis_lengths(mask: np.ndarray, spacing: np.ndarray) -> tuple[float, float, float] | None:
    """Sorted principal axis lengths (descending) from the mask's inertia tensor."""
    coords = np.argwhere(mask).astype(float) * spacing
    if coords.shape[0] < 4 or coords.shape[1] != 3:
        return None
    centred = coords - coords.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    lengths = 4.0 * np.sqrt(np.maximum(eigvals, 0.0))  # ~ axis lengths of an ellipsoid
    major, mid, minor = sorted(lengths, reverse=True)
    return float(major), float(mid), float(minor)


def _boundary_shell(mask: np.ndarray) -> np.ndarray:
    """Inner boundary voxels: in the mask but adjacent to something outside it."""
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return mask & ~eroded


def _outer_shell(mask: np.ndarray, *, thickness: int = 2) -> np.ndarray:
    """A shell of ``thickness`` voxels just outside the mask."""
    dilated = ndimage.binary_dilation(mask, iterations=max(thickness, 1), border_value=0)
    return dilated & ~mask


def _gradient_magnitude(image: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Physical-units gradient magnitude in HU per mm."""
    grads = np.gradient(image, *[float(s) for s in spacing])
    if isinstance(grads, np.ndarray):
        grads = [grads]
    return np.sqrt(sum(g**2 for g in grads))


def _glcm_features(
    image: np.ndarray, mask: np.ndarray, n_levels: int, window: tuple[float, float]
) -> dict[str, float]:
    """Grey-level co-occurrence features averaged over one offset per axis.

    Only voxel pairs with **both** members inside the mask are counted, so the
    features describe internal texture rather than the nodule/parenchyma step.
    """
    lo, hi = window
    q = np.clip(image, lo, hi)
    q = ((q - lo) / (hi - lo) * (n_levels - 1)).round().astype(np.int16)
    q = np.clip(q, 0, n_levels - 1)

    accum = np.zeros((n_levels, n_levels), dtype=np.float64)
    for axis in range(image.ndim):
        a_idx: list[slice] = [slice(None)] * image.ndim
        b_idx: list[slice] = [slice(None)] * image.ndim
        a_idx[axis] = slice(None, -1)
        b_idx[axis] = slice(1, None)
        both = mask[tuple(a_idx)] & mask[tuple(b_idx)]
        if not both.any():
            continue
        a = q[tuple(a_idx)][both]
        b = q[tuple(b_idx)][both]
        hist, _, _ = np.histogram2d(a, b, bins=n_levels, range=[[0, n_levels - 1]] * 2)
        accum += hist + hist.T  # symmetrise

    total = accum.sum()
    if total == 0:
        return {k: float("nan") for k in ("glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy")}
    p = accum / total
    i, j = np.indices(p.shape)
    diff = (i - j).astype(float)
    nz = p[p > 0]
    return {
        "glcm_contrast": float((p * diff**2).sum()),
        "glcm_homogeneity": float((p / (1.0 + np.abs(diff))).sum()),
        "glcm_energy": float((p**2).sum()),
        "glcm_entropy": float(-(nz * np.log2(nz)).sum()),
    }


def extract_pyradiomics_features(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    params: dict[str, object] | None = None,
) -> dict[str, float]:
    """Extract features with pyradiomics, if it is installed.

    Args:
        image: CT volume in HU.
        mask: Boolean nodule mask.
        spacing_mm: Voxel spacing in the array's axis order (z, y, x).
        params: Extractor settings. Defaults enable shape, first-order, GLCM, GLRLM
            and GLSZM with a fixed bin width of 25 HU and 1 mm isotropic
            resampling -- a fixed **bin width** rather than bin count, because a
            fixed count makes the discretisation depend on each nodule's own
            intensity range and is a known source of non-reproducibility.

    Returns:
        Dict of pyradiomics feature names to values, ``diagnostics_`` keys dropped.

    Raises:
        ImportError: If ``pyradiomics`` or ``SimpleITK`` is not installed.
    """
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "extract_pyradiomics_features requires pyradiomics and SimpleITK: "
            "pip install pyradiomics SimpleITK. The numpy fallback "
            "extract_simple_features() needs neither."
        ) from exc

    settings: dict[str, object] = {
        "binWidth": 25,
        "resampledPixelSpacing": [1.0, 1.0, 1.0],
        "interpolator": "sitkBSpline",
        "label": 1,
        "normalize": False,
    }
    settings.update(params or {})

    img = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
    msk = sitk.GetImageFromArray(np.asarray(mask).astype(np.uint8))
    # SimpleITK spacing is (x, y, z); our arrays are (z, y, x).
    sitk_spacing = tuple(float(s) for s in reversed(list(spacing_mm)))
    img.SetSpacing(sitk_spacing)
    msk.SetSpacing(sitk_spacing)

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    result = extractor.execute(img, msk)
    return {
        str(k): float(v)
        for k, v in result.items()
        if not str(k).startswith("diagnostics_") and np.isscalar(v)
    }


def feature_frame(
    records: Iterable[tuple[str, np.ndarray, np.ndarray, Sequence[float]]],
    *,
    backend: str = "simple",
    **kwargs: object,
) -> pd.DataFrame:
    """Extract features for many nodules into a single dataframe.

    Args:
        records: Iterable of ``(nodule_id, image, mask, spacing_mm)`` tuples.
        backend: ``"simple"`` or ``"pyradiomics"``.
        **kwargs: Forwarded to the extractor.

    Returns:
        A dataframe indexed by ``nodule_id``. A nodule whose extraction raises is
        logged and skipped rather than aborting a long run.

    Raises:
        ValueError: If ``backend`` is unknown.
    """
    if backend == "simple":
        extract = extract_simple_features
    elif backend == "pyradiomics":
        extract = extract_pyradiomics_features  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown backend {backend!r}")

    rows: list[dict[str, object]] = []
    for nodule_id, image, mask, spacing in records:
        try:
            feats = extract(image, mask, spacing, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - keep long extractions alive
            LOG.warning("feature extraction failed for %s: %s", nodule_id, exc)
            continue
        rows.append({"nodule_id": nodule_id, **feats})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("nodule_id")
