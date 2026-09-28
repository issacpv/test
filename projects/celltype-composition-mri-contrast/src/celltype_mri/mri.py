"""Regional MRI contrast extraction from CCF-registered volumes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def load_volume(path: Path | str) -> np.ndarray:
    """Load NIfTI (nibabel) or NRRD (pynrrd) into a float array."""
    p = str(path)
    if p.endswith(".nrrd"):
        import nrrd

        data, _ = nrrd.read(p)
        return np.asarray(data, float)
    import nibabel as nib

    return np.asarray(nib.load(p).dataobj, dtype=float)


def regional_stats(volume: np.ndarray, annotation: np.ndarray, structure_ids: Iterable[int],
                   stat: str = "median", erode_voxels: int = 0, min_voxels: int = 20) -> pd.Series:
    """Per-structure summary of a contrast volume on the same grid as ``annotation``.

    ``erode_voxels`` > 0 removes the outer voxel layer(s) of each structure to limit
    partial-volume contamination (uses scipy binary erosion).
    """
    if volume.shape != annotation.shape:
        raise ValueError(f"shape mismatch {volume.shape} vs {annotation.shape}")
    fn = {"median": np.nanmedian, "mean": np.nanmean}[stat]
    out = {}
    if erode_voxels > 0:
        from scipy.ndimage import binary_erosion
    for sid in structure_ids:
        m = annotation == sid
        if erode_voxels > 0 and m.sum() > min_voxels:
            m = binary_erosion(m, iterations=erode_voxels)
        if m.sum() >= min_voxels:
            out[int(sid)] = float(fn(volume[m]))
    return pd.Series(out, name=stat)


def contrast_table(volumes: Dict[str, np.ndarray], annotation: np.ndarray, structure_ids: Iterable[int], **kw
                   ) -> pd.DataFrame:
    """Columns = contrast names, rows = structures."""
    ids = list(structure_ids)
    return pd.DataFrame({name: regional_stats(v, annotation, ids, **kw) for name, v in volumes.items()})


def t1w_t2w_ratio(t1w: np.ndarray, t2w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return t1w / (t2w + eps)


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1.0)


def align_tables(densities: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both tables to their common structures (index intersection), same order."""
    idx = densities.index.intersection(contrasts.index)
    return densities.loc[idx], contrasts.loc[idx]


def structure_centroids(annotation: np.ndarray, structure_ids: Iterable[int], resolution_um: float) -> pd.DataFrame:
    """Centroid (mm) per structure for spatial nulls."""
    rows = {}
    for sid in structure_ids:
        idx = np.argwhere(annotation == sid)
        if len(idx):
            rows[int(sid)] = idx.mean(axis=0) * resolution_um / 1000.0
    return pd.DataFrame.from_dict(rows, orient="index", columns=["x_mm", "y_mm", "z_mm"])
