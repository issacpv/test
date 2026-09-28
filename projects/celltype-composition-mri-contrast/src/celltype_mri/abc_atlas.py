"""From MERFISH cell tables to regional cell-type densities.

The ABC Atlas cell metadata gives every cell a CCF coordinate (mm) and taxonomy
labels. Densities per CCF structure are counts divided by the *sampled* structure
volume (MERFISH sections do not cover every voxel), which is estimated from the
annotation volume restricted to the sampled section planes.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def cells_to_voxels(xyz_mm: np.ndarray, resolution_um: float) -> np.ndarray:
    """CCF coordinates in mm -> integer voxel indices at ``resolution_um``."""
    return np.floor(np.asarray(xyz_mm, float) * 1000.0 / resolution_um).astype(int)


def assign_structures(xyz_mm: np.ndarray, annotation: np.ndarray, resolution_um: float) -> np.ndarray:
    """Structure id for every cell (0 when outside the annotation volume)."""
    v = cells_to_voxels(xyz_mm, resolution_um)
    ok = np.all(v >= 0, axis=1) & np.all(v < np.asarray(annotation.shape)[None, :], axis=1)
    out = np.zeros(len(v), dtype=int)
    out[ok] = annotation[v[ok, 0], v[ok, 1], v[ok, 2]]
    return out


def sampled_volume_mm3(annotation: np.ndarray, resolution_um: float, sampled_planes: Optional[Sequence[int]] = None,
                       axis: int = 0) -> Dict[int, float]:
    """Volume per structure id (mm^3), optionally restricted to sampled section planes along ``axis``."""
    if sampled_planes is not None:
        sl = [slice(None)] * annotation.ndim
        sl[axis] = np.asarray(sorted(set(sampled_planes)))
        ann = annotation[tuple(sl)]
    else:
        ann = annotation
    ids, counts = np.unique(ann, return_counts=True)
    vox_mm3 = (resolution_um / 1000.0) ** 3
    return {int(i): float(c * vox_mm3) for i, c in zip(ids, counts) if i != 0}


def densities_by_structure(cells: pd.DataFrame, structure_col: str = "structure_id", label_col: str = "class",
                           volumes_mm3: Optional[Dict[int, float]] = None, min_cells: int = 200
                           ) -> pd.DataFrame:
    """Wide table: rows = structures, columns = labels; values = density (cells/mm^3) or counts.

    Structures with fewer than ``min_cells`` total cells are dropped. When ``volumes_mm3``
    is None the table holds counts.
    """
    tab = pd.crosstab(cells[structure_col], cells[label_col])
    tab = tab[tab.sum(axis=1) >= min_cells]
    if volumes_mm3 is not None:
        vol = pd.Series(volumes_mm3).reindex(tab.index)
        tab = tab.div(vol, axis=0)
        tab = tab[np.isfinite(vol.to_numpy())]
    tab.columns = [str(c) for c in tab.columns]
    return tab


def aggregate_to_parent(tab: pd.DataFrame, structure_to_parent: Dict[int, int], volumes_mm3: Dict[int, float]
                        ) -> pd.DataFrame:
    """Volume-weighted aggregation of a density table from child structures to their parents."""
    vol = pd.Series({s: volumes_mm3.get(s, np.nan) for s in tab.index})
    parent = pd.Series({s: structure_to_parent.get(s, s) for s in tab.index})
    weighted = tab.mul(vol, axis=0)
    num = weighted.groupby(parent).sum()
    den = vol.groupby(parent).sum()
    return num.div(den, axis=0)


def log_density(tab: pd.DataFrame, pseudo: float = 1.0) -> pd.DataFrame:
    return np.log(tab + pseudo)


def clr(tab: pd.DataFrame, pseudo: float = 1.0) -> pd.DataFrame:
    """Centred log-ratio transform (compositional sensitivity analysis)."""
    L = np.log(tab + pseudo)
    return L.sub(L.mean(axis=1), axis=0)


def voxel_density_map(xyz_mm: np.ndarray, shape: Tuple[int, int, int], resolution_um: float,
                      labels: Optional[np.ndarray] = None, target: Optional[object] = None) -> np.ndarray:
    """Cells per voxel (optionally only cells whose ``labels == target``) on a CCF grid."""
    v = cells_to_voxels(xyz_mm, resolution_um)
    if labels is not None and target is not None:
        v = v[np.asarray(labels) == target]
    ok = np.all(v >= 0, axis=1) & np.all(v < np.asarray(shape)[None, :], axis=1)
    out = np.zeros(shape, dtype=np.int32)
    np.add.at(out, (v[ok, 0], v[ok, 1], v[ok, 2]), 1)
    return out


def coverage_report(cells: pd.DataFrame, structure_col: str = "structure_id", section_col: Optional[str] = None
                    ) -> pd.DataFrame:
    """Cells per structure and, if available, number of sections contributing (QC for sparse structures)."""
    g = cells.groupby(structure_col)
    rep = g.size().rename("n_cells").to_frame()
    if section_col and section_col in cells:
        rep["n_sections"] = g[section_col].nunique()
    return rep.sort_values("n_cells")
