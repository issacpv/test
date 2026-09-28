"""Loaders for parcellated HCP cortical maps and parcel geometry.

Supported inputs
----------------
* CIFTI ``.pscalar.nii`` (already parcellated, e.g. HCP S1200 group-average
  ``*.MyelinMap_BC_MSMAll.32k_fs_LR.pscalar.nii``) via nibabel;
* CIFTI ``.dscalar.nii`` + ``.dlabel.nii`` parcellation, averaged per label;
* plain CSV/TSV with ``region,value`` columns (what most published maps are
  shared as, and what ``neuromaps.parcellate`` returns);
* ``neuromaps`` annotations (e.g. ``('hcps1200', 'myelinmap', 'fsLR', '32k')``)
  when the package is installed.

Geometry helpers give parcel centroids on the sphere (needed for spin tests)
from GIFTI sphere surfaces + label files, and a synthetic Fibonacci sphere
for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def parcellate_vector(values: np.ndarray, labels: np.ndarray, label_names: Optional[Dict[int, str]] = None,
                      ignore: Sequence[int] = (0,)) -> pd.Series:
    """Mean of ``values`` within each integer label (vertex- or voxel-wise).

    Parameters
    ----------
    values : (n_vertices,) float
    labels : (n_vertices,) int
    label_names : optional ``{label_id: name}``
    ignore : label ids to skip (0 = medial wall / unassigned)
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    ids = [i for i in np.unique(labels) if i not in ignore]
    out = {}
    for i in ids:
        m = labels == i
        name = label_names.get(int(i), int(i)) if label_names else int(i)
        out[name] = float(np.nanmean(values[m])) if m.any() else np.nan
    return pd.Series(out, name="value")


def load_parcellated_csv(path: PathLike, region_col: str = "region", value_col: str = "value") -> pd.Series:
    """Read a ``region,value`` table into a Series indexed by region name."""
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    return pd.Series(df[value_col].to_numpy(dtype=float), index=df[region_col].astype(str).to_numpy(), name=value_col)


def load_pscalar(path: PathLike) -> pd.Series:
    """Parcellated CIFTI scalar -> Series indexed by parcel name (needs nibabel)."""
    try:
        import nibabel as nib  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install nibabel") from e
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata()).squeeze()
    axis = img.header.get_axis(1)
    names = list(getattr(axis, "name", [str(i) for i in range(data.shape[-1])]))
    return pd.Series(data if data.ndim == 1 else data[0], index=names, name=Path(path).stem)


def load_dscalar_with_dlabel(dscalar: PathLike, dlabel: PathLike, column: int = 0) -> pd.Series:
    """Average a dense CIFTI scalar within a CIFTI label parcellation."""
    try:
        import nibabel as nib  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install nibabel") from e
    d = nib.load(str(dscalar))
    lab = nib.load(str(dlabel))
    values = np.asarray(d.get_fdata())[column]
    labels = np.asarray(lab.get_fdata())[0].astype(int)
    if values.shape[0] != labels.shape[0]:
        raise ValueError("dscalar and dlabel have different numbers of grayordinates")
    table = lab.header.get_axis(0).label[0]  # {id: (name, rgba)}
    names = {k: v[0] for k, v in table.items()}
    return parcellate_vector(values, labels, names)


def fetch_neuromaps_annotation(source: str, desc: str, space: str = "fsLR", den: str = "32k",
                               parcellation: Optional[Tuple[PathLike, PathLike]] = None,
                               data_dir: Optional[PathLike] = None) -> pd.Series:
    """Fetch and (optionally) parcellate a neuromaps annotation.

    Example: ``fetch_neuromaps_annotation('hcps1200', 'myelinmap')`` for the
    HCP S1200 T1w/T2w myelin map (Glasser & Van Essen, 2011).  ``parcellation``
    is a (lh, rh) pair of GIFTI label files in the same space.
    """
    try:
        from neuromaps import datasets, parcellate  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install neuromaps") from e
    ann = datasets.fetch_annotation(source=source, desc=desc, space=space, den=den, data_dir=data_dir)
    if parcellation is None:
        import nibabel as nib  # type: ignore
        vals = np.concatenate([nib.load(str(f)).agg_data() for f in ann])
        return pd.Series(vals, name=f"{source}-{desc}")
    parc = parcellate.Parcellater(parcellation, space)
    out = parc.fit_transform(ann, space)
    return pd.Series(np.asarray(out).squeeze(), name=f"{source}-{desc}")


def parcel_centroids_from_gifti(sphere_gii: PathLike, label_gii: PathLike, ignore: Sequence[int] = (0,)) -> pd.DataFrame:
    """Spherical parcel centroids (normalised to unit sphere) from GIFTI files.

    Returns ``x, y, z`` per label id, ordered by label id.  Use the *sphere*
    surface (e.g. ``S1200.L.sphere.32k_fs_LR.surf.gii``), not pial/midthickness,
    for spin tests.
    """
    try:
        import nibabel as nib  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install nibabel") from e
    coords = nib.load(str(sphere_gii)).agg_data("NIFTI_INTENT_POINTSET")
    labels = np.asarray(nib.load(str(label_gii)).agg_data()).astype(int)
    rows = {}
    for i in np.unique(labels):
        if i in ignore:
            continue
        c = coords[labels == i].mean(axis=0)
        rows[int(i)] = c / np.linalg.norm(c)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["x", "y", "z"])


def fibonacci_sphere(n: int, hemisphere: Optional[str] = None, seed: int = 0) -> np.ndarray:
    """Quasi-uniform points on the unit sphere (for tests and simulations).

    ``hemisphere='L'`` keeps x < 0, ``'R'`` keeps x > 0 (mirror-symmetric
    pairs are produced by :func:`split_hemispheres`).
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    if hemisphere == "L":
        pts = pts[pts[:, 0] < 0]
    elif hemisphere == "R":
        pts = pts[pts[:, 0] > 0]
    return pts


def split_hemispheres(n_per_hemi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror-symmetric synthetic parcel centroids for left/right hemispheres."""
    lh = fibonacci_sphere(2 * n_per_hemi, hemisphere="L")[:n_per_hemi]
    rh = lh * np.array([-1.0, 1.0, 1.0])
    return lh, rh


def geodesic_distance_matrix(coords: np.ndarray, radius: float = 100.0) -> np.ndarray:
    """Great-circle distances between unit-sphere points scaled by ``radius`` (mm).

    A cheap stand-in for surface geodesic distances (Burt et al., 2020 use
    exact geodesics computed with ``wb_command``); use
    ``brainsmash.workbench.geo`` for the real thing.
    """
    c = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    cosang = np.clip(c @ c.T, -1.0, 1.0)
    return radius * np.arccos(cosang)


def align_map_to_expression(map_values: pd.Series, expr: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Inner-join a parcellated map and an expression table on region labels,
    dropping regions that are NaN in either (e.g. right-hemisphere parcels
    when only left-hemisphere AHBA data are used)."""
    common = map_values.index.intersection(expr.index)
    m = map_values.loc[common]
    e = expr.loc[common]
    ok = m.notna() & e.notna().all(axis=1)
    return m[ok], e[ok]
