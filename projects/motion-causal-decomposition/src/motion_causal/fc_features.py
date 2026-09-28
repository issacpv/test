"""Motion summaries and functional-connectivity features.

Framewise displacement follows Power et al. (2012): the sum of absolute frame-to-frame
differences of the three translations (mm) and the three rotations converted to arc length on a
50 mm sphere. HCP ``Movement_Regressors.txt`` has 12 columns: x, y, z translations (mm), x, y, z
rotations (degrees), and their temporal derivatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def framewise_displacement(
    params: np.ndarray,
    radius_mm: float = 50.0,
    rotation_units: str = "deg",
    translations_first: bool = True,
) -> np.ndarray:
    """Power framewise displacement, length ``T`` (first frame is 0).

    Parameters
    ----------
    params
        ``(T, 6)`` rigid-body parameters (or ``(T, >=6)``, the first six columns are used).
    rotation_units
        ``"deg"`` or ``"rad"``.
    translations_first
        ``True`` for HCP / FSL mcflirt-derived files with translations in columns 0-2;
        ``False`` for SPM-style files with rotations first.
    """
    P = np.asarray(params, float)[:, :6]
    if not translations_first:
        P = P[:, [3, 4, 5, 0, 1, 2]]
    trans, rot = P[:, :3], P[:, 3:]
    if rotation_units == "deg":
        rot = np.deg2rad(rot)
    elif rotation_units != "rad":
        raise ValueError("rotation_units must be 'deg' or 'rad'")
    d = np.vstack([np.zeros((1, 6)), np.diff(np.column_stack([trans, rot * radius_mm]), axis=0)])
    return np.abs(d).sum(axis=1)


def load_hcp_movement_regressors(path: str | Path) -> np.ndarray:
    """Load an HCP ``Movement_Regressors.txt`` and return the ``(T, 6)`` rigid-body parameters."""
    M = np.loadtxt(path)
    if M.ndim != 2 or M.shape[1] < 6:
        raise ValueError(f"unexpected movement file shape {M.shape}")
    return M[:, :6]


def motion_summary(fd: np.ndarray, threshold: float = 0.2) -> Tuple[float, float]:
    """Return ``(mean FD, fraction of frames above threshold)``."""
    fd = np.asarray(fd, float)
    return float(fd.mean()), float((fd > threshold).mean())


def fisher_z_fc(ts: np.ndarray, censor: np.ndarray | None = None) -> np.ndarray:
    """Fisher-z Pearson correlation matrix of a ``(T, N)`` time series (diagonal set to 0).

    ``censor`` is an optional boolean mask of frames to *keep* (scrubbing).
    """
    X = np.asarray(ts, float)
    if censor is not None:
        X = X[np.asarray(censor, bool)]
    if X.shape[0] < 3:
        raise ValueError("need at least 3 frames")
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = np.inf
    R = (X / sd).T @ (X / sd) / X.shape[0]
    R = np.clip(R, -0.999999, 0.999999)
    Z = np.arctanh(R)
    np.fill_diagonal(Z, 0.0)
    return Z


def vectorize_upper(fc: np.ndarray) -> np.ndarray:
    """Upper triangle (k=1) of a square matrix as a 1-D vector."""
    fc = np.asarray(fc)
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]


def edge_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance between node centroids for each upper-triangle edge (mm)."""
    C = np.asarray(coords, float)
    D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    return vectorize_upper(D)
