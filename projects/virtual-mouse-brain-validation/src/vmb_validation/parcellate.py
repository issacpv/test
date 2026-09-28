"""Region time series from 4-D fMRI volumes or widefield frame stacks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def parcellate_array(data: np.ndarray, labels: np.ndarray, label_ids: Sequence[int],
                     min_voxels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Mean time series per label. ``data`` has shape (..., T) with spatial dims matching ``labels``.

    Returns (T, n_labels) array and a boolean mask of labels with >= ``min_voxels`` voxels.
    """
    data = np.asarray(data, float)
    labels = np.asarray(labels)
    if data.shape[:-1] != labels.shape:
        raise ValueError(f"spatial shape mismatch {data.shape[:-1]} vs {labels.shape}")
    flat = data.reshape(-1, data.shape[-1])
    lab = labels.reshape(-1)
    out = np.full((data.shape[-1], len(label_ids)), np.nan)
    ok = np.zeros(len(label_ids), dtype=bool)
    for k, lid in enumerate(label_ids):
        m = lab == lid
        if m.sum() >= min_voxels:
            out[:, k] = flat[m].mean(axis=0)
            ok[k] = True
    return out, ok


def parcellate_volume(nifti_path: Path | str, atlas_path: Path | str, label_ids: Sequence[int],
                      **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Parcellate a 4-D NIfTI with a 3-D label volume on the same grid (nibabel)."""
    import nibabel as nib

    img = nib.load(str(nifti_path))
    atlas = nib.load(str(atlas_path))
    return parcellate_array(np.asarray(img.dataobj), np.asarray(atlas.dataobj).astype(int), label_ids, **kw)


def parcellate_frames(frames: np.ndarray, label_image: np.ndarray, label_ids: Sequence[int],
                      **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Parcellate widefield frames of shape (T, H, W) with a 2-D label image (H, W)."""
    return parcellate_array(np.moveaxis(np.asarray(frames), 0, -1), np.asarray(label_image), label_ids, **kw)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass of (T, N) time series (NaN columns left as NaN)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    out = np.full_like(x, np.nan)
    good = np.isfinite(x).all(axis=0)
    out[:, good] = filtfilt(b, a, x[:, good], axis=0)
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def hemisphere_mirror_labels(label_ids: Sequence[int], left_ids: Sequence[int], right_ids: Sequence[int]
                             ) -> np.ndarray:
    """Index pairs (left, right) in ``label_ids`` order for homotopic FC."""
    pos = {lid: k for k, lid in enumerate(label_ids)}
    return np.array([[pos[l], pos[r]] for l, r in zip(left_ids, right_ids) if l in pos and r in pos])
