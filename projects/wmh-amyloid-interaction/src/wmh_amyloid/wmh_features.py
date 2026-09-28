"""WMH features from masks (numpy arrays; use nibabel to load NIfTI in production).

Periventricular vs deep WMH are separated by distance to the lateral-ventricle mask
(distance-transform rule; 10 mm is the common convention, 3 mm a stricter "juxtaventricular"
variant). Lobar volumes use any integer lobe label map in the same space.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


def wmh_volume_ml(mask: np.ndarray, voxel_size_mm: Sequence[float]) -> float:
    """Volume of a binary/probabilistic mask in millilitres."""
    vox = float(np.prod(voxel_size_mm))
    return float(np.asarray(mask, float).sum() * vox / 1000.0)


def periventricular_deep_split(
    wmh_mask: np.ndarray,
    ventricle_mask: np.ndarray,
    voxel_size_mm: Sequence[float],
    pv_distance_mm: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Split WMH into periventricular (within ``pv_distance_mm`` of the ventricles) and deep.

    Returns ``(pv_mask, deep_mask, volumes)`` with volumes in ml.
    """
    wmh = np.asarray(wmh_mask) > 0.5
    vent = np.asarray(ventricle_mask) > 0.5
    if not vent.any():
        raise ValueError("empty ventricle mask")
    dist = ndimage.distance_transform_edt(~vent, sampling=voxel_size_mm)
    pv = wmh & (dist <= pv_distance_mm)
    deep = wmh & ~pv
    vols = {
        "total_ml": wmh_volume_ml(wmh, voxel_size_mm),
        "periventricular_ml": wmh_volume_ml(pv, voxel_size_mm),
        "deep_ml": wmh_volume_ml(deep, voxel_size_mm),
    }
    return pv, deep, vols


def lobar_volumes(
    wmh_mask: np.ndarray,
    lobe_labels: np.ndarray,
    voxel_size_mm: Sequence[float],
    label_names: Optional[Dict[int, str]] = None,
    posterior_labels: Sequence[int] = (),
) -> Dict[str, float]:
    """WMH volume (ml) per lobe label and the posterior fraction (e.g. parietal + occipital)."""
    wmh = np.asarray(wmh_mask) > 0.5
    lobes = np.asarray(lobe_labels, int)
    out: Dict[str, float] = {}
    total = wmh_volume_ml(wmh, voxel_size_mm)
    for lab in np.unique(lobes):
        if lab == 0:
            continue
        name = (label_names or {}).get(int(lab), f"lobe{int(lab)}")
        out[f"{name}_ml"] = wmh_volume_ml(wmh & (lobes == lab), voxel_size_mm)
    post = wmh_volume_ml(wmh & np.isin(lobes, list(posterior_labels)), voxel_size_mm) if posterior_labels else np.nan
    out["posterior_fraction"] = post / total if total > 0 else np.nan
    return out


def normalize_wmh(volume_ml: np.ndarray, icv_ml: Optional[np.ndarray] = None, log: bool = True) -> np.ndarray:
    """log(1 + WMH) or WMH as % of ICV (log-transformed if ``log``)."""
    v = np.asarray(volume_ml, float)
    if icv_ml is not None:
        v = 100.0 * v / np.asarray(icv_ml, float)
    return np.log1p(v) if log else v


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    denom = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / denom) if denom > 0 else np.nan


def volume_agreement(v_a: np.ndarray, v_b: np.ndarray) -> Dict[str, float]:
    """ICC(2,1), Bland-Altman bias/limits and Pearson r for two tools' volumes across sessions."""
    a, b = np.asarray(v_a, float), np.asarray(v_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    Y = np.column_stack([a, b])
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    d = b - a
    return {
        "icc_2_1": float(icc),
        "bias": float(d.mean()),
        "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)),
        "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "n": int(n),
    }
