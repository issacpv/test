"""Segmentation metrics and paired statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, stats


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def _surface(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    eroded = ndimage.binary_erosion(m)
    return m & ~eroded


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Distances from surface voxels of a to surface of b and vice versa (mm)."""
    sa, sb = _surface(a), _surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return np.array([np.inf]), np.array([np.inf])
    dt_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    return dt_b[sa], dt_a[sb]


def hd95(a: np.ndarray, b: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    d_ab, d_ba = surface_distances(a, b, spacing)
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def assd(a: np.ndarray, b: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    d_ab, d_ba = surface_distances(a, b, spacing)
    return float((d_ab.sum() + d_ba.sum()) / (len(d_ab) + len(d_ba)))


def volume_error(pred: np.ndarray, ref: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict[str, float]:
    vox = float(np.prod(spacing))
    vp, vr = pred.astype(bool).sum() * vox, ref.astype(bool).sum() * vox
    return {"vol_pred": vp, "vol_ref": vr, "abs_vol_err": abs(vp - vr), "rel_vol_err": (vp - vr) / vr if vr > 0 else np.nan}


def per_structure(pred: np.ndarray, ref: np.ndarray, labels: dict[str, int] | None = None,
                  spacing=(1.0, 1.0, 1.0), distances: bool = True) -> pd.DataFrame:
    """Metrics for every label present in ``ref`` (or the given label map)."""
    labels = labels or {f"label_{v}": int(v) for v in np.unique(ref) if v != 0}
    rows = []
    for name, v in labels.items():
        p, r = pred == v, ref == v
        row = {"structure": name, "dice": dice(p, r), **volume_error(p, r, spacing)}
        if distances:
            row["hd95"] = hd95(p, r, spacing)
            row["assd"] = assd(p, r, spacing)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired difference y − x with Wilcoxon signed-rank p and bootstrap CI over subjects."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    return {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p_wilcoxon": p, "n": int(len(d))}


def tost_noninferiority(x: np.ndarray, y: np.ndarray, margin: float) -> dict[str, float]:
    """One-sided paired t-test that mean(y − x) > −margin (y non-inferior to x)."""
    d = np.asarray(y, float) - np.asarray(x, float)
    t, p_two = stats.ttest_1samp(d, -margin)
    p = p_two / 2 if t > 0 else 1 - p_two / 2
    return {"mean_diff": float(d.mean()), "margin": margin, "t": float(t), "p_noninferior": float(p)}


def silver_ceiling(silver: list[np.ndarray], manual: list[np.ndarray], labels: dict[str, int]) -> pd.DataFrame:
    """Agreement between silver (e.g., FreeSurfer) and manual labels per structure: the ceiling for silver-label validation."""
    frames = [per_structure(s, m, labels, distances=False).assign(case=i) for i, (s, m) in enumerate(zip(silver, manual))]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("structure")["dice"].agg(["mean", "std", "min", "max"]).reset_index()


def within_subject_agreement(seg_clean: np.ndarray, seg_degraded: np.ndarray, labels: dict[str, int]) -> pd.DataFrame:
    """MR-ART-style paired agreement: Dice and relative volume change of the degraded vs clean segmentation of the same subject."""
    out = per_structure(seg_degraded, seg_clean, labels, distances=False)
    return out.rename(columns={"dice": "dice_vs_clean", "rel_vol_err": "rel_vol_change"})[["structure", "dice_vs_clean", "rel_vol_change"]]
