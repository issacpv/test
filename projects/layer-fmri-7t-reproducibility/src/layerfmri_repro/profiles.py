"""Laminar profiles: extraction, normalization, shape features, similarity and vein deconvolution."""
from __future__ import annotations

import numpy as np
from scipy import signal


def laminar_profile(values: np.ndarray, bins: np.ndarray, n_bins: int, roi: np.ndarray | None = None,
                    agg=np.nanmean) -> np.ndarray:
    """Aggregate ``values`` (e.g. contrast estimates) within each depth bin 1..n_bins (deep → superficial)."""
    values, bins = np.asarray(values, float), np.asarray(bins, int)
    mask = np.ones(values.shape, bool) if roi is None else np.asarray(roi, bool)
    out = np.full(n_bins, np.nan)
    for b in range(1, n_bins + 1):
        m = mask & (bins == b)
        if m.any():
            out[b - 1] = agg(values[m])
    return out


def normalize_profile(p: np.ndarray, method: str = "zscore") -> np.ndarray:
    p = np.asarray(p, float)
    if method == "zscore":
        return (p - np.nanmean(p)) / (np.nanstd(p) + 1e-12)
    if method == "peak":
        return p / (np.nanmax(np.abs(p)) + 1e-12)
    if method == "mean":
        return p / (np.nanmean(p) + 1e-12)
    raise ValueError(method)


def superficial_bias_slope(p: np.ndarray) -> float:
    """Linear slope of the profile vs depth index (positive = increases toward the pial surface)."""
    p = np.asarray(p, float)
    x = np.arange(p.size)
    ok = np.isfinite(p)
    return float(np.polyfit(x[ok], p[ok], 1)[0]) if ok.sum() >= 2 else float("nan")


def detrend_depth(p: np.ndarray) -> np.ndarray:
    """Remove the linear depth trend (a crude superficial-bias correction)."""
    p = np.asarray(p, float)
    x = np.arange(p.size)
    ok = np.isfinite(p)
    a, b = np.polyfit(x[ok], p[ok], 1)
    return p - (a * x + b)


def detect_peaks(p: np.ndarray, prominence: float = 0.2) -> np.ndarray:
    """Indices of local maxima with the given prominence relative to the profile range."""
    p = np.asarray(p, float)
    rng_ = np.nanmax(p) - np.nanmin(p)
    pk, _ = signal.find_peaks(np.nan_to_num(p, nan=np.nanmin(p)), prominence=prominence * max(rng_, 1e-12))
    return pk


def is_double_peak(p: np.ndarray, prominence: float = 0.2, min_separation: int = 2) -> bool:
    """True if at least two peaks are separated by ≥ ``min_separation`` bins (M1 input/output signature)."""
    pk = detect_peaks(p, prominence)
    return bool(pk.size >= 2 and (np.diff(pk) >= min_separation).any())


def contrast_superficial_vs_deep(p: np.ndarray, frac: float = 1 / 3) -> float:
    """Mean of the superficial third minus the mean of the deep third."""
    p = np.asarray(p, float)
    k = max(1, int(round(frac * p.size)))
    return float(np.nanmean(p[-k:]) - np.nanmean(p[:k]))


def pearson(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.corrcoef(p1, p2)[0, 1])


def concordance_ccc(p1: np.ndarray, p2: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement, not just correlation)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    m1, m2 = p1.mean(), p2.mean()
    v1, v2 = p1.var(), p2.var()
    cov = np.mean((p1 - m1) * (p2 - m2))
    return float(2 * cov / (v1 + v2 + (m1 - m2) ** 2 + 1e-12))


def rmsd(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p1) - np.asarray(p2)) ** 2)))


# --------------------------------------------------------------------------------------
# Draining-vein (leakage) model: measured_k = true_k + λ · Σ_{j<k} w^{k-j-1} true_j
# (deep → superficial ordering). A leaky integrator of the deeper layers' signal, in the spirit
# of Markuerkiaga et al. (2016) and Havlicek & Uludağ (2020), with two scalar parameters.
# --------------------------------------------------------------------------------------
def leakage_matrix(n: int, leak: float, decay: float = 1.0) -> np.ndarray:
    L = np.eye(n)
    for k in range(n):
        for j in range(k):
            L[k, j] = leak * decay ** (k - j - 1)
    return L


def apply_draining(true_profile: np.ndarray, leak: float = 0.5, decay: float = 1.0) -> np.ndarray:
    """Forward model: add the venous leakage from deeper layers."""
    return leakage_matrix(len(true_profile), leak, decay) @ np.asarray(true_profile, float)


def devein(measured: np.ndarray, leak: float = 0.5, decay: float = 1.0) -> np.ndarray:
    """Inverse model: recover the layer-local profile from a GE-BOLD-like measured profile."""
    L = leakage_matrix(len(measured), leak, decay)
    return np.linalg.solve(L, np.asarray(measured, float))


def synthetic_profile(n_bins: int, kind: str = "double_peak", amplitude: float = 1.0) -> np.ndarray:
    """Canonical shapes: 'double_peak' (M1), 'deep' (feedback), 'middle' (feedforward), 'flat'."""
    x = np.linspace(0, 1, n_bins)
    if kind == "double_peak":
        p = np.exp(-((x - 0.3) / 0.12) ** 2) + 0.8 * np.exp(-((x - 0.75) / 0.12) ** 2)
    elif kind == "deep":
        p = np.exp(-((x - 0.2) / 0.18) ** 2)
    elif kind == "middle":
        p = np.exp(-((x - 0.5) / 0.15) ** 2)
    elif kind == "flat":
        p = np.ones(n_bins)
    else:
        raise ValueError(kind)
    return amplitude * p
