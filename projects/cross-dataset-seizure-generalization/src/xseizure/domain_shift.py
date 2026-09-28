"""Domain-shift diagnostics between EEG cohorts.

Three complementary distances are provided:

* ``mmd_rbf`` / ``mmd_permutation_test`` : kernel two-sample statistic on any
  feature representation (spectral features, tangent-space vectors, FM embeddings);
* ``psd_divergence`` : per-channel log-spectral distance and Jensen-Shannon
  divergence between cohort-mean normalized PSDs (captures line noise, filtering
  and sampling-rate artefacts after harmonization);
* ``covariance_shift`` : affine-invariant Riemannian distance between cohort
  covariance means (captures montage/re-referencing effects).

The intended use is to correlate these quantities, computed on *background*
(non-seizure) windows, with the observed cross-cohort loss in event F1.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import cdist

from .features import covariances, riemannian_distance, riemannian_mean


# --------------------------------------------------------------------------- #
# MMD
# --------------------------------------------------------------------------- #
def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """RBF bandwidth ``gamma = 1 / (2 * median_pairwise_distance^2)`` on a subsample."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    if len(Z) > max_n:
        Z = Z[rng.choice(len(Z), max_n, replace=False)]
    d = cdist(Z, Z, "euclidean")
    med = np.median(d[np.triu_indices(len(Z), 1)])
    return 1.0 / (2.0 * max(med, 1e-12) ** 2)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None, biased: bool = False) -> float:
    """Squared maximum mean discrepancy with an RBF kernel (unbiased U-statistic by default)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    m, n = len(X), len(Y)
    if biased:
        return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
    sx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sx + sy - 2 * Kxy.mean())


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 500, gamma: Optional[float] = None,
                         seed: int = 0) -> Dict[str, float]:
    """Permutation p-value for MMD^2 (labels shuffled between X and Y)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    gamma = median_heuristic_gamma(X, Y) if gamma is None else gamma
    obs = mmd_rbf(X, Y, gamma)
    Z = np.vstack([X, Y])
    m = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Z))
        null[i] = mmd_rbf(Z[perm[:m]], Z[perm[m:]], gamma)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return {"mmd2": obs, "p_value": float(p), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "gamma": float(gamma)}


# --------------------------------------------------------------------------- #
# PSD divergence
# --------------------------------------------------------------------------- #
def cohort_psd(windows: np.ndarray, fs: float, fmax: float = 70.0,
               nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean PSD per channel over windows: returns ``(freqs, psd)`` with psd shape (n_ch, n_freq)."""
    w = np.asarray(windows, float)
    nperseg = nperseg or min(w.shape[-1], int(2 * fs))
    f, pxx = welch(w, fs=fs, nperseg=nperseg, axis=-1)
    sel = f <= fmax
    return f[sel], pxx[..., sel].mean(axis=0)


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-20, None)
    return p / p.sum(axis=-1, keepdims=True)


def psd_divergence(psd_a: np.ndarray, psd_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-channel log-spectral distance (dB RMS) and Jensen-Shannon divergence (bits)."""
    a = np.atleast_2d(psd_a); b = np.atleast_2d(psd_b)
    if a.shape != b.shape:
        raise ValueError("PSDs must be computed on the same frequency grid and channels")
    lsd = np.sqrt(np.mean((10 * np.log10(np.clip(a, 1e-20, None)) - 10 * np.log10(np.clip(b, 1e-20, None))) ** 2, axis=-1))
    pa, pb = _normalize(a), _normalize(b)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log2(p / q), axis=-1)
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return {"log_spectral_distance_db": lsd, "js_divergence_bits": js}


# --------------------------------------------------------------------------- #
# covariance / montage shift
# --------------------------------------------------------------------------- #
def covariance_shift(windows_a: np.ndarray, windows_b: np.ndarray, shrinkage: float = 0.05) -> Dict[str, float]:
    """Riemannian distance between cohort covariance means, and the within-cohort spread for scale."""
    ca = covariances(windows_a, shrinkage); cb = covariances(windows_b, shrinkage)
    ma, mb = riemannian_mean(ca), riemannian_mean(cb)
    within_a = np.mean([riemannian_distance(ma, c) for c in ca])
    within_b = np.mean([riemannian_distance(mb, c) for c in cb])
    between = riemannian_distance(ma, mb)
    return {"between": between, "within_a": float(within_a), "within_b": float(within_b),
            "ratio": between / max(0.5 * (within_a + within_b), 1e-12)}


def shift_report(feat_a: np.ndarray, feat_b: np.ndarray, windows_a: Optional[np.ndarray] = None,
                 windows_b: Optional[np.ndarray] = None, fs: Optional[float] = None,
                 n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    """One-call summary of feature-space MMD, PSD divergence and covariance shift."""
    out: Dict[str, float] = {}
    mmd = mmd_permutation_test(feat_a, feat_b, n_perm=n_perm, seed=seed)
    out.update({f"mmd_{k}": v for k, v in mmd.items()})
    if windows_a is not None and windows_b is not None and fs is not None:
        fa, pa = cohort_psd(windows_a, fs)
        _, pb = cohort_psd(windows_b, fs)
        div = psd_divergence(pa, pb)
        out["psd_lsd_db_mean"] = float(np.mean(div["log_spectral_distance_db"]))
        out["psd_js_bits_mean"] = float(np.mean(div["js_divergence_bits"]))
        cov = covariance_shift(windows_a, windows_b)
        out.update({f"cov_{k}": v for k, v in cov.items()})
    return out
