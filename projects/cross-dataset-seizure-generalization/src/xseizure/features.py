"""Spectral and Riemannian baseline features and a logistic-regression baseline.

The Riemannian part (covariance -> tangent space at a reference mean) is
implemented in numpy/scipy so it runs without pyriemann; if pyriemann is
installed the same API can delegate to it via ``use_pyriemann=True``.
Riemannian *re-centering* (Zanini et al., 2018) is included as the simplest
unsupervised domain-adaptation baseline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 70.0),
}


# --------------------------------------------------------------------------- #
# windowing and labels
# --------------------------------------------------------------------------- #
def windows_from_array(data: np.ndarray, fs: float, win_s: float, step_s: float) -> np.ndarray:
    """Cut ``(n_ch, n_samples)`` into ``(n_windows, n_ch, win_samples)``; NaN rows -> 0."""
    w = int(round(win_s * fs))
    s = int(round(step_s * fs))
    n = data.shape[1]
    starts = range(0, n - w + 1, s)
    out = np.stack([np.nan_to_num(data[:, st:st + w]) for st in starts]) if n >= w else np.empty((0, data.shape[0], w))
    return out


def window_labels(n_windows: int, win_s: float, step_s: float,
                  events: Sequence[Tuple[float, float]], min_overlap: float = 0.5) -> np.ndarray:
    """Binary label per window: 1 if >= ``min_overlap`` of the window lies inside any event."""
    y = np.zeros(n_windows, dtype=int)
    starts = np.arange(n_windows) * step_s
    for on, off in events:
        ov = np.clip(np.minimum(starts + win_s, off) - np.maximum(starts, on), 0, None)
        y[ov / win_s >= min_overlap] = 1
    return y


# --------------------------------------------------------------------------- #
# spectral / time-domain features
# --------------------------------------------------------------------------- #
def spectral_feature_names(n_ch: int, bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS) -> List[str]:
    names = []
    for c in range(n_ch):
        names += [f"ch{c}_log_{b}" for b in bands]
        names += [f"ch{c}_rel_{b}" for b in bands]
        names += [f"ch{c}_linelength", f"ch{c}_hjorth_mobility", f"ch{c}_hjorth_complexity"]
    return names


def spectral_features(windows: np.ndarray, fs: float,
                      bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
                      nperseg: Optional[int] = None) -> np.ndarray:
    """Per-channel log band power, relative band power, line length and Hjorth parameters.

    Parameters
    ----------
    windows : (n_windows, n_ch, n_samples)
    fs : sampling rate in Hz

    Returns
    -------
    X : (n_windows, n_ch * (2 * n_bands + 3))
    """
    windows = np.asarray(windows, float)
    n_win, n_ch, n_samp = windows.shape
    nperseg = nperseg or min(n_samp, int(2 * fs))
    f, pxx = welch(windows, fs=fs, nperseg=nperseg, axis=-1)  # (n_win, n_ch, n_freq)
    feats = []
    total = np.trapezoid(pxx, f, axis=-1) + 1e-12
    for c in range(n_ch):
        x = windows[:, c, :]
        bp = []
        for lo, hi in bands.values():
            sel = (f >= lo) & (f < hi)
            bp.append(np.trapezoid(pxx[:, c, sel], f[sel], axis=-1) if sel.any() else np.zeros(n_win))
        bp = np.stack(bp, axis=1)
        feats.append(np.log(bp + 1e-12))
        feats.append(bp / total[:, c][:, None])
        feats.append(np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)[:, None])
        d1 = np.diff(x, axis=-1)
        d2 = np.diff(d1, axis=-1)
        v0 = x.var(axis=-1) + 1e-12
        v1 = d1.var(axis=-1) + 1e-12
        v2 = d2.var(axis=-1) + 1e-12
        mob = np.sqrt(v1 / v0)
        comp = np.sqrt(v2 / v1) / mob
        feats.append(mob[:, None])
        feats.append(comp[:, None])
    return np.concatenate(feats, axis=1)


# --------------------------------------------------------------------------- #
# Riemannian geometry (numpy implementation)
# --------------------------------------------------------------------------- #
def covariances(windows: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    """Shrinkage covariance per window: ``(1-a) * S + a * tr(S)/n * I``."""
    w = np.asarray(windows, float)
    w = w - w.mean(axis=-1, keepdims=True)
    n_samp = w.shape[-1]
    covs = np.einsum("wcs,wds->wcd", w, w) / max(n_samp - 1, 1)
    n_ch = covs.shape[-1]
    tr = np.trace(covs, axis1=1, axis2=2) / n_ch
    eye = np.eye(n_ch)
    return (1 - shrinkage) * covs + shrinkage * tr[:, None, None] * eye


def _fun_spd(C: np.ndarray, fun) -> np.ndarray:
    vals, vecs = eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * fun(vals)) @ vecs.T


def logm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, np.log)


def expm_sym(S: np.ndarray) -> np.ndarray:
    return _fun_spd(S, np.exp)


def invsqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, lambda v: 1.0 / np.sqrt(v))


def sqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, np.sqrt)


def log_euclidean_mean(covs: np.ndarray) -> np.ndarray:
    """Closed-form log-Euclidean mean of SPD matrices."""
    return expm_sym(np.mean([logm_spd(c) for c in covs], axis=0))


def riemannian_mean(covs: np.ndarray, n_iter: int = 20, tol: float = 1e-8) -> np.ndarray:
    """Affine-invariant (Karcher) mean by gradient descent, initialised at the log-Euclidean mean."""
    M = log_euclidean_mean(covs)
    for _ in range(n_iter):
        Mi = invsqrtm_spd(M)
        Ms = sqrtm_spd(M)
        T = np.mean([logm_spd(Mi @ c @ Mi) for c in covs], axis=0)
        M = Ms @ expm_sym(T) @ Ms
        if np.linalg.norm(T) < tol:
            break
    return M


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map SPD matrices to the tangent space at ``ref`` (affine-invariant metric).

    Returns the vectorized upper triangle with off-diagonal elements scaled by
    sqrt(2) so that Euclidean distances equal the Frobenius norm.
    """
    Ri = invsqrtm_spd(ref)
    n = ref.shape[0]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((len(covs), len(iu[0])))
    for i, c in enumerate(covs):
        S = logm_spd(Ri @ c @ Ri)
        out[i] = S[iu] * coef
    return out


def riemannian_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant distance ``||log(A^-1/2 B A^-1/2)||_F``."""
    Ai = invsqrtm_spd(A)
    S = logm_spd(Ai @ B @ Ai)
    return float(np.linalg.norm(S, "fro"))


def recenter(covs: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Riemannian re-centering (Zanini et al., 2018): whiten each covariance by ``ref``.

    With ``ref`` = the mean of the *target-domain background windows*, source and
    target covariances are both mapped so that their means become the identity;
    this is the standard label-free alignment used for cross-subject/cross-site BCI.
    """
    ref = riemannian_mean(covs) if ref is None else ref
    Ri = invsqrtm_spd(ref)
    return np.einsum("ij,wjk,kl->wil", Ri, covs, Ri)


def riemannian_features(windows: np.ndarray, ref: Optional[np.ndarray] = None,
                        shrinkage: float = 0.05, use_pyriemann: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Covariance -> tangent-space features. Returns ``(X, ref)`` so ``ref`` can be reused on targets."""
    covs = covariances(windows, shrinkage)
    if use_pyriemann:
        from pyriemann.tangentspace import TangentSpace  # lazy

        ts = TangentSpace(metric="riemann")
        if ref is None:
            X = ts.fit_transform(covs)
            return X, ts.reference_
        ts.reference_ = ref
        return ts.transform(covs), ref
    ref = riemannian_mean(covs) if ref is None else ref
    return tangent_space(covs, ref), ref


# --------------------------------------------------------------------------- #
# baseline model
# --------------------------------------------------------------------------- #
class LogisticBaseline:
    """Standardized, class-balanced logistic regression on precomputed features."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def smooth_probabilities(p: np.ndarray, k: int = 5) -> np.ndarray:
    """Centered moving average over ``k`` consecutive windows (edge-padded)."""
    if k <= 1:
        return np.asarray(p, float)
    pad = k // 2
    pp = np.pad(np.asarray(p, float), pad, mode="edge")
    return np.convolve(pp, np.ones(k) / k, mode="valid")
