"""SPD-matrix geometry for EEG covariance features, alignment methods and an MDM classifier.

Everything here is plain numpy/scipy so the transfer arms can be unit-tested without
``pyriemann``; production runs should cross-check against ``pyriemann`` (which also provides
Riemannian Procrustes Analysis and tangent-space + logistic regression pipelines).

Notation: trials ``X`` have shape ``(n_trials, n_channels, n_times)``; covariances ``C`` have
shape ``(n_trials, n_channels, n_channels)``.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import linalg

EPS = 1e-10


# --------------------------------------------------------------------------- covariance
def covariances(X: np.ndarray, shrinkage: Union[str, float] = "lw") -> np.ndarray:
    """Per-trial channel covariance with Ledoit-Wolf (``"lw"``) or fixed shrinkage.

    Fixed shrinkage ``a`` in [0, 1] returns ``(1-a) S + a tr(S)/C I``; ``0`` is the sample covariance.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must be (n_trials, n_channels, n_times)")
    n, c, t = X.shape
    Xc = X - X.mean(axis=2, keepdims=True)
    S = np.einsum("nct,ndt->ncd", Xc, Xc) / max(t - 1, 1)
    if shrinkage == "lw":
        from sklearn.covariance import ledoit_wolf

        out = np.empty_like(S)
        for i in range(n):
            out[i] = ledoit_wolf(Xc[i].T, assume_centered=True)[0]
        return out
    a = float(shrinkage)
    if not 0 <= a <= 1:
        raise ValueError("shrinkage must be 'lw' or a float in [0, 1]")
    mu = np.trace(S, axis1=1, axis2=2) / c
    return (1 - a) * S + a * mu[:, None, None] * np.eye(c)[None]


# --------------------------------------------------------------------------- SPD functions
def _fun_spd(C: np.ndarray, fun: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    w, V = linalg.eigh(C)
    w = np.maximum(w, EPS)
    return (V * fun(w)) @ V.T


def sqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, np.sqrt)


def invsqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, lambda w: 1.0 / np.sqrt(w))


def logm_spd(C: np.ndarray) -> np.ndarray:
    return _fun_spd(C, np.log)


def expm_spd(S: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix (returns SPD)."""
    w, V = linalg.eigh(S)
    return (V * np.exp(w)) @ V.T


def airm_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant Riemannian distance ``||log(A^-1/2 B A^-1/2)||_F``."""
    w = linalg.eigh(B, A, eigvals_only=True)
    return float(np.sqrt(np.sum(np.log(np.maximum(w, EPS)) ** 2)))


def geodesic(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Point at fraction ``t`` on the AIRM geodesic from A to B."""
    As, iAs = sqrtm_spd(A), invsqrtm_spd(A)
    return As @ _fun_spd(iAs @ B @ iAs, lambda w: w ** t) @ As


# --------------------------------------------------------------------------- means
def riemannian_mean(C: np.ndarray, tol: float = 1e-8, max_iter: int = 100, init: Optional[np.ndarray] = None,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Karcher (Frechet) mean under the affine-invariant metric, by tangent-space iteration."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    M = np.tensordot(w, C, axes=1) if init is None else np.asarray(init, dtype=float)
    for _ in range(max_iter):
        Ms, iMs = sqrtm_spd(M), invsqrtm_spd(M)
        T = sum(w[i] * logm_spd(iMs @ C[i] @ iMs) for i in range(n))
        M = Ms @ expm_spd(T) @ Ms
        if np.linalg.norm(T, "fro") < tol:
            break
    return M


def log_euclidean_mean(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    return expm_spd(np.mean([logm_spd(c) for c in C], axis=0))


# --------------------------------------------------------------------------- tangent space
def tangent_space(C: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Vectorised tangent vectors at ``ref``: upper triangle with off-diagonals scaled by sqrt(2).

    With this scaling the Euclidean norm of a tangent vector equals the AIRM distance to ``ref``.
    """
    C = np.asarray(C, dtype=float)
    iRs = invsqrtm_spd(ref)
    c = ref.shape[0]
    iu = np.triu_indices(c)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((C.shape[0], len(iu[0])))
    for i in range(C.shape[0]):
        S = logm_spd(iRs @ C[i] @ iRs)
        out[i] = S[iu] * coef
    return out


# --------------------------------------------------------------------------- alignment
def euclidean_alignment(X: np.ndarray, ref: Optional[np.ndarray] = None, shrinkage: Union[str, float] = 0.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean alignment (He & Wu, 2020): whiten trials by the arithmetic mean covariance.

    Returns ``(X_aligned, R)`` with ``X_aligned[i] = R^{-1/2} X[i]``. Pass ``ref`` to align with a
    reference estimated elsewhere (e.g. from k calibration trials only).
    """
    X = np.asarray(X, dtype=float)
    R = covariances(X, shrinkage).mean(axis=0) if ref is None else np.asarray(ref, dtype=float)
    iRs = invsqrtm_spd(R)
    return np.einsum("cd,ndt->nct", iRs, X), R


def riemannian_alignment(C: np.ndarray, ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Riemannian re-centering (Zanini et al., 2018): ``M^{-1/2} C_i M^{-1/2}`` with M the Riemannian mean.

    Returns ``(C_aligned, M)``. With ``ref`` given (e.g. from calibration trials), uses it instead.
    """
    C = np.asarray(C, dtype=float)
    M = riemannian_mean(C) if ref is None else np.asarray(ref, dtype=float)
    iMs = invsqrtm_spd(M)
    return np.einsum("cd,nde,ef->ncf", iMs, C, iMs), M


def subject_shift(C_subject: np.ndarray, C_pool_mean: np.ndarray) -> float:
    """AIRM distance between a subject's Riemannian mean and the source-pool mean (H4 predictor)."""
    return airm_distance(riemannian_mean(C_subject), C_pool_mean)


# --------------------------------------------------------------------------- classifier
class MDM:
    """Minimum Distance to (Riemannian) Mean classifier (Barachant et al., 2012)."""

    def __init__(self, metric: str = "airm") -> None:
        if metric not in ("airm", "logeuclid"):
            raise ValueError("metric must be 'airm' or 'logeuclid'")
        self.metric = metric
        self.classes_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None

    def fit(self, C: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "MDM":
        C = np.asarray(C, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        means = []
        for k in self.classes_:
            Ck = C[y == k]
            wk = None if sample_weight is None else np.asarray(sample_weight)[y == k]
            means.append(riemannian_mean(Ck, weights=wk) if self.metric == "airm" else log_euclidean_mean(Ck))
        self.means_ = np.stack(means)
        return self

    def distances(self, C: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("fit first")
        C = np.asarray(C, dtype=float)
        if self.metric == "airm":
            return np.array([[airm_distance(m, c) for m in self.means_] for c in C])
        logs = [logm_spd(m) for m in self.means_]
        return np.array([[np.linalg.norm(logm_spd(c) - lm, "fro") for lm in logs] for c in C])

    def predict(self, C: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmin(self.distances(C), axis=1)]

    def predict_proba(self, C: np.ndarray) -> np.ndarray:
        d = self.distances(C) ** 2
        z = -d + d.min(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = [np.mean(y_pred[y_true == k] == k) for k in np.unique(y_true)]
    return float(np.mean(accs))


__all__ = ["covariances", "sqrtm_spd", "invsqrtm_spd", "logm_spd", "expm_spd", "airm_distance", "geodesic",
           "riemannian_mean", "log_euclidean_mean", "tangent_space", "euclidean_alignment", "riemannian_alignment",
           "subject_shift", "MDM", "balanced_accuracy"]
