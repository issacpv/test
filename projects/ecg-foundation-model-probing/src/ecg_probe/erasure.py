"""Linear concept erasure for frozen embeddings.

Implements LEACE (Belrose et al., 2023, NeurIPS) in closed form and an iterative
INLP (Ravfogel et al., 2020) comparator. LEACE returns an affine map that makes
the target concept linearly unpredictable (linear guardedness) while changing
the representation as little as possible (minimal in the whitened norm).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _whitening(Sigma: np.ndarray, eps: float = 1e-6):
    """Symmetric whitening W and its inverse from a covariance matrix."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(vals ** -0.5) @ vecs.T
    W_inv = vecs @ np.diag(vals ** 0.5) @ vecs.T
    return W, W_inv


def leace_fit(Z: np.ndarray, y: np.ndarray):
    """Fit a LEACE eraser for a (possibly multiclass) concept y.

    Returns a callable ``erase(Z_new) -> Z_erased`` implementing the affine
    projection P(z) = z - W_inv @ P_proj @ W @ (z - mu), where P_proj projects
    onto the column space of the cross-covariance between whitened Z and one-hot y.
    """
    Z = np.asarray(Z, float)
    mu = Z.mean(0)
    Zc = Z - mu
    Sigma = np.cov(Zc, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
    W, W_inv = _whitening(Sigma)

    # one-hot encode y (drop last col to avoid collinearity)
    classes = np.unique(y)
    Y = np.stack([(y == c).astype(float) for c in classes[:-1]], axis=1) if len(classes) > 1 else y.reshape(-1, 1).astype(float)
    Yc = Y - Y.mean(0)

    Zw = Zc @ W                                  # whitened, centered
    cross = Zw.T @ Yc                            # (d, k)
    # orthonormal basis of the cross-covariance column space
    U, s, _ = np.linalg.svd(cross, full_matrices=False)
    rank = int((s > 1e-8).sum())
    U = U[:, :rank]
    P_proj = U @ U.T                             # projector onto concept subspace (whitened)

    def erase(Z_new: np.ndarray) -> np.ndarray:
        Zn = np.asarray(Z_new, float) - mu
        return mu + (Zn - (Zn @ W) @ P_proj @ W_inv)

    return erase


def inlp_fit(Z: np.ndarray, y: np.ndarray, n_iter: int = 10, seed: int = 0):
    """Iterative Null-space Projection: repeatedly null the direction a linear
    classifier uses to predict y. Comparator to LEACE."""
    Z = np.asarray(Z, float)
    d = Z.shape[1]
    P = np.eye(d)
    Zc = Z - Z.mean(0)
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        Zp = Zc @ P
        clf = LogisticRegression(max_iter=500)
        try:
            clf.fit(StandardScaler().fit_transform(Zp), y)
        except Exception:  # pragma: no cover
            break
        w = clf.coef_.reshape(-1)
        nrm = np.linalg.norm(w)
        if nrm < 1e-8:
            break
        w = w / nrm
        P = P @ (np.eye(d) - np.outer(w, w))
    mu = Z.mean(0)

    def erase(Z_new: np.ndarray) -> np.ndarray:
        return mu + (np.asarray(Z_new, float) - mu) @ P

    return erase


def utility_cost(auc_before: float, auc_after: float) -> float:
    """Diagnostic-AUROC loss from erasure (positive = utility lost)."""
    return float(auc_before - auc_after)
