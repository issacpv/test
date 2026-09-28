"""Functional gradients via diffusion-map embedding (Margulies et al., 2016, PNAS).

The default pipeline mirrors BrainSpace (Vos de Wael et al., 2020, Commun
Biol): row-wise thresholding at 90 % sparsity, normalised-angle affinity,
diffusion map with alpha = 0.5, Procrustes alignment to a reference. A
self-contained NumPy/SciPy implementation is provided so that the whole
heritability pipeline is reproducible without BrainSpace; when BrainSpace is
installed, :func:`compute_gradients` can delegate to it for cross-checking.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg


# --------------------------------------------------------------------------- #
# Affinity
# --------------------------------------------------------------------------- #
def threshold_rows(fc: np.ndarray, sparsity: float = 0.9) -> np.ndarray:
    """Keep the top ``(1 - sparsity)`` fraction of entries in every row (others -> 0)."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    fc = np.asarray(fc, dtype=float).copy()
    n = fc.shape[1]
    k = int(np.ceil((1 - sparsity) * n))
    if k <= 0:
        raise ValueError("sparsity too high: no entries would survive")
    thr = np.sort(fc, axis=1)[:, -k][:, None]
    fc[fc < thr] = 0.0
    return fc


def normalized_angle_affinity(x: np.ndarray) -> np.ndarray:
    """Normalised-angle kernel: ``1 - arccos(cosine_similarity) / pi`` (in [0, 1])."""
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cos = (x / norms) @ (x / norms).T
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - np.arccos(cos) / np.pi


def cosine_affinity(x: np.ndarray) -> np.ndarray:
    """Cosine-similarity kernel clipped at 0."""
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.clip((x / norms) @ (x / norms).T, 0.0, 1.0)


def build_affinity(fc: np.ndarray, sparsity: float = 0.9, kernel: str = "normalized_angle") -> np.ndarray:
    """Threshold rows then compute the affinity matrix used by the diffusion map."""
    thr = threshold_rows(fc, sparsity)
    if kernel == "normalized_angle":
        A = normalized_angle_affinity(thr)
    elif kernel == "cosine":
        A = cosine_affinity(thr)
    elif kernel is None or kernel == "none":
        A = np.clip(thr, 0, None)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    return (A + A.T) / 2.0


# --------------------------------------------------------------------------- #
# Diffusion map
# --------------------------------------------------------------------------- #
def diffusion_map(
    affinity: np.ndarray,
    n_components: int = 10,
    alpha: float = 0.5,
    diffusion_time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diffusion-map embedding (Coifman & Lafon, 2006).

    Parameters
    ----------
    affinity
        Symmetric non-negative ``(N, N)`` kernel matrix.
    n_components
        Number of non-trivial eigenvectors to return.
    alpha
        Anisotropic normalisation (0.5 = Fokker-Planck; 1 = Laplace-Beltrami).
    diffusion_time
        ``t`` in ``lambda^t``; ``0`` uses the multi-scale ``lambda / (1 - lambda)``
        weighting as in BrainSpace / mapalign.

    Returns
    -------
    embedding : ``(N, n_components)``
    lambdas   : eigenvalues of the diffusion operator (descending, trivial one removed)
    """
    W = np.asarray(affinity, dtype=float)
    if W.shape[0] != W.shape[1]:
        raise ValueError("affinity must be square")
    if np.any(W < 0):
        raise ValueError("affinity must be non-negative")
    d = W.sum(axis=1)
    d[d == 0] = np.finfo(float).eps
    d_alpha = d ** (-alpha)
    L = W * d_alpha[:, None] * d_alpha[None, :]
    d_L = L.sum(axis=1)
    d_L[d_L == 0] = np.finfo(float).eps
    # symmetric conjugate of the Markov matrix D_L^-1 L
    d_half = d_L ** (-0.5)
    M_sym = L * d_half[:, None] * d_half[None, :]
    M_sym = (M_sym + M_sym.T) / 2.0
    n = W.shape[0]
    k = min(n_components + 1, n - 1)
    evals, evecs = linalg.eigh(M_sym, subset_by_index=[n - k, n - 1])
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # right eigenvectors of the Markov matrix
    psi = evecs * d_half[:, None]
    psi = psi / psi[:, [0]]  # normalise by the trivial eigenvector
    lambdas = evals[1:]
    psi = psi[:, 1:]
    if diffusion_time == 0:
        scale = lambdas / (1.0 - np.clip(lambdas, None, 1 - 1e-10))
    else:
        scale = lambdas**diffusion_time
    return psi * scale[None, :], lambdas


def explained_ratio(lambdas: np.ndarray) -> np.ndarray:
    """Fraction of the eigen-spectrum carried by each component (BrainSpace 'lambdas')."""
    lambdas = np.asarray(lambdas, dtype=float)
    return lambdas / lambdas.sum()


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def procrustes_align(source: np.ndarray, reference: np.ndarray, scaling: bool = False) -> np.ndarray:
    """Orthogonal Procrustes rotation of ``source`` onto ``reference`` (no centering)."""
    S = np.asarray(source, dtype=float)
    R = np.asarray(reference, dtype=float)
    if S.shape != R.shape:
        raise ValueError("source and reference must have the same shape")
    U, _, Vt = np.linalg.svd(S.T @ R)
    Q = U @ Vt
    aligned = S @ Q
    if scaling:
        num = np.trace(aligned.T @ R)
        den = np.trace(aligned.T @ aligned)
        aligned = aligned * (num / den if den > 0 else 1.0)
    return aligned


def align_group(gradients: Sequence[np.ndarray], n_iter: int = 10, reference: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Iterative Procrustes alignment of many subjects to a common template.

    If ``reference`` is None the first subject seeds the template and the
    template is updated to the mean of aligned subjects for ``n_iter`` rounds
    (the BrainSpace ``procrustes_alignment`` scheme).
    """
    grads = [np.asarray(g, dtype=float) for g in gradients]
    ref = grads[0].copy() if reference is None else np.asarray(reference, dtype=float)
    aligned = grads
    for _ in range(n_iter):
        aligned = [procrustes_align(g, ref) for g in grads]
        if reference is not None:
            break
        ref = np.mean(aligned, axis=0)
    return aligned


# --------------------------------------------------------------------------- #
# High-level API
# --------------------------------------------------------------------------- #
def compute_gradients(
    fc: np.ndarray,
    n_components: int = 10,
    sparsity: float = 0.9,
    kernel: str = "normalized_angle",
    alpha: float = 0.5,
    reference: Optional[np.ndarray] = None,
    use_brainspace: bool = False,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """FC matrix -> aligned gradients ``(N, n_components)`` and eigenvalues.

    Set ``use_brainspace=True`` to delegate to
    :class:`brainspace.gradient.GradientMaps` (must be installed); the own
    implementation is used otherwise. Both apply the same thresholding, kernel
    and Procrustes alignment, so results should match up to sign/scale.
    """
    fc = np.asarray(fc, dtype=float)
    if use_brainspace:
        try:
            from brainspace.gradient import GradientMaps  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("brainspace not installed: pip install brainspace") from exc
        gm = GradientMaps(
            n_components=n_components,
            approach="dm",
            kernel="normalized_angle" if kernel == "normalized_angle" else kernel,
            alignment="procrustes" if reference is not None else None,
            random_state=random_state,
        )
        if reference is not None:
            gm.fit(fc, sparsity=sparsity, reference=reference)
            grads = gm.aligned_
        else:
            gm.fit(fc, sparsity=sparsity)
            grads = gm.gradients_
        return np.asarray(grads), np.asarray(gm.lambdas_)
    A = build_affinity(fc, sparsity=sparsity, kernel=kernel)
    grads, lambdas = diffusion_map(A, n_components=n_components, alpha=alpha)
    if reference is not None:
        grads = procrustes_align(grads, np.asarray(reference)[:, : grads.shape[1]])
    return grads, lambdas


def gradient_dispersion(gradients: np.ndarray, k: int = 3) -> Tuple[np.ndarray, float]:
    """Regional eccentricity and global dispersion in the first ``k`` gradients.

    Eccentricity = Euclidean distance of each region from the centroid of the
    k-dimensional gradient space (Bethlehem et al., 2020, Neurobiol Aging);
    global dispersion = sum of squared eccentricities.
    """
    G = np.asarray(gradients, dtype=float)[:, :k]
    centroid = G.mean(axis=0)
    ecc = np.linalg.norm(G - centroid, axis=1)
    return ecc, float((ecc**2).sum())


def subject_gradient_features(
    fc_list: Sequence[np.ndarray],
    n_components: int = 3,
    reference: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute aligned gradients for many subjects.

    Returns ``loadings`` of shape ``(n_subjects, N, n_components)`` and
    ``eccentricity`` of shape ``(n_subjects, N)``. If ``reference`` is None a
    group template is built from the mean FC (group-level gradients) and every
    subject is aligned to it, which is the recommended practice for twin
    analyses (gradient sign/order must be identical across subjects).
    """
    if reference is None:
        mean_fc = np.mean(np.stack(fc_list), axis=0)
        reference, _ = compute_gradients(mean_fc, n_components=n_components, **kwargs)
    loads, ecc = [], []
    for fc in fc_list:
        g, _ = compute_gradients(fc, n_components=n_components, reference=reference, **kwargs)
        loads.append(g[:, :n_components])
        ecc.append(gradient_dispersion(g, k=n_components)[0])
    return np.stack(loads), np.stack(ecc)
