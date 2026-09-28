"""Weighted linear DTI and DKI fitters and a multi-compartment signal simulator.

The signal model for DKI (Jensen et al., 2005) in direction ``n`` with b-value ``b`` is::

    ln S(b, n) = ln S0 - b D(n) + (b^2 / 6) MD^2 K(n)

with ``D(n) = n' D n`` and ``K(n) = MD^2 W(n) / D(n)^2`` where ``W`` is the fourth-order kurtosis
tensor (15 unique elements). The linear fit estimates ``D`` and ``W' = MD^2 W``. DTI is the same
fit without the kurtosis columns (valid for b <= ~1000-1200 s/mm^2).

These fitters exist for transparency and testing; production analyses should use DIPY's
``dipy.reconst.dti`` / ``dipy.reconst.dki`` (identical models with more robust estimators).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# unique kurtosis tensor elements and their multiplicities (Tabesh et al., 2011 ordering)
_W_INDEX: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2),
    (0, 0, 0, 1), (0, 0, 0, 2), (0, 1, 1, 1), (1, 1, 1, 2), (0, 2, 2, 2), (1, 2, 2, 2),
    (0, 0, 1, 1), (0, 0, 2, 2), (1, 1, 2, 2),
    (0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2),
)
_W_MULT = np.array([1, 1, 1, 4, 4, 4, 4, 4, 4, 6, 6, 6, 12, 12, 12], float)


@dataclass
class DTIResult:
    S0: np.ndarray
    tensor: np.ndarray  # (..., 3, 3)
    evals: np.ndarray   # (..., 3) descending
    evecs: np.ndarray   # (..., 3, 3) columns
    FA: np.ndarray
    MD: np.ndarray
    AD: np.ndarray
    RD: np.ndarray


@dataclass
class DKIResult(DTIResult):
    W: np.ndarray = None   # (..., 15) kurtosis tensor elements
    MK: np.ndarray = None
    AK: np.ndarray = None
    RK: np.ndarray = None


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` approximately uniformly distributed unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def _dti_columns(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    g = np.asarray(bvecs, float)
    b = np.asarray(bvals, float)[:, None]
    return -b * np.column_stack(
        [g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]]
    )


def _w_directional(g: np.ndarray) -> np.ndarray:
    """Rows of ``sum_ijkl mult * n_i n_j n_k n_l`` for the 15 unique elements, shape ``(n_dirs, 15)``."""
    g = np.asarray(g, float)
    return np.column_stack([m * g[:, i] * g[:, j] * g[:, k] * g[:, l] for (i, j, k, l), m in zip(_W_INDEX, _W_MULT)])


def _dki_columns(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    b = np.asarray(bvals, float)[:, None]
    return (b ** 2 / 6.0) * _w_directional(bvecs)


def _tensor_from_params(d6: np.ndarray) -> np.ndarray:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = (d6[..., i] for i in range(6))
    T = np.empty(d6.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = Dxx, Dyy, Dzz
    T[..., 0, 1] = T[..., 1, 0] = Dxy
    T[..., 0, 2] = T[..., 2, 0] = Dxz
    T[..., 1, 2] = T[..., 2, 1] = Dyz
    return T


def _wlls(X: np.ndarray, signal: np.ndarray, min_signal: float) -> np.ndarray:
    """Weighted linear least squares of ``ln S`` on ``X`` with weights ``S`` (per voxel)."""
    S = np.clip(np.asarray(signal, float), min_signal, None)
    y = np.log(S)
    flat = y.reshape(-1, y.shape[-1])
    w = S.reshape(-1, S.shape[-1])
    beta = np.empty((flat.shape[0], X.shape[1]))
    for v in range(flat.shape[0]):
        sw = w[v]
        beta[v], *_ = np.linalg.lstsq(X * sw[:, None], flat[v] * sw, rcond=None)
    return beta.reshape(y.shape[:-1] + (X.shape[1],))


def _tensor_metrics(d6: np.ndarray):
    T = _tensor_from_params(d6)
    evals, evecs = np.linalg.eigh(T)
    order = np.argsort(evals, axis=-1)[..., ::-1]
    evals = np.take_along_axis(evals, order, axis=-1)
    evecs = np.take_along_axis(evecs, order[..., None, :], axis=-1)
    evals = np.clip(evals, 0, None)
    md = evals.mean(axis=-1)
    num = np.sqrt(((evals - md[..., None]) ** 2).sum(-1))
    den = np.sqrt((evals ** 2).sum(-1))
    fa = np.sqrt(1.5) * num / np.where(den == 0, np.inf, den)
    return T, evals, evecs, fa, md, evals[..., 0], evals[..., 1:].mean(-1)


def fit_dti(signal: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, min_signal: float = 1e-6) -> DTIResult:
    """Weighted linear least-squares diffusion tensor fit. ``signal`` is ``(..., n_volumes)``."""
    X = np.column_stack([np.ones(len(bvals)), _dti_columns(bvals, bvecs)])
    beta = _wlls(X, signal, min_signal)
    T, evals, evecs, fa, md, ad, rd = _tensor_metrics(beta[..., 1:7])
    return DTIResult(np.exp(beta[..., 0]), T, evals, evecs, fa, md, ad, rd)


def fit_dki(
    signal: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    min_signal: float = 1e-6,
    n_dirs: int = 100,
    kurtosis_clip: Tuple[float, float] = (-3.0, 10.0),
) -> DKIResult:
    """Weighted linear least-squares diffusion kurtosis fit with MK/AK/RK by directional sampling."""
    X = np.column_stack([np.ones(len(bvals)), _dti_columns(bvals, bvecs), _dki_columns(bvals, bvecs)])
    beta = _wlls(X, signal, min_signal)
    T, evals, evecs, fa, md, ad, rd = _tensor_metrics(beta[..., 1:7])
    Wp = beta[..., 7:22]                       # W' = MD^2 W
    md_safe = np.where(md <= 0, np.nan, md)
    W = Wp / (md_safe[..., None] ** 2)

    def apparent_kurtosis(dirs: np.ndarray) -> np.ndarray:
        Dn = np.einsum("...ij,ki,kj->...k", T, dirs, dirs)          # (..., n_dirs)
        Wn = np.einsum("...w,kw->...k", W, _w_directional(dirs))
        K = (md_safe[..., None] ** 2) * Wn / np.where(Dn <= 0, np.nan, Dn) ** 2
        return np.clip(K, *kurtosis_clip)

    dirs = fibonacci_sphere(n_dirs)
    MK = np.nanmean(apparent_kurtosis(dirs), axis=-1)
    # axial: along e1; radial: mean over a circle perpendicular to e1 (per voxel)
    e1 = evecs[..., :, 0]
    AK = np.empty(md.shape)
    RK = np.empty(md.shape)
    flat_T, flat_W, flat_md, flat_e1 = (a.reshape(-1, *a.shape[md.ndim:]) for a in (T, W, md_safe, e1))
    angles = np.linspace(0, np.pi, 18, endpoint=False)
    for v in range(flat_T.shape[0]):
        e = flat_e1[v]
        u = np.cross(e, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(e, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(e, u)
        perp = np.array([np.cos(a) * u + np.sin(a) * w for a in angles])
        dirs_v = np.vstack([e[None, :], perp])
        Dn = np.einsum("ij,ki,kj->k", flat_T[v], dirs_v, dirs_v)
        Wn = _w_directional(dirs_v) @ flat_W[v]
        K = np.clip(flat_md[v] ** 2 * Wn / np.where(Dn <= 0, np.nan, Dn) ** 2, *kurtosis_clip)
        AK.flat[v], RK.flat[v] = K[0], np.nanmean(K[1:])
    return DKIResult(np.exp(beta[..., 0]), T, evals, evecs, fa, md, ad, rd, W=W, MK=MK, AK=AK, RK=RK)


def simulate_multicompartment_signal(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    f_stick: float = 0.5,
    f_zeppelin: float = 0.4,
    f_ball: float = 0.1,
    d_par: float = 1.7e-3,
    d_perp: float = 0.4e-3,
    d_iso: float = 3.0e-3,
    orientation: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    S0: float = 1.0,
    snr: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Stick + zeppelin + ball signal (non-Gaussian, so kurtosis > 0), optional Rician noise.

    A pure Gaussian tensor is obtained with ``f_stick = f_ball = 0, f_zeppelin = 1``.
    """
    mu = np.asarray(orientation, float)
    mu /= np.linalg.norm(mu)
    b = np.asarray(bvals, float)
    g = np.asarray(bvecs, float)
    cos2 = (g @ mu) ** 2
    stick = np.exp(-b * d_par * cos2)
    zepp = np.exp(-b * ((d_par - d_perp) * cos2 + d_perp))
    ball = np.exp(-b * d_iso)
    S = S0 * (f_stick * stick + f_zeppelin * zepp + f_ball * ball)
    if snr is not None:
        rng = np.random.default_rng() if rng is None else rng
        sigma = S0 / snr
        S = np.sqrt((S + rng.normal(0, sigma, S.shape)) ** 2 + rng.normal(0, sigma, S.shape) ** 2)
    return S
