"""Whole-brain models: linear OU (analytic FC) and Hopf oscillators (simulated), plus forward filters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def linear_fc(W: np.ndarray, G: float, tau: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Analytic FC of dx = (-x/tau + G W x) dt + sigma dB (Galan-style linear model).

    The stationary covariance C solves A C + C A^T + sigma^2 I = 0 with A = -I/tau + G W;
    FC is the correlation matrix of C. Returns NaN matrix when A is unstable.
    """
    n = len(W)
    A = -np.eye(n) / tau + G * W
    if np.max(np.real(np.linalg.eigvals(A))) >= 0:
        return np.full((n, n), np.nan)
    C = solve_continuous_lyapunov(A, -sigma ** 2 * np.eye(n))
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def max_stable_coupling(W: np.ndarray, tau: float = 1.0) -> float:
    """Largest G for which the linear model is stable (real part of leading eigenvalue < 0)."""
    ev = np.real(np.linalg.eigvals(W)).max()
    return float(1.0 / (tau * ev)) if ev > 0 else float("inf")


@dataclass
class HopfParams:
    a: float = -0.02          # bifurcation parameter (negative: noisy fixed point)
    omega_hz: Sequence[float] | float = 0.05
    sigma: float = 0.02
    dt: float = 0.1           # s
    duration: float = 600.0   # s
    burn_in: float = 60.0     # s


def simulate_hopf(W: np.ndarray, G: float, p: HopfParams = HopfParams(), seed: int = 0) -> np.ndarray:
    """Stuart-Landau (Hopf) network model; returns (T, N) real part after burn-in.

    dz_j = [a + i w_j - |z_j|^2] z_j dt + G sum_k W_jk (z_k - z_j) dt + sigma dB_j
    """
    rng = np.random.default_rng(seed)
    n = len(W)
    omega = 2 * np.pi * (np.full(n, p.omega_hz) if np.isscalar(p.omega_hz) else np.asarray(p.omega_hz))
    z = 0.1 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    n_steps = int(round(p.duration / p.dt))
    n_burn = int(round(p.burn_in / p.dt))
    out = np.empty((n_steps - n_burn, n))
    sq = np.sqrt(p.dt)
    Wsum = W.sum(axis=1)
    for t in range(n_steps):
        coupling = G * (W @ z - Wsum * z)
        dz = (p.a + 1j * omega - np.abs(z) ** 2) * z + coupling
        z = z + dz * p.dt + p.sigma * sq * (rng.normal(size=n) + 1j * rng.normal(size=n))
        if t >= n_burn:
            out[t - n_burn] = z.real
    return out


def exponential_kernel(dt: float, tau_rise: float, tau_decay: float, length: float) -> np.ndarray:
    """Normalised difference-of-exponentials kernel (calcium indicator or crude hemodynamic response)."""
    t = np.arange(0, length, dt)
    k = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    return k / max(k.sum(), 1e-12)


def apply_kernel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution of each column of (T, N) with ``kernel``."""
    T, n = x.shape
    out = np.empty_like(x)
    for j in range(n):
        out[:, j] = np.convolve(x[:, j], kernel)[:T]
    return out


def gcamp_kernel(dt: float) -> np.ndarray:
    return exponential_kernel(dt, tau_rise=0.05, tau_decay=1.0, length=6.0)


def hrf_kernel(dt: float) -> np.ndarray:
    """Coarse HRF as difference of gammas via exponentials (peak ~ 3-4 s); use TVB's Balloon model for the real study."""
    return exponential_kernel(dt, tau_rise=1.0, tau_decay=3.0, length=20.0)


def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    T = (x.shape[0] // factor) * factor
    return x[:T].reshape(T // factor, factor, x.shape[1]).mean(axis=1)


def coupling_sweep(W: np.ndarray, fc_emp: np.ndarray, gs: Sequence[float],
                   score: Callable[[np.ndarray, np.ndarray], float],
                   model: str = "linear", hopf: Optional[HopfParams] = None, seed: int = 0,
                   kernel: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Fit score as a function of global coupling for the linear or Hopf model."""
    scores = np.full(len(gs), np.nan)
    for i, g in enumerate(gs):
        if model == "linear":
            fc = linear_fc(W, g)
        else:
            x = simulate_hopf(W, g, hopf or HopfParams(), seed=seed)
            if kernel is not None:
                x = apply_kernel(x, kernel)
            fc = np.corrcoef(x.T)
        if np.isfinite(fc).all():
            scores[i] = score(fc, fc_emp)
    best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
    return {"G": np.asarray(gs, float), "score": scores, "best_G": np.asarray(gs, float)[best] if best >= 0 else np.nan,
            "best_score": scores[best] if best >= 0 else np.nan}
