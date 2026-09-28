"""Lag-optimized voxelwise CVR amplitude and delay estimation.

For each candidate delay the fine-resolution regressor (PetCO2, boxcar ⊗ HRF, or RVT) is shifted,
sampled at the volume times, and a GLM with Legendre drift terms and optional confounds is solved
for all voxels at once. The delay with the highest R² is kept; amplitude is expressed as %BOLD per
unit of the regressor (per mmHg when the regressor is PetCO2). This mirrors the approach of Moia
et al. (2020/2021) and Stickland et al. (2021) and the phys2cvr implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from . import physio


@dataclass
class CVRResult:
    amplitude: np.ndarray   # %BOLD per regressor unit, (n_voxels,)
    delay: np.ndarray       # seconds, (n_voxels,)
    r2: np.ndarray          # (n_voxels,)
    tstat: np.ndarray       # (n_voxels,)
    lags: np.ndarray        # searched lags (s)
    r2_by_lag: np.ndarray   # (n_lags, n_voxels) for diagnostics


def legendre_drift(n: int, order: int) -> np.ndarray:
    """Columns: Legendre polynomials of degree 0..order on [-1, 1]."""
    x = np.linspace(-1, 1, n)
    return np.column_stack([legendre.legval(x, [0] * k + [1]) for k in range(order + 1)])


def _fit_all_voxels(X: np.ndarray, Y: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS of every column of ``Y`` (n_t, n_vox) on ``X``; return beta[col], t[col], R² (regressor-only, partial)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = X.shape[0] - X.shape[1]
    sigma2 = np.sum(resid**2, axis=0) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * XtX_inv[col, col])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta[col] / se
    # partial R² of the regressor: compare with the nuisance-only model
    Xn = np.delete(X, col, axis=1)
    beta_n, *_ = np.linalg.lstsq(Xn, Y, rcond=None)
    ss_n = np.sum((Y - Xn @ beta_n) ** 2, axis=0)
    ss_f = np.sum(resid**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_n > 0, 1 - ss_f / ss_n, 0.0)
    return beta[col], t, r2


def fit_cvr(bold: np.ndarray, regressor: np.ndarray, reg_fs: float, tr: float, lag_range: tuple[float, float] = (-9.0, 9.0),
            lag_step: float = 0.3, drift_order: int = 3, confounds: np.ndarray | None = None,
            percent_signal: bool = True) -> CVRResult:
    """Voxelwise lag-optimized CVR fit.

    Parameters
    ----------
    bold : (n_voxels, n_vols) BOLD series (raw units; converted to % of the voxel mean if ``percent_signal``).
    regressor : fine-resolution regressor (e.g. PetCO2 in mmHg) sampled at ``reg_fs`` Hz,
        aligned so that t = 0 is the first volume.
    tr : repetition time (s).
    lag_range, lag_step : delay search grid in seconds (positive = BOLD lags the regressor).
    drift_order : Legendre polynomial order for slow drifts.
    confounds : optional (n_vols, k) nuisance regressors (motion etc.).
    """
    bold = np.asarray(bold, float)
    n_vox, n_vols = bold.shape
    mean = bold.mean(axis=1, keepdims=True)
    Y = (100.0 * (bold - mean) / np.where(mean == 0, 1, mean)).T if percent_signal else (bold - mean).T
    lags = np.arange(lag_range[0], lag_range[1] + 1e-9, lag_step)
    drift = legendre_drift(n_vols, drift_order)
    nuis = drift if confounds is None else np.column_stack([drift, np.asarray(confounds, float)])
    r2_by_lag = np.empty((lags.size, n_vox))
    beta_by_lag = np.empty((lags.size, n_vox))
    t_by_lag = np.empty((lags.size, n_vox))
    for i, lag in enumerate(lags):
        reg = physio.resample_to_tr(regressor, reg_fs, tr, n_vols, shift=lag)
        reg = reg - reg.mean()
        X = np.column_stack([nuis, reg])
        b, t, r2 = _fit_all_voxels(X, Y, col=X.shape[1] - 1)
        beta_by_lag[i], t_by_lag[i], r2_by_lag[i] = b, t, r2
    best = np.argmax(r2_by_lag, axis=0)
    idx = (best, np.arange(n_vox))
    return CVRResult(amplitude=beta_by_lag[idx], delay=lags[best], r2=r2_by_lag[idx], tstat=t_by_lag[idx],
                     lags=lags, r2_by_lag=r2_by_lag)


def null_regressor(regressor: np.ndarray, reg_fs: float, rng: np.random.Generator, method: str = "random_onsets",
                   onsets: np.ndarray | None = None, durations: np.ndarray | None = None,
                   min_gap: float = 5.0) -> np.ndarray:
    """Build one null regressor.

    ``random_onsets`` (recommended for block designs): the same number/duration of hold blocks placed at
    random, non-overlapping onsets and convolved with the HRF. ``phase``: Fourier phase randomization,
    which preserves the amplitude spectrum; for *periodic* block designs the dominant block frequency is
    kept and, combined with the lag search, this null is far too permissive (documented caveat).
    """
    n = regressor.size
    if method == "phase":
        F = np.fft.rfft(regressor - regressor.mean())
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size))
        phases[0] = 1.0
        return np.fft.irfft(np.abs(F) * phases, n=n) + regressor.mean()
    if method != "random_onsets":
        raise ValueError(method)
    if onsets is None or durations is None:
        raise ValueError("random_onsets needs the real onsets and durations")
    total = n / reg_fs
    durs = np.asarray(durations, float)
    for _ in range(1000):
        cand = np.sort(rng.uniform(0, total - durs.max(), durs.size))
        if np.all(np.diff(cand) >= durs[:-1] + min_gap):
            break
    x = physio.boxcar(n, reg_fs, cand, durs)
    x = physio.convolve_hrf(x, reg_fs)
    return x / (x.max() + 1e-12)


def r2_floor(bold: np.ndarray, regressor: np.ndarray, reg_fs: float, tr: float, n_null: int = 20,
             quantile: float = 0.95, rng: np.random.Generator | None = None, method: str = "random_onsets",
             onsets: np.ndarray | None = None, durations: np.ndarray | None = None, **fit_kwargs) -> float:
    """Empirical R² floor: refit with null regressors and return the ``quantile`` of max-over-lag R²
    across voxels and null realisations. Voxels below the floor should not contribute delay estimates."""
    rng = np.random.default_rng() if rng is None else rng
    vals = []
    for _ in range(n_null):
        surr = null_regressor(regressor, reg_fs, rng, method, onsets, durations)
        vals.append(fit_cvr(bold, surr, reg_fs, tr, **fit_kwargs).r2)
    return float(np.quantile(np.concatenate(vals), quantile))


def simulate_breath_hold_bold(n_vols: int, tr: float, onsets: np.ndarray, durations: np.ndarray,
                              amplitudes: np.ndarray, delays: np.ndarray, reg_fs: float = 10.0,
                              noise_sd: float = 0.3, baseline: float = 1000.0, drift: float = 5.0,
                              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic breath-hold data: per-voxel ``amplitudes`` (%BOLD per unit regressor) and ``delays`` (s).

    Returns ``(bold (n_vox, n_vols), regressor (fine, reg_fs))`` where the regressor is the boxcar ⊗ HRF.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_fine = int(np.ceil(n_vols * tr * reg_fs)) + int(40 * reg_fs)
    reg = physio.convolve_hrf(physio.boxcar(n_fine, reg_fs, onsets, durations), reg_fs)
    reg = reg / reg.max()
    bold = np.empty((amplitudes.size, n_vols))
    t = np.arange(n_vols) * tr
    for v, (a, d) in enumerate(zip(amplitudes, delays)):
        x = physio.resample_to_tr(reg, reg_fs, tr, n_vols, shift=d)
        pct = a * (x - x.mean()) + drift * (t / t[-1] - 0.5) + noise_sd * rng.standard_normal(n_vols)
        bold[v] = baseline * (1 + pct / 100.0)
    return bold, reg
