"""Reference-tissue kinetic models in pure numpy/scipy.

All models take a target TAC ``ct`` and a reference TAC ``cr`` (same frames),
a ``FrameTiming`` and optional frame weights, and return a dict with ``bp``
(BP_ND) plus model-specific parameters and fit diagnostics.

Conventions (Innis et al., 2007): k2a = k2 / (1 + BP_ND); k2′ = reference
region efflux constant; DVR = BP_ND + 1. Times in minutes.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import fftconvolve

from .tacs import FrameTiming

DT_DEFAULT = 1.0 / 12.0  # 5-second fine grid (frames are ≥ 10 s)
_BASIS_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# ----------------------------------------------------------------------------
# forward model helpers
# ----------------------------------------------------------------------------


def _fine_grid(timing: FrameTiming, dt: float = 1.0 / 60.0) -> np.ndarray:
    return np.arange(0.0, timing.total_minutes + dt, dt)


def _interp_tac(t_fine: np.ndarray, timing: FrameTiming, tac: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of a frame TAC (mid-times) onto a fine grid, zero at t=0."""
    x = np.concatenate([[0.0], timing.mid])
    y = np.concatenate([[0.0], tac])
    return np.interp(t_fine, x, y)


def _frame_average(t_fine: np.ndarray, y_fine: np.ndarray, timing: FrameTiming) -> np.ndarray:
    out = np.empty(len(timing.start))
    for i, (s, e) in enumerate(zip(timing.start, timing.end)):
        m = (t_fine >= s) & (t_fine < e)
        out[i] = y_fine[m].mean() if m.any() else np.interp((s + e) / 2, t_fine, y_fine)
    return out


def _conv_exp(t_fine: np.ndarray, y_fine: np.ndarray, rate: float) -> np.ndarray:
    """(y ⊗ exp(−rate·t))(t) on a uniform fine grid."""
    dt = t_fine[1] - t_fine[0]
    kern = np.exp(-rate * t_fine)
    return fftconvolve(y_fine, kern)[: len(t_fine)] * dt


def default_k2a_grid() -> np.ndarray:
    return np.logspace(np.log10(0.006), np.log10(0.6), 100)


def basis_matrix(cr: np.ndarray, timing: FrameTiming, k2a_grid: np.ndarray | None = None,
                 dt: float = DT_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Frame-averaged basis functions C_R ⊗ exp(−k2a·t) for every k2a (cached; independent of the target)."""
    k2a_grid = default_k2a_grid() if k2a_grid is None else np.asarray(k2a_grid, float)
    cr = np.asarray(cr, float)
    key = hashlib.md5(cr.tobytes() + timing.start.tobytes() + timing.duration.tobytes() + k2a_grid.tobytes()
                      + np.float64(dt).tobytes()).hexdigest()
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    t = _fine_grid(timing, dt)
    crf = _interp_tac(t, timing, cr)
    B = np.stack([_frame_average(t, _conv_exp(t, crf, k), timing) for k in k2a_grid], axis=1)
    if len(_BASIS_CACHE) > 256:
        _BASIS_CACHE.clear()
    _BASIS_CACHE[key] = (k2a_grid, B)
    return k2a_grid, B


def srtm_forward(cr: np.ndarray, timing: FrameTiming, r1: float, k2: float, bp: float, dt: float = DT_DEFAULT) -> np.ndarray:
    """Frame-averaged target TAC predicted by SRTM from a reference TAC."""
    t = _fine_grid(timing, dt)
    crf = _interp_tac(t, timing, cr)
    k2a = k2 / (1.0 + bp)
    ctf = r1 * crf + (k2 - r1 * k2a) * _conv_exp(t, crf, k2a)
    return _frame_average(t, ctf, timing)


def frame_weights(timing: FrameTiming, ct: np.ndarray, scheme: str = "duration") -> np.ndarray:
    """Weights for least squares: 'uniform', 'duration' (∝ frame length) or 'counts' (∝ duration/ct)."""
    if scheme == "uniform":
        w = np.ones(len(ct))
    elif scheme == "duration":
        w = timing.duration.copy()
    elif scheme == "counts":
        w = timing.duration / np.clip(np.abs(ct), 1e-6, None)
    else:
        raise ValueError(scheme)
    return w / w.sum() * len(w)


# ----------------------------------------------------------------------------
# SRTM (basis functions, Gunn et al., 1997) and SRTM2 (Wu & Carson, 2002)
# ----------------------------------------------------------------------------


def fit_srtm(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, weights: np.ndarray | None = None,
             k2a_grid: np.ndarray | None = None, dt: float = DT_DEFAULT) -> dict:
    """SRTM via basis functions: grid over k2a, linear LS for (R1, θ2)."""
    ct = np.asarray(ct, float)
    cr = np.asarray(cr, float)
    w = np.ones(len(ct)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    k2a_grid, B = basis_matrix(cr, timing, k2a_grid, dt)
    best = None
    for j, k2a in enumerate(k2a_grid):
        basis = B[:, j]
        A = np.column_stack([cr, basis]) * sw[:, None]
        theta, *_ = np.linalg.lstsq(A, ct * sw, rcond=None)
        resid = ct - (theta[0] * cr + theta[1] * basis)
        rss = float(np.sum(w * resid ** 2))
        if best is None or rss < best["rss"]:
            best = {"k2a": k2a, "theta": theta, "rss": rss, "fitted": theta[0] * cr + theta[1] * basis}
    r1, theta2 = best["theta"]
    k2a = best["k2a"]
    k2 = theta2 + r1 * k2a
    bp = k2 / k2a - 1.0
    return {"model": "srtm", "bp": float(bp), "r1": float(r1), "k2": float(k2), "k2a": float(k2a),
            "k2prime": float(k2 / r1) if r1 != 0 else np.nan, "rss": best["rss"], "fitted": best["fitted"]}


def fit_srtm2(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, k2prime: float, weights: np.ndarray | None = None,
              k2a_grid: np.ndarray | None = None, dt: float = DT_DEFAULT) -> dict:
    """SRTM2: k2′ fixed; for each k2a fit R1 only (1-parameter linear LS)."""
    ct = np.asarray(ct, float)
    cr = np.asarray(cr, float)
    w = np.ones(len(ct)) if weights is None else np.asarray(weights, float)
    k2a_grid, B = basis_matrix(cr, timing, k2a_grid, dt)
    best = None
    for j, k2a in enumerate(k2a_grid):
        basis = B[:, j]
        x = cr + (k2prime - k2a) * basis
        r1 = float(np.sum(w * x * ct) / np.sum(w * x * x))
        resid = ct - r1 * x
        rss = float(np.sum(w * resid ** 2))
        if best is None or rss < best["rss"]:
            best = {"k2a": k2a, "r1": r1, "rss": rss, "fitted": r1 * x}
    k2 = best["r1"] * k2prime
    bp = k2 / best["k2a"] - 1.0
    return {"model": "srtm2", "bp": float(bp), "r1": best["r1"], "k2": float(k2), "k2a": float(best["k2a"]),
            "k2prime": float(k2prime), "rss": best["rss"], "fitted": best["fitted"]}


# ----------------------------------------------------------------------------
# Logan reference (Logan et al., 1996) and MRTM / MRTM2 (Ichise et al., 2003)
# ----------------------------------------------------------------------------


def _cumint(timing: FrameTiming, tac: np.ndarray) -> np.ndarray:
    """∫0^t TAC using trapezoid on mid-times with a zero at t=0, evaluated at mid-times."""
    x = np.concatenate([[0.0], timing.mid])
    y = np.concatenate([[0.0], tac])
    return cumulative_trapezoid(y, x, initial=0.0)[1:]


def fit_logan_ref(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, t_star: float, k2prime: float | None = None,
                  weights: np.ndarray | None = None) -> dict:
    """Logan reference-tissue graphical analysis; BP = DVR − 1. ``k2prime=None`` drops the C_R/k2′ term."""
    ct = np.asarray(ct, float)
    cr = np.asarray(cr, float)
    int_ct, int_cr = _cumint(timing, ct), _cumint(timing, cr)
    m = (timing.mid >= t_star) & (ct > 0)
    if m.sum() < 3:
        return {"model": "logan_ref", "bp": np.nan, "dvr": np.nan, "n_points": int(m.sum()), "rss": np.nan}
    x = (int_cr + (cr / k2prime if k2prime else 0.0))[m] / ct[m]
    y = int_ct[m] / ct[m]
    w = np.ones(m.sum()) if weights is None else np.asarray(weights, float)[m]
    A = np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None]
    (dvr, intercept), *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    resid = y - (dvr * x + intercept)
    return {"model": "logan_ref", "bp": float(dvr - 1.0), "dvr": float(dvr), "intercept": float(intercept),
            "n_points": int(m.sum()), "rss": float(np.sum(w * resid ** 2))}


def fit_mrtm(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, t_star: float, weights: np.ndarray | None = None) -> dict:
    """MRTM (3 parameters): C_T = γ1∫C_R + γ2 C_R + γ3∫C_T for t ≥ t*; BP = −γ1/γ3 − 1; k2′ = γ1/γ2."""
    ct = np.asarray(ct, float)
    cr = np.asarray(cr, float)
    int_ct, int_cr = _cumint(timing, ct), _cumint(timing, cr)
    m = timing.mid >= t_star
    w = np.ones(m.sum()) if weights is None else np.asarray(weights, float)[m]
    A = np.column_stack([int_cr[m], cr[m], int_ct[m]]) * np.sqrt(w)[:, None]
    g, *_ = np.linalg.lstsq(A, ct[m] * np.sqrt(w), rcond=None)
    g1, g2, g3 = g
    resid = ct[m] - (A / np.sqrt(w)[:, None]) @ g
    return {"model": "mrtm", "bp": float(-g1 / g3 - 1.0), "k2prime": float(g1 / g2) if g2 != 0 else np.nan,
            "n_points": int(m.sum()), "rss": float(np.sum(w * resid ** 2))}


def fit_mrtm2(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, t_star: float, k2prime: float,
              weights: np.ndarray | None = None) -> dict:
    """MRTM2 (2 parameters, k2′ fixed): C_T = γ1(∫C_R + C_R/k2′) + γ3∫C_T; BP = −γ1/γ3 − 1."""
    ct = np.asarray(ct, float)
    cr = np.asarray(cr, float)
    int_ct, int_cr = _cumint(timing, ct), _cumint(timing, cr)
    m = timing.mid >= t_star
    w = np.ones(m.sum()) if weights is None else np.asarray(weights, float)[m]
    x1 = (int_cr + cr / k2prime)[m]
    A = np.column_stack([x1, int_ct[m]]) * np.sqrt(w)[:, None]
    (g1, g3), *_ = np.linalg.lstsq(A, ct[m] * np.sqrt(w), rcond=None)
    resid = ct[m] - (g1 * x1 + g3 * int_ct[m])
    return {"model": "mrtm2", "bp": float(-g1 / g3 - 1.0), "k2prime": float(k2prime), "n_points": int(m.sum()),
            "rss": float(np.sum(w * resid ** 2))}


def fit_suvr(ct: np.ndarray, cr: np.ndarray, timing: FrameTiming, window: tuple[float, float]) -> dict:
    """Late-window ratio: SUVR = mean(C_T)/mean(C_R) over [start, end] minutes; BP_ratio = SUVR − 1."""
    m = (timing.start >= window[0] - 1e-9) & (timing.end <= window[1] + 1e-9)
    if m.sum() == 0:
        return {"model": "suvr", "bp": np.nan, "suvr": np.nan, "n_frames": 0}
    w = timing.duration[m]
    suvr = float(np.sum(w * np.asarray(ct)[m]) / np.sum(w * np.asarray(cr)[m]))
    return {"model": "suvr", "bp": suvr - 1.0, "suvr": suvr, "n_frames": int(m.sum())}


def population_k2prime(tacs: dict[str, np.ndarray], cr: np.ndarray, timing: FrameTiming,
                       high_binding_regions: list[str], weights: np.ndarray | None = None) -> float:
    """Median SRTM k2′ over high-binding regions (the SRTM2/MRTM2 convention)."""
    vals = [fit_srtm(tacs[r], cr, timing, weights)["k2prime"] for r in high_binding_regions if r in tacs]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else np.nan


# ----------------------------------------------------------------------------
# Simulation (for tests and model validation)
# ----------------------------------------------------------------------------


def simulate_reference_tac(timing: FrameTiming, peak: float = 30.0, t_peak: float = 3.0, washout: float = 0.03) -> np.ndarray:
    """Gamma-variate-like reference TAC (kBq/mL) evaluated as frame averages."""
    t = _fine_grid(timing)
    alpha = 2.0
    beta = alpha / t_peak
    y = (t ** alpha) * np.exp(-beta * t)
    y = y / y.max() * peak
    y = y * np.exp(-washout * np.clip(t - t_peak, 0, None)) + 0.15 * peak * (1 - np.exp(-t / 5.0))
    return _frame_average(t, y, timing)


def add_noise(tac: np.ndarray, timing: FrameTiming, level: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian noise with SD ∝ sqrt(activity / frame_duration), scaled so the late frames have ~``level`` relative SD."""
    rng = np.random.default_rng() if rng is None else rng
    sd = level * np.sqrt(np.clip(tac, 1e-6, None) * tac.max() / np.clip(timing.duration, 1e-3, None) / 10.0)
    sd = sd / sd[-1] * level * max(tac[-1], 1e-6) if sd[-1] > 0 else sd
    return tac + rng.normal(0, sd)


def simulate_test_retest_dataset(n_subjects: int = 12, seed: int = 0, noise: float = 0.05,
                                 regions: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Simulated dataset with 2 sessions per subject, true BP per region with between-subject variability.

    Returns a long DataFrame (subject, session, region, frame, start, duration, ct, cr) and the truth.
    """
    rng = np.random.default_rng(seed)
    timing = FrameTiming.standard_90min()
    regions = regions or {"putamen": 3.0, "caudate": 2.6, "thalamus": 0.6, "frontal": 0.35, "temporal": 0.3, "occipital": 0.25}
    rows, truth = [], {"regions": regions, "subjects": {}}
    for s in range(n_subjects):
        sid = f"sub-{s + 1:02d}"
        subj_scale = np.exp(rng.normal(0, 0.12))
        r1 = float(np.clip(rng.normal(1.0, 0.08), 0.7, 1.3))
        k2 = float(np.clip(rng.normal(0.12, 0.02), 0.06, 0.25))
        truth["subjects"][sid] = {"scale": subj_scale, "r1": r1, "k2": k2}
        for ses in (1, 2):
            cr_clean = simulate_reference_tac(timing, peak=float(rng.normal(30, 3)))
            cr = add_noise(cr_clean, timing, noise, rng)
            for region, bp0 in regions.items():
                bp = bp0 * subj_scale * np.exp(rng.normal(0, 0.03))  # small session-to-session wobble
                ct = add_noise(srtm_forward(cr_clean, timing, r1, k2, bp), timing, noise, rng)
                for i in range(len(timing.start)):
                    rows.append({"subject": sid, "session": ses, "region": region, "frame": i,
                                 "start_min": timing.start[i], "duration_min": timing.duration[i], "ct": ct[i], "cr": cr[i]})
    return pd.DataFrame(rows), truth
