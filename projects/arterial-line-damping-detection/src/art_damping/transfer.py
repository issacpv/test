"""Catheter-transducer dynamics: second-order model, Gardner adequacy, synthetic waveforms.

A fluid-filled arterial line behaves as an under/over-damped second-order
system with natural frequency ``fn`` (Hz) and damping coefficient ``zeta``
(Gardner, Anesthesiology 1981). Under-damping (low zeta, low fn) overshoots
systolic pressure and exaggerates dP/dt; over-damping (high zeta, or air
bubbles/clots lowering fn) blunts the pulse. Mean pressure is nearly unaffected
by either, which is why the harm is concentrated in systolic/diastolic-based
decisions and in pulse-contour analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import bilinear, lfilter, lfilter_zi


def catheter_system(fn_hz: float, zeta: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Digital (bilinear, pre-warped) coefficients of H(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2)."""
    if fn_hz <= 0 or fn_hz >= fs / 2:
        raise ValueError("fn_hz must be in (0, fs/2)")
    wn = 2 * np.pi * fn_hz
    # pre-warp so the digital natural frequency matches fn_hz
    wn_w = 2 * fs * np.tan(wn / (2 * fs))
    b, a = bilinear([wn_w ** 2], [1.0, 2 * zeta * wn_w, wn_w ** 2], fs)
    return b, a


def apply_catheter_system(p_true: np.ndarray, fs: float, fn_hz: float, zeta: float) -> np.ndarray:
    """Measured pressure = true pressure passed through the catheter-transducer system."""
    b, a = catheter_system(fn_hz, zeta, fs)
    zi = lfilter_zi(b, a) * p_true[0]
    y, _ = lfilter(b, a, p_true, zi=zi)
    return y


# Approximate lower bound of the adequate damping band as a function of fn (Gardner-style chart).
_ADEQ_FN = np.array([5.0, 7.5, 10.0, 15.0, 25.0, 40.0])
_ADEQ_ZMIN = np.array([0.95, 0.80, 0.50, 0.30, 0.15, 0.10])
ZETA_MAX_ADEQUATE = 0.85


def gardner_adequacy(fn_hz: float, zeta: float) -> str:
    """Classify a (fn, zeta) pair as ``adequate``, ``underdamped``, ``overdamped`` or ``inadequate_low_fn``.

    This is a piecewise-linear *approximation* of the adequacy region of Gardner's (1981) chart;
    validate the breakpoints against the printed nomogram before clinical interpretation.
    """
    if not np.isfinite(fn_hz) or not np.isfinite(zeta):
        return "unknown"
    if fn_hz < 5.0:
        return "inadequate_low_fn"
    zmin = float(np.interp(fn_hz, _ADEQ_FN, _ADEQ_ZMIN))
    if zeta > ZETA_MAX_ADEQUATE:
        return "overdamped"
    if zeta < zmin:
        return "underdamped"
    return "adequate"


@dataclass
class SyntheticABP:
    t: np.ndarray
    abp: np.ndarray
    onsets: np.ndarray
    fs: float


def ejection_profile(t_ej: float, fs: float, t_peak_frac: float = 0.06, sharpness: float = 1.0,
                     backflow_frac: float = 0.06) -> np.ndarray:
    """Unit-area aortic flow pulse: fast skewed rise (gamma-like, exponent ``sharpness``) to a peak at
    ``t_peak_frac * t_ej`` followed by a decay, then a short negative lobe (valve-closure backflow)
    that produces the dicrotic notch.

    A fast upstroke is essential: it is the high-frequency content of the pulse that a resonant
    (under-damped) catheter system amplifies.
    """
    n = max(4, int(round(t_ej * fs)))
    tau = np.arange(n) / n
    tp = t_peak_frac
    k = sharpness
    pos = (tau / tp) ** k * np.exp(k * (1 - tau / tp))
    n_back = max(2, int(round(0.08 * n)))
    back = -backflow_frac * np.sin(np.pi * np.arange(n_back) / n_back)
    prof = np.r_[pos, back]
    return prof / (prof[prof > 0].sum() / fs)  # positive lobe integrates to 1 mL per mL of SV


def synthetic_true_abp(fs: float = 125.0, n_beats: int = 40, hr: float = 75.0, sv_ml: float = 70.0,
                       R: float = 1.0, C: float = 1.5, noise_mmhg: float = 0.3, seed: int = 0,
                       t_peak_frac: float = 0.06, sharpness: float = 1.0) -> SyntheticABP:
    """Intra-arterial ('true') pressure from a two-element Windkessel driven by ``ejection_profile``."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    rr = 60.0 / hr
    t_ej = 0.30 * np.sqrt(rr / 0.8)
    onset_times = np.cumsum(np.r_[0.5, rr * (1 + 0.02 * rng.normal(size=n_beats - 1))])
    n = int(np.ceil((onset_times[-1] + rr + 0.5) * fs))
    t = np.arange(n) / fs
    q = np.zeros(n)
    prof = ejection_profile(t_ej, fs, t_peak_frac, sharpness)
    for t0 in onset_times:
        sv = sv_ml * (1 + 0.05 * rng.normal())
        i0 = int(round(t0 * fs))
        i1 = min(n, i0 + prof.size)
        q[i0:i1] += sv * prof[: i1 - i0]
    p = np.empty(n)
    p[0] = 80.0
    for i in range(1, n):
        p[i] = p[i - 1] + dt * (q[i - 1] / C - p[i - 1] / (R * C))
    p += noise_mmhg * rng.normal(size=n)
    return SyntheticABP(t, p, np.round(onset_times * fs).astype(int), fs)


def synthetic_flush(fs: float, fn_hz: float, zeta: float, baseline: float | np.ndarray = 80.0,
                    plateau_mmhg: float = 300.0, pre_s: float = 1.0, plateau_s: float = 0.6, post_s: float = 2.0) -> np.ndarray:
    """Fast-flush (square-wave) test as *measured* through a catheter system with given fn/zeta.

    ``baseline`` may be a scalar or a pulsatile array of length ``round((pre_s+plateau_s+post_s)*fs)``.
    """
    n_pre, n_pl, n_post = int(pre_s * fs), int(plateau_s * fs), int(post_s * fs)
    n = n_pre + n_pl + n_post
    base = np.full(n, float(baseline)) if np.isscalar(baseline) else np.asarray(baseline, float)[:n]
    true = base.copy()
    true[n_pre:n_pre + n_pl] = plateau_mmhg
    return apply_catheter_system(true, fs, fn_hz, zeta)
