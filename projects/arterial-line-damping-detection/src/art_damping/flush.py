"""Mining fast-flush (square-wave) tests from recorded ABP to label damping objectively.

Nurses flush arterial lines routinely; the ~300 mmHg square wave and the ringing
that follows its release are recorded in the raw waveform. Instead of counting
overshoot peaks by hand (Gardner, Anesthesiology 1981), we *identify the system*
from the release: with the line at steady state on the plateau ``P``, the
measured pressure after the release at time ``t_off`` is

    y(t) = b0 (1 - s(t)) + b1 t (1 - s(t)) + P s(t),

where ``s`` is the second-order system's response to a unit level that steps
to zero at ``t_off`` (fractional-sample), ``b0 + b1 t`` the returning arterial
baseline (linear over the short window) and ``P`` the plateau. For each
``(fn, zeta, t_off)`` on a grid the linear coefficients ``(b0, b1, P)`` are solved
in closed form; the best grid point is refined by Nelder-Mead. Over-damped
responses (no oscillation) are handled naturally, which extrema counting cannot do.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import find_peaks, lfilter, lfilter_zi

from .transfer import catheter_system, gardner_adequacy


@dataclass
class FlushResult:
    release_idx: int
    fn_hz: float
    zeta: float
    baseline: float
    plateau: float
    r2: float
    n_extrema: int
    adequacy: str
    t_release_s: float = float("nan")


def detect_flush_events(abp: np.ndarray, fs: float, plateau_mmhg: float = 250.0,
                        min_plateau_s: float = 0.2, merge_gap_s: float = 1.0) -> list[tuple[int, int]]:
    """Contiguous runs above ``plateau_mmhg`` lasting >= ``min_plateau_s``; nearby runs are merged.

    Returns ``[(start, end)]`` with ``end`` = first index after the plateau (approximate release point).
    """
    above = np.asarray(abp) > plateau_mmhg
    if not above.any():
        return []
    edges = np.diff(above.astype(int), prepend=0, append=0)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= int(min_plateau_s * fs)]
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= int(merge_gap_s * fs):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _step_down_response(n: int, t_off: float, fs: float, fn: float, zeta: float) -> np.ndarray:
    """Response of the catheter system, initially at steady state 1, to a level that drops to 0 at
    fractional sample ``t_off`` (sample i covers [i, i+1); its input is the fraction of it before t_off)."""
    i = np.arange(n)
    x = np.clip(t_off - i, 0.0, 1.0)
    b, a = catheter_system(fn, zeta, fs)
    y, _ = lfilter(b, a, x, zi=lfilter_zi(b, a) * 1.0)
    return y


def _solve_linear(y: np.ndarray, t: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.c_[1 - s, t * (1 - s), s]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(np.sum((y - X @ coef) ** 2))
    return coef, sse


def fit_flush_response(abp: np.ndarray, fs: float, start: int, end: int, pre_s: float = 0.12, post_s: float = 0.4,
                       fn_grid: np.ndarray | None = None, zeta_grid: np.ndarray | None = None,
                       offset_grid: np.ndarray | None = None, refine: bool = True) -> FlushResult:
    """Identify ``(fn, zeta)`` of the catheter system from the release of one recorded flush ``[start, end)``.

    ``pre_s`` of plateau before the nominal release and ``post_s`` after it are modelled; the release
    time is fitted within ``offset_grid`` samples of ``end``.
    """
    a, b = max(0, end - int(pre_s * fs)), min(len(abp), end + int(post_s * fs))
    y = np.asarray(abp[a:b], float)
    n = b - a
    if n < 8:
        return FlushResult(end, np.nan, np.nan, np.nan, np.nan, np.nan, 0, "unknown")
    t = np.arange(n) / fs
    fn_grid = np.geomspace(3.0, min(55.0, 0.45 * fs), 16) if fn_grid is None else fn_grid
    zeta_grid = np.geomspace(0.05, 3.0, 12) if zeta_grid is None else zeta_grid
    offset_grid = np.arange(-4.0, 3.01, 1.0) if offset_grid is None else offset_grid
    nominal = float(end - a)
    best = (np.inf, np.nan, np.nan, np.nan, None)
    for off in offset_grid:
        t_off = nominal + off
        for fn in fn_grid:
            for z in zeta_grid:
                coef, sse = _solve_linear(y, t, _step_down_response(n, t_off, fs, fn, z))
                if sse < best[0]:
                    best = (sse, fn, z, t_off, coef)
    sse, fn, z, t_off, coef = best

    if refine and np.isfinite(fn):
        def obj(p: np.ndarray) -> float:
            f, zz, to = float(np.exp(p[0])), float(np.exp(p[1])), float(p[2])
            if not (1.0 < f < 0.49 * fs) or not (0.02 < zz < 5.0) or not (0 < to < n):
                return 1e30
            return _solve_linear(y, t, _step_down_response(n, to, fs, f, zz))[1]

        res = minimize(obj, np.r_[np.log([fn, z]), t_off], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 300})
        if res.success and res.fun < sse:
            fn, z, t_off = float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(res.x[2])
            coef, sse = _solve_linear(y, t, _step_down_response(n, t_off, fs, fn, z))
    r2 = 1.0 - sse / float(np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else np.nan
    k0 = int(np.ceil(t_off)) + 1
    post = y[k0:] - (coef[0] + coef[1] * t[k0:])
    pk_hi, _ = find_peaks(post, prominence=3.0)
    pk_lo, _ = find_peaks(-post, prominence=3.0)
    return FlushResult(end, float(fn), float(z), float(coef[0]), float(coef[2]), float(r2),
                       int(len(pk_hi) + len(pk_lo)), gardner_adequacy(fn, z), (a + t_off) / fs)


def ringing_parameters(abp: np.ndarray, fs: float, release_idx: int, plateau_s: float = 0.6, **kw) -> FlushResult:
    """Convenience wrapper when only the release index is known (assumes a plateau of ``plateau_s``)."""
    start = max(0, release_idx - int(plateau_s * fs))
    return fit_flush_response(abp, fs, start, release_idx, **kw)


def flush_labels(abp: np.ndarray, fs: float, min_r2: float = 0.8, **kw) -> pd.DataFrame:
    """All flush events in a record with (fn, zeta, adequacy); poor fits (r2 < min_r2) are marked 'unknown'."""
    rows = []
    for start, end in detect_flush_events(abp, fs):
        r = fit_flush_response(abp, fs, start, end, **kw)
        adequacy = r.adequacy if (np.isfinite(r.r2) and r.r2 >= min_r2) else "unknown"
        rows.append({"release_idx": r.release_idx, "t_release_s": r.t_release_s if np.isfinite(r.t_release_s) else r.release_idx / fs,
                     "fn_hz": r.fn_hz, "zeta": r.zeta, "baseline": r.baseline, "plateau": r.plateau, "r2": r.r2,
                     "n_extrema": r.n_extrema, "adequacy": adequacy})
    return pd.DataFrame(rows, columns=["release_idx", "t_release_s", "fn_hz", "zeta", "baseline", "plateau", "r2",
                                       "n_extrema", "adequacy"])


def propagate_labels(labels: pd.DataFrame, t_grid_s: np.ndarray, max_age_s: float = 4 * 3600.0) -> np.ndarray:
    """Assign each grid time the adequacy of the most recent flush within ``max_age_s`` (else 'unlabelled')."""
    out = np.full(len(t_grid_s), "unlabelled", dtype=object)
    if labels.empty:
        return out
    lab = labels.sort_values("t_release_s")
    idx = np.searchsorted(lab["t_release_s"].to_numpy(), t_grid_s, side="right") - 1
    for k, i in enumerate(idx):
        if i >= 0 and t_grid_s[k] - lab["t_release_s"].iloc[i] <= max_age_s:
            out[k] = lab["adequacy"].iloc[i]
    return out
