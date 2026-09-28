"""Intrinsic timescale and tuning estimators for single units.

The intrinsic timescale is the decay constant of the spike-count autocorrelation
(Murray et al., 2014). The estimators here are deliberately simple and transparent;
the README lists the aBC estimator (Zeraati et al., 2022) as a sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ------------------------------------------------------------------ autocorrelation
def bin_spikes(spike_times: np.ndarray, t_start: float, t_end: float, bin_size: float) -> np.ndarray:
    """Spike counts in consecutive bins of ``bin_size`` seconds within [t_start, t_end)."""
    edges = np.arange(t_start, t_end + bin_size * 0.5, bin_size)
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts.astype(float)


def spike_count_autocorr(
    spike_times: np.ndarray,
    epochs: Sequence[Tuple[float, float]],
    bin_size: float = 0.005,
    max_lag: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trial/epoch-averaged spike-count autocorrelation function.

    For each epoch the counts are mean-subtracted (within epoch) and the autocovariance at
    each lag is accumulated over all epochs, then normalised by the lag-0 value. This is the
    standard "pooled" estimator; short epochs bias long lags downward, so use epochs
    several times longer than ``max_lag``.

    Returns
    -------
    lags : (L,) lags in seconds (excluding 0).
    acf : (L,) autocorrelation values.
    """
    n_lags = int(round(max_lag / bin_size))
    num = np.zeros(n_lags + 1)
    cnt = np.zeros(n_lags + 1)
    for t0, t1 in epochs:
        x = bin_spikes(spike_times, t0, t1, bin_size)
        if len(x) <= n_lags + 1:
            continue
        x = x - x.mean()
        for k in range(n_lags + 1):
            num[k] += np.dot(x[: len(x) - k], x[k:])
            cnt[k] += len(x) - k
    with np.errstate(invalid="ignore", divide="ignore"):
        acov = num / cnt
        acf = acov / acov[0]
    lags = np.arange(1, n_lags + 1) * bin_size
    return lags, acf[1:]


def _exp_offset(t: np.ndarray, a: float, tau: float, b: float) -> np.ndarray:
    return a * np.exp(-t / tau) + b


def _two_exp(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, b: float) -> np.ndarray:
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + b


@dataclass
class TimescaleFit:
    tau: float                      # seconds (dominant timescale)
    amplitude: float
    offset: float
    r2: float
    tau_secondary: Optional[float] = None
    amplitude_secondary: Optional[float] = None


def fit_exponential_timescale(
    lags: np.ndarray,
    acf: np.ndarray,
    min_lag: float = 0.0,
    max_lag: Optional[float] = None,
    two_timescales: bool = False,
) -> TimescaleFit:
    """Fit ``a * exp(-lag/tau) + b`` (or a two-exponential) to an autocorrelation function.

    Lags below ``min_lag`` (e.g. to skip refractory/bursting effects at the shortest lag)
    and above ``max_lag`` are excluded. Returns the dominant (largest-amplitude) timescale.
    """
    lags = np.asarray(lags, float)
    acf = np.asarray(acf, float)
    m = np.isfinite(acf) & (lags >= min_lag)
    if max_lag is not None:
        m &= lags <= max_lag
    t, y = lags[m], acf[m]
    if len(t) < 5:
        raise ValueError("too few lags to fit")
    span = t.max() - t.min()
    try:
        if two_timescales:
            p0 = [y[0] * 0.5, span * 0.05, y[0] * 0.5, span * 0.5, y[-5:].mean()]
            lb = [0, 1e-4, 0, 1e-4, -1]
            ub = [2, span * 5, 2, span * 50, 1]
            popt, _ = curve_fit(_two_exp, t, y, p0=p0, bounds=(lb, ub), maxfev=20000)
            pred = _two_exp(t, *popt)
            (a1, tau1, a2, tau2, b) = popt
            if a2 > a1:
                a1, tau1, a2, tau2 = a2, tau2, a1, tau1
            r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            return TimescaleFit(tau=float(tau1), amplitude=float(a1), offset=float(b), r2=float(r2),
                                tau_secondary=float(tau2), amplitude_secondary=float(a2))
        p0 = [max(y[0], 1e-3), span * 0.2, y[-5:].mean()]
        popt, _ = curve_fit(_exp_offset, t, y, p0=p0, bounds=([0, 1e-4, -1], [2, span * 50, 1]), maxfev=20000)
    except RuntimeError as exc:  # pragma: no cover - optimiser failure
        raise ValueError(f"timescale fit failed: {exc}") from exc
    pred = _exp_offset(t, *popt)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return TimescaleFit(tau=float(popt[1]), amplitude=float(popt[0]), offset=float(popt[2]), r2=float(r2))


def intrinsic_timescale(
    spike_times: np.ndarray,
    epochs: Sequence[Tuple[float, float]],
    bin_size: float = 0.005,
    max_lag: float = 0.5,
    min_fit_lag: float = 0.01,
    min_rate_hz: float = 0.5,
) -> Optional[TimescaleFit]:
    """Convenience wrapper: autocorrelation + exponential fit, ``None`` if the unit is too sparse."""
    total = sum(t1 - t0 for t0, t1 in epochs)
    n = sum(np.sum((spike_times >= t0) & (spike_times < t1)) for t0, t1 in epochs)
    if total <= 0 or n / total < min_rate_hz:
        return None
    lags, acf = spike_count_autocorr(spike_times, epochs, bin_size, max_lag)
    return fit_exponential_timescale(lags, acf, min_lag=min_fit_lag)


# ------------------------------------------------------------------ tuning metrics
def orientation_selectivity(responses: np.ndarray, orientations_deg: np.ndarray) -> Dict[str, float]:
    """Vector-based OSI and DSI (Mazurek et al., 2014) plus preferred orientation/direction.

    Parameters
    ----------
    responses : mean firing rate per stimulus condition (non-negative).
    orientations_deg : direction of motion per condition (0-360).
    """
    r = np.clip(np.asarray(responses, float), 0, None)
    th = np.deg2rad(np.asarray(orientations_deg, float))
    denom = r.sum()
    if denom <= 0:
        return {"osi": 0.0, "dsi": 0.0, "pref_ori_deg": np.nan, "pref_dir_deg": np.nan}
    osi_vec = np.sum(r * np.exp(2j * th)) / denom
    dsi_vec = np.sum(r * np.exp(1j * th)) / denom
    return {"osi": float(abs(osi_vec)), "dsi": float(abs(dsi_vec)),
            "pref_ori_deg": float(np.rad2deg(np.angle(osi_vec)) / 2 % 180),
            "pref_dir_deg": float(np.rad2deg(np.angle(dsi_vec)) % 360)}


def response_latency(psth: np.ndarray, bin_size: float, baseline_bins: int, n_sd: float = 3.0,
                     min_consecutive: int = 2) -> float:
    """Latency (s) of the first ``min_consecutive`` bins exceeding baseline mean + n_sd * SD; NaN if none."""
    psth = np.asarray(psth, float)
    base = psth[:baseline_bins]
    thr = base.mean() + n_sd * (base.std(ddof=1) if len(base) > 1 else 0.0)
    above = psth[baseline_bins:] > thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_consecutive:
            return float((i - min_consecutive + 1) * bin_size)
    return float("nan")


def adaptation_index(psth: np.ndarray, onset_bin: int, early_bins: int, late_bins: int) -> float:
    """(late - early) / (late + early) response ratio; negative = adapting."""
    psth = np.asarray(psth, float)
    early = psth[onset_bin: onset_bin + early_bins].mean()
    late = psth[onset_bin + early_bins: onset_bin + early_bins + late_bins].mean()
    s = early + late
    return float((late - early) / s) if s > 0 else float("nan")


def preferred_temporal_frequency(responses: np.ndarray, tfs_hz: np.ndarray) -> float:
    """Response-weighted log-mean temporal frequency (Hz), robust to sparse sampling."""
    r = np.clip(np.asarray(responses, float), 0, None)
    tfs = np.asarray(tfs_hz, float)
    if r.sum() <= 0:
        return float("nan")
    return float(np.exp(np.sum(r * np.log(tfs)) / r.sum()))


def simulate_ou_spike_train(tau: float, rate_hz: float, duration: float, seed: int = 0,
                            dt: float = 0.001, modulation: float = 0.6) -> np.ndarray:
    """Inhomogeneous Poisson spikes driven by an Ornstein-Uhlenbeck rate with timescale ``tau``.

    Used for tests and for calibrating the estimator's bias; the spike-count autocorrelation
    of such a train decays with the OU timescale.
    """
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    x = np.zeros(n)
    a = np.exp(-dt / tau)
    s = np.sqrt(1 - a ** 2)
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * rng.standard_normal()
    rate = rate_hz * np.clip(1 + modulation * x, 0, None)
    spikes = rng.random(n) < rate * dt
    return np.nonzero(spikes)[0] * dt
