"""Circular statistics for clock times of ICU events."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

HOURS = 24.0


def hours_to_angle(hours: np.ndarray) -> np.ndarray:
    """Clock hours (0-24, fractional) to radians on the unit circle."""
    return 2 * np.pi * (np.asarray(hours, float) % HOURS) / HOURS


def angle_to_hours(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta, float) % (2 * np.pi)) * HOURS / (2 * np.pi)


def as_datetime_series(times) -> pd.Series:
    """Coerce a Series / DatetimeIndex / array of timestamps to a datetime Series (index preserved for Series)."""
    if isinstance(times, pd.Series):
        return pd.to_datetime(times)
    return pd.Series(pd.to_datetime(times))


def clock_hour(times) -> pd.Series:
    """Fractional hour of day from datetimes (Series, DatetimeIndex or array)."""
    t = as_datetime_series(times)
    return t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0


def resultant_vector(hours: np.ndarray) -> Tuple[float, float]:
    """Mean resultant length R in [0, 1] and circular mean (hours)."""
    th = hours_to_angle(hours)
    if len(th) == 0:
        return float("nan"), float("nan")
    z = np.mean(np.exp(1j * th))
    return float(abs(z)), float(angle_to_hours(np.angle(z)))


def rayleigh_test(hours: np.ndarray) -> Dict[str, float]:
    """Rayleigh test of uniformity on the circle (H0: no preferred hour).

    Uses the standard approximation p = exp(sqrt(1 + 4n + 4(n^2 - R_n^2)) - (1 + 2n)) with
    R_n = n * R (Zar, 1999), accurate for n >= 10.
    """
    hours = np.asarray(hours, float)
    n = len(hours)
    R, mean_h = resultant_vector(hours)
    if n < 2 or not np.isfinite(R):
        return {"n": n, "R": R, "mean_hour": mean_h, "z": float("nan"), "p": float("nan")}
    Rn = n * R
    z = Rn ** 2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    return {"n": int(n), "R": R, "mean_hour": mean_h, "z": float(z), "p": float(min(max(p, 0.0), 1.0))}


def hour_histogram(hours: np.ndarray, n_bins: int = 24, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, HOURS, n_bins + 1)
    counts, _ = np.histogram(np.asarray(hours, float) % HOURS, bins=edges)
    if density:
        counts = counts / max(counts.sum(), 1) * n_bins / HOURS
    return edges, counts


def peak_trough_ratio(hours: np.ndarray, n_bins: int = 24, smooth: int = 3) -> Dict[str, float]:
    """Ratio of the highest to the lowest (circularly smoothed) hourly rate, plus their hours."""
    _, h = hour_histogram(hours, n_bins, density=False)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        h = np.real(np.fft.ifft(np.fft.fft(h) * np.fft.fft(k, n_bins)))
    i_pk, i_tr = int(np.argmax(h)), int(np.argmin(h))
    return {"ratio": float(h[i_pk] / max(h[i_tr], 1e-9)), "peak_hour": i_pk * HOURS / n_bins, "trough_hour": i_tr * HOURS / n_bins}


def shift_change_locking(hours: np.ndarray, shift_hours: Sequence[float] = (7.0, 19.0), half_width_h: float = 1.0,
                         n_perm: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Excess of events within +/- half_width_h of shift changes, relative to uniform expectation.

    Returns the observed fraction, the expected fraction under uniformity, their ratio and two
    p-values: ``p_binomial`` against a uniform clock (events independent, rate constant) and
    ``p_rotation`` from rotating all events by random offsets, which preserves the shape of
    the distribution and only destroys its phase relative to the shift times. Note that the
    rotation null cannot distinguish any 12-h-periodic structure from locking to shift
    changes 12 h apart, so it is the conservative one; report both.
    """
    rng = np.random.default_rng(seed)
    h = np.asarray(hours, float) % HOURS
    n = len(h)

    def frac(x: np.ndarray) -> float:
        d = np.min([np.abs(((x - s + 12) % 24) - 12) for s in shift_hours], axis=0)
        return float(np.mean(d <= half_width_h))

    obs = frac(h)
    expected = min(1.0, len(shift_hours) * 2 * half_width_h / HOURS)
    k = int(round(obs * n))
    p_binom = float(stats.binomtest(k, n, expected, alternative="greater").pvalue) if n > 0 else float("nan")
    null = np.array([frac((h + rng.uniform(0, 24)) % HOURS) for _ in range(n_perm)])
    p_rot = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"n": int(n), "observed_frac": obs, "expected_frac": expected, "ratio": obs / expected,
            "p_binomial": p_binom, "p_rotation": p_rot}


def circular_difference_hours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed difference b - a on the 24-h circle, in (-12, 12]."""
    return ((np.asarray(b, float) - np.asarray(a, float) + 12) % 24) - 12


def bootstrap_R(hours: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    h = np.asarray(hours, float)
    boots = np.array([resultant_vector(rng.choice(h, len(h), replace=True))[0] for _ in range(n_boot)])
    R, mean_h = resultant_vector(h)
    return {"R": R, "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)), "mean_hour": mean_h}


def timing_audit(events: pd.DataFrame, group_cols: Sequence[str] = ("event_type",), hour_col: str = "clock_hour",
                 min_n: int = 100) -> pd.DataFrame:
    """Per-group circular summary of event clock hours (R, mean hour, Rayleigh p, peak/trough, shift locking)."""
    rows = []
    for key, g in events.groupby(list(group_cols)):
        h = g[hour_col].dropna().to_numpy(float)
        if len(h) < min_n:
            continue
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(rayleigh_test(h))
        row.update({f"pt_{k}": v for k, v in peak_trough_ratio(h).items()})
        row.update({f"shift_{k}": v for k, v in shift_change_locking(h, n_perm=500).items() if k != "n"})
        rows.append(row)
    return pd.DataFrame(rows)
