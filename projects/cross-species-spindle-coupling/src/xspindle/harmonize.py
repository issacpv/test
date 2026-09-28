"""Species presets, resampling, crude NREM scoring, a one-call harmonised metric vector and the detector
multiverse.

``CONVENTIONAL`` presets reproduce typical literature settings per species/recording scale; ``HARMONIZED``
is one scale-free setting applied identically everywhere. The difference between the two, per recording, is
the 'detector-convention' component of any cross-species difference.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from . import coupling as cp
from . import detect as dt
from . import spectrum as sp


@dataclass(frozen=True)
class Preset:
    """Detector specification. Spindle duration limits are in cycles unless ``spindle_dur_s`` is set."""

    name: str
    so_band: Tuple[float, float] = (0.3, 1.5)
    so_percentile: float = 75.0
    so_dur_s: Tuple[float, float] = (0.5, 2.5)
    sigma_search: Tuple[float, float] = (8.0, 18.0)
    fixed_sigma: Optional[float] = None          # if set, use this centre frequency instead of the peak
    half_bandwidth: float = 2.0
    spindle_percentile: float = 95.0
    spindle_cycles: Tuple[float, float] = (5.0, 30.0)
    spindle_dur_s: Optional[Tuple[float, float]] = None
    so_polarity: str = "negative_first"
    target_fs: float = 200.0


HARMONIZED = Preset(name="harmonized")

CONVENTIONAL: Dict[str, Preset] = {
    # human scalp: SO 0.16-1.25 Hz, spindles fixed 11-16 Hz, 0.5-3 s
    "human_scalp": Preset(name="human_scalp", so_band=(0.16, 1.25), fixed_sigma=13.5, half_bandwidth=2.5,
                          spindle_dur_s=(0.5, 3.0)),
    # human intracranial: same bands, but depth polarity (down-state positive in many contacts)
    "human_ieeg": Preset(name="human_ieeg", so_band=(0.16, 1.25), fixed_sigma=13.5, half_bandwidth=2.5,
                         spindle_dur_s=(0.5, 3.0), so_polarity="positive_first"),
    # rat LFP: 'delta' 0.5-4 Hz SO band, spindles 10-16 Hz (rough literature convention), 0.4-2 s
    "rat_lfp": Preset(name="rat_lfp", so_band=(0.5, 4.0), fixed_sigma=13.0, half_bandwidth=3.0,
                      spindle_dur_s=(0.4, 2.0), so_polarity="positive_first"),
    # mouse LFP: spindles often 10-20 Hz; shorter events
    "mouse_lfp": Preset(name="mouse_lfp", so_band=(0.5, 4.0), fixed_sigma=15.0, half_bandwidth=5.0,
                        spindle_dur_s=(0.3, 2.0), so_polarity="positive_first"),
}


def resample_to(x: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, float]:
    """Polyphase resampling to ``target_fs`` (returns the signal and the exact new rate)."""
    if abs(fs - target_fs) < 1e-9:
        return np.asarray(x, dtype=float), fs
    from math import gcd
    up, down = int(round(target_fs)), int(round(fs))
    g = gcd(up, down)
    y = resample_poly(np.asarray(x, dtype=float), up // g, down // g)
    return y, fs * (up // g) / (down // g)


def crude_nrem_mask(x: np.ndarray, fs: float, epoch_s: float = 10.0, delta=(0.5, 4.0), theta=(5.0, 9.0),
                    delta_z_min: float = 0.0, ratio_max: float = 1.0) -> np.ndarray:
    """Very rough NREM scoring for sessions without state labels: epochs with above-median delta power and
    a theta/delta ratio below ``ratio_max``. Returns a boolean mask per sample. Use only as a fallback and
    validate against provided labels where available."""
    n = int(epoch_s * fs)
    n_ep = len(x) // n
    ep = np.asarray(x[: n_ep * n], dtype=float).reshape(n_ep, n)
    f, p = welch(ep, fs=fs, nperseg=min(int(4 * fs), n), axis=1)
    def _bp(lo, hi):
        s = (f >= lo) & (f < hi)
        return np.trapezoid(p[:, s], f[s], axis=1)
    d = np.log(_bp(*delta) + 1e-12)
    t = np.log(_bp(*theta) + 1e-12)
    dz = (d - d.mean()) / (d.std() + 1e-12)
    nrem = (dz > delta_z_min) & (np.exp(t - d) < ratio_max)
    mask = np.zeros(len(x), dtype=bool)
    mask[: n_ep * n] = np.repeat(nrem, n)
    return mask


def harmonized_metrics(x: np.ndarray, fs: float, preset: Preset = HARMONIZED, n_perm: int = 300,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Full metric vector for one NREM signal under one preset.

    Steps: resample -> sigma peak (or fixed) -> spindles -> SOs -> SO phase -> coupling -> summary.
    Metrics: sigma_peak_hz, sigma_found, spindle_density_per_min, spindle_dur_s, spindle_cycles,
    so_density_per_min, so_duration_s, so_freq_hz, frac_coupled, mean_phase, mvl, mvl_z, rayleigh_p,
    offset_cycles_mean, offset_s_mean, tort_mi.
    """
    y, fs2 = resample_to(x, fs, preset.target_fs)
    dur = len(y) / fs2
    pk = sp.sigma_peak(y, fs2, search_range=preset.sigma_search)
    center = preset.fixed_sigma if preset.fixed_sigma is not None else pk["peak_freq"]
    out: Dict[str, float] = {"preset": preset.name, "duration_s": dur, "sigma_peak_hz": pk["peak_freq"],
                             "sigma_found": pk["found"], "aperiodic_exponent": pk["aperiodic_exponent"]}
    if not np.isfinite(center):
        out.update({k: np.nan for k in ("spindle_density_per_min", "spindle_dur_s", "spindle_cycles",
                                        "so_density_per_min", "so_duration_s", "so_freq_hz", "frac_coupled",
                                        "mean_phase", "mvl", "mvl_z", "rayleigh_p", "offset_cycles_mean",
                                        "offset_s_mean", "tort_mi", "n_coupled")})
        return out
    if preset.spindle_dur_s is not None:
        spindles = dt.detect_spindles(y, fs2, center, preset.half_bandwidth, preset.spindle_percentile,
                                      min_dur=preset.spindle_dur_s[0], max_dur=preset.spindle_dur_s[1])
    else:
        spindles = dt.detect_spindles(y, fs2, center, preset.half_bandwidth, preset.spindle_percentile,
                                      min_cycles=preset.spindle_cycles[0], max_cycles=preset.spindle_cycles[1])
    so = dt.detect_slow_oscillations(y, fs2, preset.so_band, preset.so_dur_s[0], preset.so_dur_s[1],
                                     preset.so_percentile, polarity=preset.so_polarity)
    phase = cp.so_phase(y, fs2, preset.so_band, invert=(preset.so_polarity == "positive_first"))
    coupled = cp.couple_spindles_to_so(spindles, so, phase, fs2)
    summ = cp.coupling_summary(coupled, phase, fs2, n_perm=n_perm, rng=rng)
    env, _ = dt.spindle_envelope(y, fs2, center, preset.half_bandwidth)
    out.update({
        "sigma_center_used_hz": float(center),
        "spindle_density_per_min": dt.event_density(spindles, dur),
        "spindle_dur_s": float(spindles["duration_s"].mean()) if len(spindles) else np.nan,
        "spindle_cycles": float(spindles["n_cycles"].mean()) if len(spindles) else np.nan,
        "so_density_per_min": dt.event_density(so, dur),
        "so_duration_s": float(so["duration_s"].mean()) if len(so) else np.nan,
        "so_freq_hz": float(1.0 / so["duration_s"].mean()) if len(so) else np.nan,
        "frac_coupled": float(len(coupled) / len(spindles)) if len(spindles) else np.nan,
        "n_coupled": int(summ["n"]), "mean_phase": summ["mean_phase"], "mvl": summ["mvl"],
        "mvl_z": summ["mvl_z"], "rayleigh_p": summ["rayleigh_p"],
        "offset_cycles_mean": summ["offset_cycles_mean"], "offset_s_mean": summ["offset_s_mean"],
        "tort_mi": cp.tort_modulation_index(phase, env),
    })
    return out


def multiverse_grid(so_bands: Iterable[Tuple[float, float]] = ((0.16, 1.25), (0.3, 1.5), (0.5, 4.0)),
                    sigma_rules: Iterable[Optional[float]] = (None, 13.5),
                    percentiles: Iterable[float] = (90.0, 95.0),
                    duration_rules: Iterable[str] = ("cycles", "seconds")) -> List[Preset]:
    """Factorial grid of presets (default 3 x 2 x 2 x 2 = 24 specifications)."""
    grid = []
    for so_band, sig, pct, drule in itertools.product(so_bands, sigma_rules, percentiles, duration_rules):
        name = f"so{so_band[0]}-{so_band[1]}_sig{'peak' if sig is None else sig}_p{pct:g}_{drule}"
        p = Preset(name=name, so_band=so_band, fixed_sigma=sig, spindle_percentile=pct,
                   spindle_dur_s=(0.5, 3.0) if drule == "seconds" else None)
        grid.append(p)
    return grid


def detector_multiverse(x: np.ndarray, fs: float, presets: Iterable[Preset], n_perm: int = 100,
                        rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Run ``harmonized_metrics`` for every preset; one row per specification."""
    rows = [harmonized_metrics(x, fs, p, n_perm=n_perm, rng=rng) for p in presets]
    return pd.DataFrame(rows)


def specification_variance(df: pd.DataFrame, metric: str, factors: Iterable[str]) -> pd.Series:
    """Share of variance of ``metric`` explained by each factor (eta squared from one-way decompositions)."""
    y = df[metric].to_numpy(dtype=float)
    ok = np.isfinite(y)
    y = y[ok]
    tot = ((y - y.mean()) ** 2).sum()
    out = {}
    for f in factors:
        g = df.loc[ok, f]
        ss_between = sum(((y[g.to_numpy() == lvl].mean() - y.mean()) ** 2) * (g == lvl).sum() for lvl in g.unique())
        out[f] = float(ss_between / tot) if tot > 0 else np.nan
    return pd.Series(out, name=f"eta2_{metric}")


def synthetic_nrem(duration_s: float, fs: float, rng: np.random.Generator, so_freq: float = 0.8,
                   spindle_freq: float = 13.0, coupling_phase: float = 0.0, spindle_prob: float = 0.6,
                   so_amp: float = 3.0, spindle_amp: float = 1.5, noise_amp: float = 1.0) -> np.ndarray:
    """Synthetic NREM LFP: an SO train with spindles locked to a chosen SO phase, plus 1/f noise.

    ``coupling_phase`` = 0 puts spindle peaks on the SO peak (up-state under the 'negative_first' convention).
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    so = so_amp * np.cos(2 * np.pi * so_freq * t)
    x = so + noise_amp * sp.pink_noise(n, fs, rng)
    period = 1.0 / so_freq
    spindle_dur = 0.8
    for k in range(int(duration_s * so_freq)):
        if rng.random() > spindle_prob:
            continue
        # phase phi of cos(2 pi f t) occurs at t = (phi / 2pi) / f within each cycle
        centre = k * period + (coupling_phase / (2 * np.pi)) * period
        i0 = int((centre - spindle_dur / 2) * fs)
        i1 = int((centre + spindle_dur / 2) * fs)
        if i0 < 0 or i1 >= n:
            continue
        tt = t[i0:i1] - centre
        env = np.hanning(i1 - i0)
        x[i0:i1] += spindle_amp * env * np.sin(2 * np.pi * spindle_freq * tt)
    return x
