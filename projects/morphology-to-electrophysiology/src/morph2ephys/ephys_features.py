"""Intrinsic electrophysiology features from long-square current steps.

Own, dependency-free implementation mirroring the definitions used by the Allen Institute's
``ipfx`` (feature names match the Allen Cell Types table where possible) so that features can be
recomputed identically for Cell Types, Patch-seq and simulated (biophysical baseline) sweeps.
If ``ipfx`` is installed, ``extract_with_ipfx`` wraps it for cross-validation of this extractor.

A sweep is ``{"t": s, "v": mV, "i": pA}`` sampled uniformly; the stimulus window is detected from
the current trace. ``simulate_sweep`` provides an LIF-with-sag model for tests and sanity checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Sweep:
    t: np.ndarray
    v: np.ndarray
    i: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def stim_window(self, tol_pA: float = 1.0) -> Tuple[int, int]:
        """Index range ``[start, end)`` where the current deviates from its pre-stimulus baseline."""
        base = np.median(self.i[: max(10, self.i.size // 20)])
        on = np.flatnonzero(np.abs(self.i - base) > tol_pA)
        if on.size == 0:
            return 0, 0
        return int(on[0]), int(on[-1]) + 1

    @property
    def amplitude_pA(self) -> float:
        s, e = self.stim_window()
        if e <= s:
            return 0.0
        base = np.median(self.i[: max(10, s)]) if s > 0 else 0.0
        return float(np.median(self.i[s:e]) - base)


# ---------------------------------------------------------------------- spikes
def detect_spikes(sweep: Sweep, dvdt_thresh: float = 20.0, min_peak_mV: float = -30.0,
                  refractory_ms: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Spike threshold and peak indices using a dV/dt criterion (mV/ms), as in ipfx."""
    dt_ms = sweep.dt * 1e3
    dvdt = np.gradient(sweep.v, dt_ms)
    above = dvdt >= dvdt_thresh
    starts = np.flatnonzero(above & ~np.roll(above, 1))
    thr, pk = [], []
    last = -np.inf
    n = sweep.v.size
    for s in starts:
        if sweep.t[s] - last < refractory_ms * 1e-3:
            continue
        window_end = min(n, s + int(5.0 / dt_ms))  # peak must occur within 5 ms
        seg = sweep.v[s:window_end]
        if seg.size == 0:
            continue
        p = s + int(np.argmax(seg))
        if sweep.v[p] < min_peak_mV:
            continue
        thr.append(s)
        pk.append(p)
        last = sweep.t[s]
    return np.asarray(thr, dtype=int), np.asarray(pk, dtype=int)


def ap_width_half_height(sweep: Sweep, thr_idx: int, peak_idx: int) -> float:
    """AP width (ms) at half the threshold-to-peak amplitude."""
    v_thr, v_pk = sweep.v[thr_idx], sweep.v[peak_idx]
    half = v_thr + 0.5 * (v_pk - v_thr)
    up = sweep.v[thr_idx:peak_idx + 1]
    rise_i = thr_idx + int(np.flatnonzero(up >= half)[0]) if np.any(up >= half) else peak_idx
    after = sweep.v[peak_idx:]
    below = np.flatnonzero(after <= half)
    fall_i = peak_idx + int(below[0]) if below.size else sweep.v.size - 1
    return float((fall_i - rise_i) * sweep.dt * 1e3)


def adaptation_index(isis: Sequence[float]) -> float:
    """Allen 'adaptation': mean of (ISI[k+1] - ISI[k]) / (ISI[k+1] + ISI[k])."""
    isis = np.asarray(isis, dtype=float)
    if isis.size < 2:
        return float("nan")
    return float(np.mean((isis[1:] - isis[:-1]) / (isis[1:] + isis[:-1])))


# ---------------------------------------------------------------------- subthreshold
def steady_state_and_peak(sweep: Sweep, ss_fraction: float = 0.1) -> Tuple[float, float, float]:
    """(baseline, peak deflection voltage, steady-state voltage) of a subthreshold step."""
    s, e = sweep.stim_window()
    base = float(np.mean(sweep.v[max(0, s - int(0.05 / sweep.dt)):s])) if s > 0 else float(sweep.v[0])
    seg = sweep.v[s:e]
    n_ss = max(5, int(ss_fraction * seg.size))
    ss = float(np.mean(seg[-n_ss:]))
    amp = sweep.amplitude_pA
    peak = float(seg.min()) if amp < 0 else float(seg.max())
    return base, peak, ss


def sag_ratio(sweep: Sweep) -> float:
    """ipfx-style sag: (v_peak - v_steady) / (v_peak - v_baseline); 0 = no sag."""
    base, peak, ss = steady_state_and_peak(sweep)
    denom = peak - base
    return float((peak - ss) / denom) if abs(denom) > 1e-9 else float("nan")


def input_resistance_MOhm(sweeps: Sequence[Sweep]) -> float:
    """Slope of steady-state deflection (mV) vs current (pA) over subthreshold sweeps -> MOhm."""
    xs, ys = [], []
    for sw in sweeps:
        if detect_spikes(sw)[0].size:
            continue
        base, _, ss = steady_state_and_peak(sw)
        xs.append(sw.amplitude_pA)
        ys.append(ss - base)
    if len(xs) < 2:
        return float("nan")
    slope = np.polyfit(xs, ys, 1)[0]  # mV / pA = GOhm
    return float(slope * 1e3)


def membrane_time_constant_ms(sweep: Sweep, fit_fraction: float = 0.6) -> float:
    """Single-exponential fit of the onset response of a hyperpolarising step (ms)."""
    s, e = sweep.stim_window()
    base, peak, _ = steady_state_and_peak(sweep)
    seg = sweep.v[s:e]
    n_fit = max(10, int(fit_fraction * np.argmin(seg))) if sweep.amplitude_pA < 0 else max(10, int(0.2 * seg.size))
    y = np.abs(seg[:n_fit] - base)
    y_inf = np.abs(peak - base)
    resid = y_inf - y
    ok = resid > 1e-6
    if ok.sum() < 5:
        return float("nan")
    t = np.arange(n_fit) * sweep.dt * 1e3
    slope = np.polyfit(t[ok], np.log(resid[ok]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


# ---------------------------------------------------------------------- summary
def extract_long_square_features(sweeps: Sequence[Sweep]) -> Dict[str, float]:
    """Compute rin, sag, tau, rheobase, ap_width, adaptation, fi_slope, latency from a set of sweeps."""
    sweeps = sorted(sweeps, key=lambda s: s.amplitude_pA)
    hyper = [s for s in sweeps if s.amplitude_pA < 0]
    out: Dict[str, float] = {
        "rin": input_resistance_MOhm(sweeps),
        "sag": float(np.nanmean([sag_ratio(s) for s in hyper])) if hyper else float("nan"),
        "tau": float(np.nanmean([membrane_time_constant_ms(s) for s in hyper])) if hyper else float("nan"),
        "rheobase": float("nan"), "ap_width": float("nan"), "adaptation": float("nan"),
        "fi_slope": float("nan"), "latency": float("nan"),
    }
    rates, amps = [], []
    first_spiking: Optional[Tuple[Sweep, np.ndarray, np.ndarray]] = None
    for sw in sweeps:
        if sw.amplitude_pA <= 0:
            continue
        thr, pk = detect_spikes(sw)
        s, e = sw.stim_window()
        dur = (e - s) * sw.dt
        rates.append(thr.size / dur if dur > 0 else 0.0)
        amps.append(sw.amplitude_pA)
        if thr.size and first_spiking is None:
            first_spiking = (sw, thr, pk)
    if first_spiking is not None:
        sw, thr, pk = first_spiking
        s, _ = sw.stim_window()
        out["rheobase"] = sw.amplitude_pA
        out["ap_width"] = ap_width_half_height(sw, int(thr[0]), int(pk[0]))
        out["latency"] = float((sw.t[thr[0]] - sw.t[s]) * 1e3)
    # adaptation from the sweep with the most spikes (>= 3)
    best = None
    for sw in sweeps:
        thr, _ = detect_spikes(sw)
        if thr.size >= 3 and (best is None or thr.size > best[1].size):
            best = (sw, thr)
    if best is not None:
        out["adaptation"] = adaptation_index(np.diff(best[0].t[best[1]]))
    if len(amps) >= 2 and any(r > 0 for r in rates):
        out["fi_slope"] = float(np.polyfit(amps, rates, 1)[0])  # Hz / pA
    return out


def extract_with_ipfx(nwb_path: str) -> Dict[str, float]:  # pragma: no cover - optional dependency
    """Cross-check: run ipfx's data-set feature extraction on an Allen NWB file."""
    from ipfx.dataset.create import create_ephys_data_set  # type: ignore
    from ipfx.data_set_features import extract_data_set_features  # type: ignore

    ds = create_ephys_data_set(nwb_file=nwb_path)
    cell_features, _, _, _ = extract_data_set_features(ds)
    ls = cell_features.get("long_squares", {})
    return {"rin": ls.get("input_resistance"), "sag": ls.get("sag"), "tau": ls.get("tau"),
            "rheobase": ls.get("rheobase_i"), "fi_slope": ls.get("fi_fit_slope")}


# ---------------------------------------------------------------------- simulator (tests / nulls)
def simulate_sweep(amp_pA: float, r_in_MOhm: float = 150.0, tau_ms: float = 20.0, sag_frac: float = 0.2,
                   tau_sag_ms: float = 80.0, v_rest: float = -70.0, v_thresh: float = -45.0,
                   v_reset: float = -60.0, ap_width_ms: float = 1.0, adapt_pA_per_spike: float = 0.0,
                   tau_adapt_ms: float = 100.0, dt_ms: float = 0.05, pre_ms: float = 100.0,
                   stim_ms: float = 1000.0, post_ms: float = 100.0, seed: Optional[int] = None) -> Sweep:
    """LIF neuron with a slow sag conductance and spike-triggered adaptation; returns a Sweep.

    Passive response: ``dv/dt = (-(v - v_rest) + R (I - I_sag - I_adapt)) / tau`` with a slow sag
    current that cancels ``sag_frac`` of the step. Spikes are drawn as triangular waveforms of
    ``ap_width_ms`` at half height so width extraction is well defined.
    """
    rng = np.random.default_rng(seed)
    n = int((pre_ms + stim_ms + post_ms) / dt_ms)
    t = np.arange(n) * dt_ms * 1e-3
    i = np.zeros(n)
    s, e = int(pre_ms / dt_ms), int((pre_ms + stim_ms) / dt_ms)
    i[s:e] = amp_pA
    v = np.full(n, v_rest)
    R = r_in_MOhm * 1e-3  # mV / pA
    i_sag = 0.0
    i_adapt = 0.0
    spike_len = max(2, int(2 * ap_width_ms / dt_ms))
    k = 1
    while k < n:
        i_sag += dt_ms / tau_sag_ms * (sag_frac * i[k] - i_sag)
        i_adapt += -dt_ms / tau_adapt_ms * i_adapt
        dv = (-(v[k - 1] - v_rest) + R * (i[k] - i_sag - i_adapt)) / tau_ms
        v[k] = v[k - 1] + dt_ms * dv + rng.normal(0, 0.02)
        if v[k] >= v_thresh and i[k] > 0:
            # triangular AP: rise to +30 mV then fall to reset
            half = spike_len // 2
            up = np.linspace(v_thresh, 30.0, half)
            down = np.linspace(30.0, v_reset, spike_len - half)
            wave = np.concatenate([up, down])
            end = min(n, k + spike_len)
            v[k:end] = wave[: end - k]
            i_adapt += adapt_pA_per_spike
            k = end
            continue
        k += 1
    return Sweep(t=t, v=v, i=i)


__all__ = ["Sweep", "detect_spikes", "ap_width_half_height", "adaptation_index", "sag_ratio", "input_resistance_MOhm",
           "membrane_time_constant_ms", "extract_long_square_features", "extract_with_ipfx", "simulate_sweep"]
