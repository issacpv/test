"""Single-compartment (linear R-C) respiratory mechanics.

Equation of motion for a passive patient: ``Paw(t) = PEEP + R * Flow(t) + V(t) / C``.

Units used throughout: pressures in cmH2O, volume in mL, flow in L/s,
resistance R in cmH2O/(L/s), compliance C in mL/cmH2O, elastance E = 1/C
in cmH2O/mL (E_L = E * 1000 in cmH2O/L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def driving_pressure(pplat: float, peep: float) -> float:
    """Driving pressure = plateau pressure - PEEP (cmH2O)."""
    return float(pplat - peep)


def static_compliance(vt_ml: float, pplat: float, peep: float) -> float:
    """Static compliance Crs = VT / (Pplat - PEEP) in mL/cmH2O."""
    dp = pplat - peep
    if dp <= 0:
        return float("nan")
    return float(vt_ml / dp)


def airway_resistance(ppeak: float, pplat: float, flow_l_s: float) -> float:
    """Inspiratory resistance R = (Ppeak - Pplat) / Flow in cmH2O/(L/s)."""
    if flow_l_s <= 0:
        return float("nan")
    return float((ppeak - pplat) / flow_l_s)


def mechanical_power(rr: float, vt_ml: float, ppeak: float, pplat: float, peep: float) -> float:
    """Mechanical power (J/min), simplified formula of Gattinoni et al. (2016):

    ``MP = 0.098 * RR * VT_L * (Ppeak - 0.5 * (Pplat - PEEP))``.
    """
    return float(0.098 * rr * (vt_ml / 1000.0) * (ppeak - 0.5 * (pplat - peep)))


def time_constant_s(r: float, c_ml: float) -> float:
    """Expiratory time constant tau = R * C (seconds), with C converted from mL to L."""
    return float(r * c_ml / 1000.0)


def ti_from_rr_ie(rr: float, ie_ratio: float) -> float:
    """Inspiratory time (s) from respiratory rate and I:E ratio expressed as I/E (e.g. 0.5 for 1:2)."""
    if rr <= 0 or ie_ratio <= 0:
        return float("nan")
    cycle = 60.0 / rr
    return float(cycle * ie_ratio / (1.0 + ie_ratio))


def inspiratory_flow_l_s(vt_ml: float, ti_s: float) -> float:
    """Mean inspiratory flow (L/s) for a square-flow breath: VT / Ti."""
    if ti_s <= 0:
        return float("nan")
    return float(vt_ml / 1000.0 / ti_s)


@dataclass(frozen=True)
class Breath:
    t: np.ndarray
    paw: np.ndarray
    flow: np.ndarray   # L/s, positive inspiratory
    volume: np.ndarray  # mL above FRC


def simulate_vcv_breath(
    r: float,
    c_ml: float,
    peep: float,
    vt_ml: float,
    flow_l_s: float,
    fs: float = 100.0,
    te_s: float | None = None,
    noise_cmh2o: float = 0.0,
    seed: int = 0,
) -> Breath:
    """Simulate one volume-controlled breath with square inspiratory flow and passive expiration.

    Inspiration lasts ``VT / flow``; expiration is exponential with tau = R*C
    until ``te_s`` (default 3 tau, at least 1 s).
    """
    rng = np.random.default_rng(seed)
    ti = vt_ml / 1000.0 / flow_l_s
    tau = time_constant_s(r, c_ml)
    te = te_s if te_s is not None else max(1.0, 3.0 * tau)
    t = np.arange(0, ti + te, 1.0 / fs)
    insp = t < ti
    flow = np.where(insp, flow_l_s, 0.0)
    volume = np.where(insp, flow_l_s * 1000.0 * t, 0.0)
    v_end = flow_l_s * 1000.0 * ti
    exp_t = t[~insp] - ti
    volume[~insp] = v_end * np.exp(-exp_t / tau)
    flow[~insp] = -(v_end / 1000.0 / tau) * np.exp(-exp_t / tau)
    paw = peep + r * np.clip(flow, 0, None) + volume / c_ml
    paw[~insp] = peep  # passive expiration against the PEEP valve
    paw = paw + noise_cmh2o * rng.normal(size=len(t))
    return Breath(t=t, paw=paw, flow=flow, volume=volume)


def fit_breath_least_squares(paw: np.ndarray, flow: np.ndarray, volume: np.ndarray, inspiratory_only: bool = True) -> dict[str, float]:
    """Multiple linear regression fit of ``Paw = E*V + R*Flow + P0`` on one breath.

    Returns R (cmH2O/(L/s)), C (mL/cmH2O), PEEP estimate P0 and the RMS residual.
    """
    paw = np.asarray(paw, float)
    flow = np.asarray(flow, float)
    volume = np.asarray(volume, float)
    mask = flow > 0 if inspiratory_only else np.ones(len(paw), bool)
    X = np.column_stack([volume[mask], flow[mask], np.ones(mask.sum())])
    coef, *_ = np.linalg.lstsq(X, paw[mask], rcond=None)
    e, r, p0 = coef
    resid = paw[mask] - X @ coef
    return {"R": float(r), "C": float(1.0 / e) if e > 0 else float("nan"), "E": float(e), "PEEP": float(p0), "rmse": float(np.sqrt(np.mean(resid**2)))}


def charted_point_estimates(ppeak: float, pplat: float, peep: float, vt_ml: float, flow_l_s: float, rr: float) -> dict[str, float]:
    """Algebraic estimates from one charted row (NaN-tolerant)."""
    out = {"dp": float("nan"), "C": float("nan"), "R": float("nan"), "MP": float("nan")}
    if np.isfinite(pplat) and np.isfinite(peep):
        out["dp"] = driving_pressure(pplat, peep)
        if np.isfinite(vt_ml):
            out["C"] = static_compliance(vt_ml, pplat, peep)
    if np.isfinite(ppeak) and np.isfinite(pplat) and np.isfinite(flow_l_s):
        out["R"] = airway_resistance(ppeak, pplat, flow_l_s)
    if all(np.isfinite(v) for v in (rr, vt_ml, ppeak, pplat, peep)):
        out["MP"] = mechanical_power(rr, vt_ml, ppeak, pplat, peep)
    return out
