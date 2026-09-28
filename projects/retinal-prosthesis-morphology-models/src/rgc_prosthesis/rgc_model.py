"""Multicompartment RGC with Fohlmeister-Miller-type Na/K/leak kinetics and extracellular stimulation.

Kinetics (Fohlmeister & Miller, 1997, J Neurophysiol; V in mV, rates in 1/ms):

    alpha_m = -0.6 (V+30) / (exp(-0.1 (V+30)) - 1)     beta_m = 20 exp(-(V+55)/18)
    alpha_h =  0.4 exp(-(V+50)/20)                      beta_h = 6 / (1 + exp(-0.1 (V+20)))
    alpha_n = -0.02 (V+40) / (exp(-0.1 (V+40)) - 1)    beta_n = 0.4 exp(-(V+50)/80)

    I_ion = g_Na m^3 h (V - E_Na) + g_K n^4 (V - E_K) + g_L (V - E_L)

The five-channel FM model (with Ca, K_Ca and A currents) is reserved for the NEURON calibration
subset; the starter keeps Na/K/leak with region-specific densities (dendrite, soma, hillock,
AIS/sodium-channel band, distal axon). For a stable resting state without the Ca-dependent
currents, the leak reversal is balanced per compartment so that the total ionic current is zero
at ``v_rest`` (a standard "e_pas" fit), and ``equilibrate`` verifies the drift.

Numerics: implicit Euler for the linear cable part with conductances frozen from the previous
step and Rush-Larsen gate updates; extracellular potentials enter through the discrete
activating current ``-L Ve``. Units: mV, ms, uF, mS, uA; geometry from :class:`Compartments`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from .electrode import Pulse
from .swc_morph import AIS, AXON, BASAL, APICAL, HILLOCK, SOMA, Compartments


@dataclass(frozen=True)
class RGCParams:
    c_m: float = 1.0          # uF/cm^2
    r_a: float = 110.0        # ohm cm
    e_na: float = 35.0
    e_k: float = -75.0
    v_rest: float = -65.0
    g_l: float = 0.15         # mS/cm^2 (raised from FM's 0.05 to compensate for the omitted K_Ca/A currents)
    # Na and K densities (mS/cm^2) per region
    g_na: Dict[int, float] = field(default_factory=lambda: {BASAL: 25.0, APICAL: 25.0, SOMA: 80.0, HILLOCK: 150.0,
                                                            AIS: 400.0, AXON: 100.0})
    g_k: Dict[int, float] = field(default_factory=lambda: {BASAL: 12.0, APICAL: 12.0, SOMA: 18.0, HILLOCK: 25.0,
                                                           AIS: 40.0, AXON: 25.0})


def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
    """x / (exp(x/y) - 1) with the removable singularity handled."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x / y) < 1e-6
    out = np.empty_like(x)
    out[~small] = x[~small] / (np.exp(x[~small] / y) - 1.0)
    out[small] = y * (1.0 - x[small] / (2 * y))
    return out


def fm_rates(v: np.ndarray) -> Tuple[np.ndarray, ...]:
    v = np.clip(np.asarray(v, dtype=float), -150.0, 150.0)
    am = 0.6 * _vtrap(-(v + 30.0), 10.0)
    bm = 20.0 * np.exp(-(v + 55.0) / 18.0)
    ah = 0.4 * np.exp(-(v + 50.0) / 20.0)
    bh = 6.0 / (1.0 + np.exp(-0.1 * (v + 20.0)))
    an = 0.02 * _vtrap(-(v + 40.0), 10.0)
    bn = 0.4 * np.exp(-(v + 50.0) / 80.0)
    return am, bm, ah, bh, an, bn


class RGCCable:
    """Compartmental RGC built from :class:`Compartments`."""

    def __init__(self, comp: Compartments, params: RGCParams = RGCParams()) -> None:
        self.comp = comp
        self.p = params
        self.n = comp.n
        area = comp.area_cm2
        self.area = area
        self.C = params.c_m * area                                    # uF
        self.g_na = np.array([params.g_na.get(int(t), 25.0) for t in comp.region]) * area   # mS
        self.g_k = np.array([params.g_k.get(int(t), 12.0) for t in comp.region]) * area
        self.g_l = params.g_l * area
        # axial conductances (mS): 1 / (Ra (L_i / (2 A_i) + L_j / (2 A_j))), A = pi (d/2)^2 in cm^2
        r_half = (comp.length * 1e-4) / (2 * np.pi * (comp.diam * 1e-4 / 2) ** 2) * params.r_a  # ohm
        rows, cols, vals = [], [], []
        for i in range(self.n):
            j = comp.parent[i]
            if j < 0:
                continue
            g = 1e3 / (r_half[i] + r_half[j])
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [g, g, -g, -g]
        self.L = csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        # balanced leak reversal so that the total ionic current vanishes at v_rest
        v0 = np.full(self.n, params.v_rest)
        am, bm, ah, bh, an, bn = fm_rates(v0)
        m, h, nn = am / (am + bm), ah / (ah + bh), an / (an + bn)
        i_na = self.g_na * m ** 3 * h * (v0 - params.e_na)
        i_k = self.g_k * nn ** 4 * (v0 - params.e_k)
        self.e_l = v0 + (i_na + i_k) / np.maximum(self.g_l, 1e-12)
        self._v0 = v0
        self._gates0 = (m, h, nn)
        # detection compartment: the distal end of the axon if present, else the soma
        ax = np.where(comp.region == AXON)[0]
        self.detect = int(ax[-1]) if len(ax) else 0
        self.soma_idx = int(np.where(comp.region == SOMA)[0][0]) if (comp.region == SOMA).any() else 0

    # ------------------------------------------------------------------ integration
    def _step_matrix(self, g_tot: np.ndarray, dt: float):
        return splu(self.L + diags(self.C / dt + g_tot, 0, format="csc"))

    def run(self, ve_unit_mV: Optional[np.ndarray], amplitude: float, pulse: Optional[Pulse], dt: float = 0.01,
            t_end: float = 3.0, record: bool = False, v_init: Optional[np.ndarray] = None,
            gates_init: Optional[Tuple[np.ndarray, ...]] = None) -> Dict[str, object]:
        """Integrate; ``ve_unit_mV`` is the extracellular potential per uA at each compartment."""
        steps = int(np.ceil(t_end / dt))
        v = self._v0.copy() if v_init is None else v_init.copy()
        m, h, nn = (self._gates0 if gates_init is None else gates_init)
        m, h, nn = m.copy(), h.copy(), nn.copy()
        wave = pulse.waveform(np.arange(steps) * dt) if pulse is not None else np.zeros(steps)
        ve = np.zeros(self.n) if ve_unit_mV is None else np.asarray(ve_unit_mV, dtype=float)
        C_dt = self.C / dt
        v_max = v.copy()
        trace = []
        for k in range(steps):
            am, bm, ah, bh, an, bn = fm_rates(v)
            m += (am / (am + bm) - m) * (1 - np.exp(-dt * (am + bm)))
            h += (ah / (ah + bh) - h) * (1 - np.exp(-dt * (ah + bh)))
            nn += (an / (an + bn) - nn) * (1 - np.exp(-dt * (an + bn)))
            gna = self.g_na * m ** 3 * h
            gk = self.g_k * nn ** 4
            g_tot = gna + gk + self.g_l
            i_rev = gna * self.p.e_na + gk * self.p.e_k + self.g_l * self.e_l
            rhs = C_dt * v + i_rev - self.L @ (ve * amplitude * wave[k])
            v = self._step_matrix(g_tot, dt).solve(rhs)
            v_max = np.maximum(v_max, v)
            if record:
                trace.append(v.copy())
        out: Dict[str, object] = {"spiked": bool(v_max[self.detect] > 0.0), "v_max": v_max, "v_end": v,
                                  "gates_end": (m, h, nn), "detect": self.detect,
                                  "soma_spiked": bool(v_max[self.soma_idx] > 0.0)}
        if record:
            out["t"] = np.arange(steps) * dt
            out["v"] = np.array(trace)
        return out

    def equilibrate(self, t_ms: float = 30.0, dt: float = 0.02) -> Dict[str, float]:
        """Run without stimulus from the analytic rest and adopt the end state; returns drift statistics."""
        res = self.run(None, 0.0, None, dt=dt, t_end=t_ms)
        v_end = res["v_end"]
        drift = float(np.max(np.abs(v_end - self.p.v_rest)))
        spont = bool(np.max(res["v_max"]) > 0.0)
        self._v0 = v_end
        self._gates0 = res["gates_end"]
        return {"max_drift_mV": drift, "spontaneous_spike": spont, "v_soma_mV": float(v_end[self.soma_idx])}


def threshold_current(cell: RGCCable, ve_unit_mV: np.ndarray, pulse: Pulse = Pulse(), lo: float = 0.0,
                      hi: float = 1000.0, rel_tol: float = 0.03, max_iter: int = 25, dt: float = 0.01,
                      t_end: Optional[float] = None) -> float:
    """Threshold electrode current (uA) by bisection on the propagated-spike criterion; inf if ``hi`` fails."""
    t_end = t_end if t_end is not None else pulse.duration_ms + 2.5
    if not cell.run(ve_unit_mV, hi, pulse, dt=dt, t_end=t_end)["spiked"]:
        return float("inf")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if cell.run(ve_unit_mV, mid, pulse, dt=dt, t_end=t_end)["spiked"]:
            hi = mid
        else:
            lo = mid
        if hi - lo <= rel_tol * hi:
            break
    return float(hi)


def activating_function(cell: RGCCable, ve_unit_mV: np.ndarray) -> np.ndarray:
    """Discrete activating current per uA (uA); positive depolarises."""
    return -np.asarray(cell.L @ np.asarray(ve_unit_mV, dtype=float)).ravel()


def first_spiking_region(cell: RGCCable, ve_unit_mV: np.ndarray, amplitude: float, pulse: Pulse,
                         dt: float = 0.01, t_end: Optional[float] = None) -> Dict[str, object]:
    """Region code and compartment index where V first crosses 0 mV (activation site)."""
    t_end = t_end if t_end is not None else pulse.duration_ms + 2.5
    res = cell.run(ve_unit_mV, amplitude, pulse, dt=dt, t_end=t_end, record=True)
    v = res["v"]
    above = v > 0.0
    if not above.any():
        return {"region": None, "index": None, "t_ms": None}
    k = int(np.argmax(above.any(axis=1)))
    i = int(np.argmax(v[k]))
    return {"region": int(cell.comp.region[i]), "index": i, "t_ms": float(res["t"][k])}
