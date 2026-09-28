"""Myelinated axon along an arbitrary 3-D path: CRRSS/Sweeney node kinetics, McNeal-type internodes.

Model
-----
Nodes of Ranvier are placed every ``internode_ratio x D`` (default 100 x fibre diameter) along
the resampled path; the internode is a perfect insulator (McNeal, 1976), so only nodes carry
membrane current and consecutive nodes are coupled by the axoplasmic conductance

    G_a = pi (d/2)^2 / (rho_a L_int)         d = 0.7 D (axon diameter), rho_a = 54.7 ohm cm

Node membrane (Sweeney, Mortimer & Durand, 1987; the CRRSS mammalian model at 37 C):

    C dV/dt = -A [ g_Na m^2 h (V - E_Na) + g_L (V - E_L) ] + sum_j G_a ((V_j - V_i) + (Ve_j - Ve_i))

with V the transmembrane potential and Ve the extracellular potential at the node; the last
term is the discrete activating current. Time stepping is implicit Euler for the linear part
with conductances frozen from the previous step and Rush-Larsen updates of the gates, which is
unconditionally stable and accurate enough for threshold estimation at dt = 5 us.

Units: potentials mV, time ms, lengths cm internally (paths are given in mm, diameters in um),
capacitance uF, conductance mS, current uA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

V_REST = -80.0  # mV


@dataclass(frozen=True)
class CRRSSParams:
    g_na: float = 1445.0      # mS/cm^2
    g_l: float = 128.0        # mS/cm^2
    e_na: float = 35.64       # mV
    e_l: float = -80.01       # mV
    c_m: float = 2.5          # uF/cm^2
    rho_a: float = 54.7       # ohm cm
    node_length_um: float = 1.5
    internode_ratio: float = 100.0   # internode length = ratio * fibre diameter
    axon_diam_ratio: float = 0.7     # axon (node) diameter / fibre diameter


def _rates(v_rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CRRSS rate constants (1/ms) as functions of V - V_rest (mV)."""
    v = np.clip(v_rel, -200.0, 400.0)
    am = (97.0 + 0.363 * v) / (1.0 + np.exp((31.0 - v) / 5.3))
    bm = am / np.exp((v - 23.8) / 4.17)
    bh = 15.6 / (1.0 + np.exp((24.0 - v) / 10.0))
    ah = bh / np.exp((v - 5.5) / 5.0)
    return am, bm, ah, bh


@dataclass
class Pulse:
    """Stimulus time course: ``shape`` in {"cathodic", "anodic", "biphasic"}; ``width_ms`` per phase."""

    width_ms: float = 0.06
    shape: str = "cathodic"
    delay_ms: float = 0.02
    interphase_ms: float = 0.0

    def waveform(self, t: np.ndarray) -> np.ndarray:
        """Multiplier of the unit-amplitude field at times ``t`` (ms); cathodic phases are negative."""
        w = np.zeros_like(t, dtype=float)
        p1 = (t >= self.delay_ms) & (t < self.delay_ms + self.width_ms)
        if self.shape == "cathodic":
            w[p1] = -1.0
        elif self.shape == "anodic":
            w[p1] = 1.0
        elif self.shape == "biphasic":
            w[p1] = -1.0
            t2 = self.delay_ms + self.width_ms + self.interphase_ms
            w[(t >= t2) & (t < t2 + self.width_ms)] = 1.0
        else:
            raise ValueError(self.shape)
        return w

    @property
    def duration_ms(self) -> float:
        return self.delay_ms + 2 * self.width_ms + self.interphase_ms


def resample_path(points_mm: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Equally spaced points (mm) along a polyline, including both ends (at least 3 points)."""
    p = np.asarray(points_mm, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(3, int(np.floor(total / spacing_mm)) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(targets, s, p[:, k]) for k in range(3)])
    return out


class MyelinatedAxon:
    """CRRSS axon with nodes at ``nodes_mm`` (n x 3) connected by ``parent`` (tree; -1 = root).

    For an unbranched path ``parent = [-1, 0, 1, ...]``. Branched trees are supported by the
    same solver (the axial matrix is a graph Laplacian).
    """

    def __init__(self, nodes_mm: np.ndarray, fibre_diam_um: float, parent: Optional[Sequence[int]] = None,
                 params: CRRSSParams = CRRSSParams()) -> None:
        self.nodes = np.asarray(nodes_mm, dtype=float)
        self.n = len(self.nodes)
        self.D = float(fibre_diam_um)
        self.p = params
        self.parent = np.arange(-1, self.n - 1) if parent is None else np.asarray(parent, dtype=int)
        d_cm = self.p.axon_diam_ratio * self.D * 1e-4
        L_node = self.p.node_length_um * 1e-4
        area = np.pi * d_cm * L_node                                # cm^2 (identical nodes)
        self.area = np.full(self.n, area)
        self.C = self.p.c_m * self.area                              # uF per node
        # axial conductances (mS) between node i and parent
        rows, cols, vals = [], [], []
        for i in range(self.n):
            j = self.parent[i]
            if j < 0:
                continue
            L_int = np.linalg.norm(self.nodes[i] - self.nodes[j]) * 0.1  # mm -> cm
            L_int = max(L_int, 1e-4)
            g = np.pi * (d_cm / 2) ** 2 / (self.p.rho_a * L_int) * 1e3   # S -> mS
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [g, g, -g, -g]
        self.L = csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        self._vrest = np.full(self.n, V_REST)

    # ------------------------------------------------------------------ dynamics
    def simulate(self, ve_unit_mV: np.ndarray, amplitude: float, pulse: Pulse, dt: float = 0.005,
                 t_end: Optional[float] = None, record: bool = False) -> Dict[str, object]:
        """Integrate the response to ``amplitude * waveform(t) * ve_unit_mV``.

        Returns ``{"spiked": bool, "v_max": per-node max V, "detect_node": int, "t": ..., "v": ...}``.
        Spike detection: V > -20 mV at the node farthest (in index) from the node of maximal
        activating function, i.e. a propagated action potential rather than local depolarisation.
        """
        ve = np.asarray(ve_unit_mV, dtype=float)
        t_end = t_end if t_end is not None else pulse.duration_ms + 1.5
        steps = int(np.ceil(t_end / dt))
        v = self._vrest.copy()
        am, bm, ah, bh = _rates(v - V_REST)
        m = am / (am + bm)
        h = ah / (ah + bh)
        af = activating_function(ve, self.L)
        i_max = int(np.argmax(-af * (1 if pulse.shape != "anodic" else -1)))
        detect = 0 if i_max > self.n // 2 else self.n - 1
        v_max = v.copy()
        trace = [] if record else None
        C_dt = self.C / dt
        wave = pulse.waveform(np.arange(steps) * dt)
        for k in range(steps):
            am, bm, ah, bh = _rates(v - V_REST)
            m = m + (am / (am + bm) - m) * (1 - np.exp(-dt * (am + bm)))
            h = h + (ah / (ah + bh) - h) * (1 - np.exp(-dt * (ah + bh)))
            g_na = self.p.g_na * m * m * h * self.area           # mS
            g_l = self.p.g_l * self.area
            g_tot = g_na + g_l
            i_rev = g_na * self.p.e_na + g_l * self.p.e_l          # uA
            ve_now = ve * amplitude * wave[k]
            rhs = C_dt * v + i_rev - self.L @ ve_now
            A = self.L + diags(C_dt + g_tot, 0, format="csc")
            v = splu(A).solve(rhs)
            v_max = np.maximum(v_max, v)
            if record:
                trace.append(v.copy())
        out: Dict[str, object] = {"spiked": bool(v_max[detect] > -20.0), "v_max": v_max, "detect_node": detect}
        if record:
            out["t"] = np.arange(steps) * dt
            out["v"] = np.array(trace)
        return out


def activating_function(ve_mV: np.ndarray, L: csc_matrix) -> np.ndarray:
    """Discrete activating current -L @ Ve (uA); positive values depolarise."""
    return -np.asarray(L @ np.asarray(ve_mV, dtype=float)).ravel()


def threshold_current(axon: MyelinatedAxon, ve_unit_mV: np.ndarray, pulse: Pulse = Pulse(), lo: float = 0.0,
                      hi: float = 20.0, rel_tol: float = 0.02, max_iter: int = 30, dt: float = 0.005,
                      start: float = 0.05, growth: float = 2.0) -> float:
    """Threshold amplitude (same unit as the field's amplitude, e.g. mA).

    Because excitation is not monotonic at very high amplitudes (anodal surround / depolarisation
    block), the bracket is found by scanning upward geometrically from ``start`` to the first
    spiking amplitude (or ``hi``) and bisection then refines it. Returns ``inf`` if nothing up to
    ``hi`` excites the axon.
    """
    amp, prev, found = min(start, hi), lo, False
    while amp <= hi:
        if axon.simulate(ve_unit_mV, amp, pulse, dt=dt)["spiked"]:
            found = True
            break
        prev, amp = amp, amp * growth
    if not found:
        if amp / growth < hi and axon.simulate(ve_unit_mV, hi, pulse, dt=dt)["spiked"]:
            prev, amp = amp / growth, hi
        else:
            return float("inf")
    lo, hi = prev, amp
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if axon.simulate(ve_unit_mV, mid, pulse, dt=dt)["spiked"]:
            hi = mid
        else:
            lo = mid
        if hi - lo <= rel_tol * hi:
            break
    return float(hi)


def build_axon_on_path(points_mm: np.ndarray, fibre_diam_um: float, params: CRRSSParams = CRRSSParams()) -> MyelinatedAxon:
    """Resample a path at the internode spacing for this diameter and build the axon."""
    spacing_mm = params.internode_ratio * fibre_diam_um * 1e-3
    nodes = resample_path(points_mm, spacing_mm)
    return MyelinatedAxon(nodes, fibre_diam_um, params=params)


def af_statistics(ve_unit_mV: np.ndarray, axon: MyelinatedAxon, branch_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Summary statistics of the activating function used by the threshold surrogate (H5)."""
    af = activating_function(ve_unit_mV, axon.L)
    pos = af > 0
    peaks = np.where((af[1:-1] > af[:-2]) & (af[1:-1] > af[2:]) & (af[1:-1] > 0))[0] + 1
    out = {"af_max": float(af.max()), "af_min": float(af.min()), "n_pos_nodes": int(pos.sum()),
           "n_local_max": int(len(peaks)), "af_max_over_C": float(af.max() / axon.C.max())}
    if branch_mask is not None and branch_mask.any():
        i_max = int(np.argmax(af))
        d = np.linalg.norm(axon.nodes[branch_mask] - axon.nodes[i_max], axis=1)
        out["dist_max_to_branch_mm"] = float(d.min())
    return out
