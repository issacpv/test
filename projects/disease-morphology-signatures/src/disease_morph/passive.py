"""Passive compartmental model of a reconstruction (steady state and frequency domain).

Each SWC segment (node -> parent) is a cylinder of length l and diameter d
(average of the two node diameters). Membrane conductance g_m = A / R_m,
capacitance C = A C_m with A = pi d l, and axial conductance g_a = pi d^2 / (4 R_a l).
The soma is treated as an isopotential sphere whose surface area is taken from the
SWC soma radius. Solving (G + i w C) V = I gives the somatic input impedance and
the transfer ratio to every compartment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve

from .swc import DENDRITE_TYPES, SOMA, Morphology, segment_lengths


@dataclass
class PassiveParams:
    rm_ohm_cm2: float = 20000.0
    ra_ohm_cm: float = 150.0
    cm_uf_cm2: float = 1.0


def _um2_to_cm2(a: np.ndarray | float) -> np.ndarray | float:
    return a * 1e-8


def _um_to_cm(x: np.ndarray | float) -> np.ndarray | float:
    return x * 1e-4


def build_matrices(m: Morphology, p: PassiveParams = PassiveParams(),
                   types: Sequence[int] = (SOMA, *DENDRITE_TYPES)) -> Tuple[csc_matrix, np.ndarray, np.ndarray, int]:
    """Return (G, C, keep_index, soma_index): conductance matrix (S), membrane capacitance (F) per node.

    Nodes whose type is not in ``types`` are dropped (e.g. axon). The soma node
    is the first SOMA-typed node (or the root when no soma is present).
    """
    keep = np.where(np.isin(m.types, types))[0]
    remap = -np.ones(m.n, dtype=int)
    remap[keep] = np.arange(len(keep))
    n = len(keep)
    seg_len = segment_lengths(m)
    g_m = np.zeros(n)
    c_m = np.zeros(n)
    rows, cols, vals = [], [], []
    soma_nodes = keep[m.types[keep] == SOMA]
    soma_idx = int(remap[soma_nodes[0]]) if len(soma_nodes) else int(remap[keep[m.parent_index[keep] < 0][0]])
    # soma: sphere of the (max) soma radius; all soma nodes merged into one isopotential node
    if len(soma_nodes):
        r_soma = float(m.radius[soma_nodes].max())
        area_soma = _um2_to_cm2(4 * np.pi * r_soma ** 2)
        g_m[soma_idx] += area_soma / p.rm_ohm_cm2
        c_m[soma_idx] += area_soma * p.cm_uf_cm2 * 1e-6
    for i in keep:
        pi = m.parent_index[i]
        ii = remap[i]
        if m.types[i] == SOMA:
            if ii != soma_idx:  # merge extra soma nodes with a very large conductance
                rows += [ii, soma_idx, ii, soma_idx]
                cols += [ii, soma_idx, soma_idx, ii]
                vals += [1e3, 1e3, -1e3, -1e3]
            continue
        if pi < 0 or remap[pi] < 0:
            continue  # dendrite root without kept parent
        pj = remap[pi]
        if m.types[pi] == SOMA:
            pj = soma_idx
        l = max(float(seg_len[i]), 1e-3)
        d = float(m.radius[i] + m.radius[pi])
        if m.types[pi] == SOMA:
            d = 2 * float(m.radius[i])
        area = _um2_to_cm2(np.pi * d * l)
        g_m[ii] += area / p.rm_ohm_cm2
        c_m[ii] += area * p.cm_uf_cm2 * 1e-6
        g_ax = np.pi * (_um_to_cm(d) ** 2) / (4.0 * p.ra_ohm_cm * _um_to_cm(l))
        rows += [ii, pj, ii, pj]
        cols += [ii, pj, pj, ii]
        vals += [g_ax, g_ax, -g_ax, -g_ax]
    rows += list(range(n))
    cols += list(range(n))
    vals += list(g_m)
    G = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return G, c_m, keep, soma_idx


def input_impedance(m: Morphology, p: PassiveParams = PassiveParams(), freqs_hz: Sequence[float] = (0.0,),
                    types: Sequence[int] = (SOMA, *DENDRITE_TYPES)) -> Dict[str, object]:
    """Somatic input impedance |Z(f)| (Ohm) and the node transfer ratios at each frequency.

    Returns dict with ``freqs``, ``z_abs`` (Ohm), ``transfer`` (n_freq x n_nodes complex ratios
    V_node / V_soma), ``keep`` and ``soma_index``.
    """
    G, c_m, keep, soma_idx = build_matrices(m, p, types)
    n = G.shape[0]
    rhs = np.zeros(n, dtype=complex)
    rhs[soma_idx] = 1.0  # 1 A injected: V = Z
    z_abs, transfer = [], []
    for f in freqs_hz:
        Y = (G + 1j * 2 * np.pi * f * diags(c_m)).tocsc()
        v = spsolve(Y, rhs)
        z_abs.append(abs(v[soma_idx]))
        transfer.append(v / v[soma_idx])
    return {"freqs": np.asarray(freqs_hz, float), "z_abs": np.asarray(z_abs), "transfer": np.asarray(transfer),
            "keep": keep, "soma_index": soma_idx}


def electrotonic_summary(m: Morphology, p: PassiveParams = PassiveParams()) -> Dict[str, float]:
    """Steady-state input resistance (MOhm), mean/max soma->tip attenuation and mean electrotonic length.

    Attenuation is expressed as the log of the voltage ratio soma/tip for somatic
    injection (the reciprocal transfer, tip->soma, has the same ratio for a passive
    tree by reciprocity). Electrotonic length uses lambda = sqrt((d/4) Rm/Ra) per segment.
    """
    res = input_impedance(m, p, freqs_hz=(0.0,))
    keep, soma_idx = res["keep"], res["soma_index"]
    tr = np.real(res["transfer"][0])
    from .swc import DENDRITE_TYPES as DT

    children = m.children()
    tips_global = [i for i in keep if m.types[i] in DT and len(children[i]) == 0]
    remap = {g: k for k, g in enumerate(keep)}
    tip_ratio = np.array([tr[remap[i]] for i in tips_global]) if tips_global else np.array([1.0])
    tip_ratio = np.clip(tip_ratio, 1e-12, None)
    # electrotonic length per tip: sum over the path of l / lambda
    seg = segment_lengths(m)
    lam = np.full(m.n, np.inf)
    ok = (m.parent_index >= 0) & np.isin(m.types, DT)
    d_cm = _um_to_cm(2 * m.radius[ok])
    lam[ok] = np.sqrt((d_cm / 4.0) * p.rm_ohm_cm2 / p.ra_ohm_cm) * 1e4  # um
    el = np.zeros(m.n)
    stack = [i for i, pp in enumerate(m.parent_index) if pp < 0]
    while stack:
        i = stack.pop()
        for k in children[i]:
            el[k] = el[i] + (seg[k] / lam[k] if np.isfinite(lam[k]) and lam[k] > 0 else 0.0)
            stack.append(k)
    el_tips = el[tips_global] if tips_global else np.array([0.0])
    return {
        "input_resistance_mohm": float(res["z_abs"][0] * 1e-6),
        "mean_log_attenuation": float(np.mean(-np.log(tip_ratio))),
        "max_log_attenuation": float(np.max(-np.log(tip_ratio))),
        "mean_electrotonic_length": float(np.mean(el_tips)),
        "max_electrotonic_length": float(np.max(el_tips)),
        "n_tips": int(len(tips_global)),
    }


def ball_and_stick_input_resistance(r_soma_um: float, d_um: float, l_um: float,
                                    p: PassiveParams = PassiveParams()) -> float:
    """Analytic input resistance (MOhm) of a spherical soma plus one sealed-end cylinder (Rall)."""
    a_soma = _um2_to_cm2(4 * np.pi * r_soma_um ** 2)
    g_soma = a_soma / p.rm_ohm_cm2
    d = _um_to_cm(d_um)
    lam = np.sqrt((d / 4.0) * p.rm_ohm_cm2 / p.ra_ohm_cm)
    r_inf = (2.0 / np.pi) * np.sqrt(p.rm_ohm_cm2 * p.ra_ohm_cm) * d ** (-1.5)
    L = _um_to_cm(l_um) / lam
    g_cyl = (1.0 / r_inf) * np.tanh(L)  # sealed end
    return float(1.0 / (g_soma + g_cyl) * 1e-6)
