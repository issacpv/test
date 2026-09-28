"""Passive somatodendritic load seen by the AIS.

The AIS spike must (1) overcome the steady-state leak of the tree (input
conductance) and (2) charge the tree's capacitance on a ~0.2-1 ms time scale
(effective capacitance at 500-2000 Hz). Both come from the complex admittance
solution ``(G + i w C) V = I`` of the passive tree; the axon is dropped and
replaced by a virtual AIS in :mod:`ais_theory`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve

from .swc import APICAL, BASAL, SOMA, Tree


@dataclass
class Passive:
    rm_ohm_cm2: float = 20000.0
    ra_ohm_cm: float = 150.0
    cm_uf_cm2: float = 1.0


def assemble(tree: Tree, p: Passive = Passive(), keep_types: Sequence[int] = (SOMA, BASAL, APICAL)
             ) -> Tuple[csc_matrix, np.ndarray, int, np.ndarray]:
    """Conductance matrix G (S), membrane capacitance vector (F), soma index, kept row indices."""
    keep = np.where(np.isin(tree.types, keep_types))[0]
    remap = -np.ones(tree.n, dtype=int)
    remap[keep] = np.arange(len(keep))
    n = len(keep)
    seg = tree.seg_length()
    gm = np.zeros(n)
    cm = np.zeros(n)
    rows: list = []
    cols: list = []
    vals: list = []
    soma_nodes = keep[tree.types[keep] == SOMA]
    if len(soma_nodes):
        soma = int(remap[soma_nodes[0]])
        area = 4 * np.pi * tree.soma_radius() ** 2 * 1e-8  # cm^2
    else:
        soma = int(remap[keep[tree.parent[keep] < 0][0]])
        area = 4 * np.pi * float(tree.radius[keep[soma]]) ** 2 * 1e-8
    gm[soma] += area / p.rm_ohm_cm2
    cm[soma] += area * p.cm_uf_cm2 * 1e-6
    for i in keep:
        ii = remap[i]
        pi = tree.parent[i]
        if tree.types[i] == SOMA:
            if ii != soma:  # collapse extra soma nodes onto the soma node
                rows += [ii, soma, ii, soma]
                cols += [ii, soma, soma, ii]
                vals += [1e3, 1e3, -1e3, -1e3]
            continue
        if pi < 0 or remap[pi] < 0:
            continue
        pj = soma if tree.types[pi] == SOMA else remap[pi]
        l_um = max(float(seg[i]), 1e-3)
        d_um = 2 * float(tree.radius[i]) if tree.types[pi] == SOMA else float(tree.radius[i] + tree.radius[pi])
        a = np.pi * d_um * l_um * 1e-8
        gm[ii] += a / p.rm_ohm_cm2
        cm[ii] += a * p.cm_uf_cm2 * 1e-6
        g_ax = np.pi * (d_um * 1e-4) ** 2 / (4.0 * p.ra_ohm_cm * l_um * 1e-4)
        rows += [ii, pj, ii, pj]
        cols += [ii, pj, pj, ii]
        vals += [g_ax, g_ax, -g_ax, -g_ax]
    rows += list(range(n))
    cols += list(range(n))
    vals += list(gm)
    G = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return G, cm, soma, keep


def somatic_impedance(tree: Tree, freqs_hz: Sequence[float], p: Passive = Passive()) -> np.ndarray:
    """Complex somatic input impedance Z(f) in Ohm for each frequency."""
    G, cm, soma, _ = assemble(tree, p)
    rhs = np.zeros(G.shape[0], dtype=complex)
    rhs[soma] = 1.0
    out = np.empty(len(freqs_hz), dtype=complex)
    for k, f in enumerate(freqs_hz):
        Y = (G + 1j * 2 * np.pi * f * diags(cm)).tocsc()
        out[k] = spsolve(Y, rhs)[soma]
    return out


def load_metrics(tree: Tree, p: Passive = Passive(), freqs_hz: Sequence[float] = (500.0, 1000.0, 2000.0)
                 ) -> Dict[str, float]:
    """Summary of the passive load the AIS sees at the soma.

    Returns input conductance (nS), input resistance (MOhm), total membrane
    capacitance (pF), effective capacitance at each frequency (pF, from the
    imaginary part of the admittance: C_eff = Im(Y)/(2 pi f)), and the ratio
    of effective to total capacitance at the highest frequency (how much of the
    tree the AIS spike actually 'sees').
    """
    G, cm, soma, _ = assemble(tree, p)
    z = somatic_impedance(tree, (0.0, *freqs_hz), p)
    g_in = 1.0 / z[0].real
    out: Dict[str, float] = {
        "input_conductance_ns": float(g_in * 1e9),
        "input_resistance_mohm": float(1.0 / g_in * 1e-6),
        "total_capacitance_pf": float(cm.sum() * 1e12),
        "membrane_area_um2": float(cm.sum() / (p.cm_uf_cm2 * 1e-6) * 1e8),
    }
    for f, zf in zip(freqs_hz, z[1:]):
        y = 1.0 / zf
        out[f"c_eff_{int(f)}hz_pf"] = float(y.imag / (2 * np.pi * f) * 1e12)
        out[f"z_abs_{int(f)}hz_mohm"] = float(abs(zf) * 1e-6)
    f_hi = int(max(freqs_hz))
    out["c_eff_fraction"] = out[f"c_eff_{f_hi}hz_pf"] / out["total_capacitance_pf"]
    return out


def dendritic_load_index(metrics: Dict[str, float], f_hz: int = 1000) -> float:
    """Single scalar 'load' used by the theory: effective capacitance (pF) at ``f_hz``."""
    return float(metrics[f"c_eff_{f_hz}hz_pf"])
