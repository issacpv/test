"""Build a NEURON model from an SWC somatodendritic tree with a synthetic hillock / AIS / axon.

NEURON is imported lazily so that the rest of the package (and the tests) work
without it. The geometry of the appended axon follows common practice in AIS
modelling studies: a tapering hillock, an AIS of adjustable start and length,
then a myelinated axon with nodes. Channel mechanisms come from ModelDB
downloads (see data/README.md) and are inserted by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .swc import APICAL, AXON, BASAL, SOMA, Tree


@dataclass
class AxonSpec:
    hillock_len_um: float = 10.0
    hillock_diam_start_um: float = 3.0
    ais_start_um: float = 10.0        # distance from soma edge to AIS start (hillock length adjusts)
    ais_len_um: float = 40.0
    ais_diam_um: float = 1.0
    n_myelin: int = 5
    myelin_len_um: float = 100.0
    node_len_um: float = 1.0
    axon_diam_um: float = 0.8


@dataclass
class ChannelSpec:
    """Mechanism names and densities per compartment class (from a ModelDB entry)."""

    soma: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dend: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ais: Dict[str, Dict[str, float]] = field(default_factory=dict)
    node: Dict[str, Dict[str, float]] = field(default_factory=dict)
    myelin: Dict[str, Dict[str, float]] = field(default_factory=dict)
    passive: Dict[str, float] = field(default_factory=lambda: {"g_pas": 1.0 / 20000.0, "e_pas": -70.0,
                                                               "cm": 1.0, "Ra": 150.0})


def _require_neuron():
    try:
        from neuron import h  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without NEURON
        raise ImportError("NEURON (pip install neuron) is required for simulations; the passive/theory "
                          "modules do not need it") from exc
    h.load_file("stdrun.hoc")
    return h


def _insert(sec, mechs: Dict[str, Dict[str, float]], passive: Dict[str, float]) -> None:
    sec.insert("pas")
    for seg in sec:
        seg.pas.g = passive["g_pas"]
        seg.pas.e = passive["e_pas"]
    sec.cm = passive["cm"]
    sec.Ra = passive["Ra"]
    for name, params in mechs.items():
        sec.insert(name)
        for k, v in params.items():
            setattr(sec, f"{k}_{name}", v)


def build_model(tree: Tree, axon: AxonSpec = AxonSpec(), chans: ChannelSpec = ChannelSpec(),
                seg_len_um: float = 10.0):
    """Create NEURON sections from ``tree`` (soma + dendrites only) and append the synthetic axon.

    Returns a dict with ``soma``, ``dends``, ``hillock``, ``ais``, ``nodes``, ``myelin``
    section handles and the NEURON ``h`` object. Each SWC dendritic node becomes a
    point in a section that is split at branch points.
    """
    h = _require_neuron()
    children = tree.children()
    soma = h.Section(name="soma")
    r = tree.soma_radius()
    soma.L = soma.diam = 2 * r
    _insert(soma, chans.soma, chans.passive)

    dends: List = []
    # unbranched sections: walk from each soma child / dendrite root until a bifurcation
    def make_section(start: int, parent_sec, parent_x: float):
        sec = h.Section(name=f"dend_{len(dends)}")
        pts = [start]
        node = start
        while len(children[node]) == 1 and tree.types[children[node][0]] in (BASAL, APICAL):
            node = children[node][0]
            pts.append(node)
        h.pt3dclear(sec=sec)
        p0 = tree.parent[start]
        if p0 >= 0:
            x, y, z = tree.xyz[p0]
            h.pt3dadd(float(x), float(y), float(z), float(2 * tree.radius[start]), sec=sec)
        for i in pts:
            x, y, z = tree.xyz[i]
            h.pt3dadd(float(x), float(y), float(z), float(2 * tree.radius[i]), sec=sec)
        sec.nseg = max(1, int(sec.L / seg_len_um) | 1)
        sec.connect(parent_sec(parent_x))
        _insert(sec, chans.dend, chans.passive)
        dends.append(sec)
        for k in children[node]:
            if tree.types[k] in (BASAL, APICAL):
                make_section(k, sec, 1.0)

    for i in range(tree.n):
        p = tree.parent[i]
        if tree.types[i] in (BASAL, APICAL) and (p < 0 or tree.types[p] == SOMA):
            make_section(i, soma, 0.5)

    hillock = h.Section(name="hillock")
    hillock.L = max(axon.ais_start_um, 1.0)
    hillock.nseg = max(1, int(hillock.L / 2) | 1)
    h.pt3dclear(sec=hillock)
    h.pt3dadd(0, 0, 0, axon.hillock_diam_start_um, sec=hillock)
    h.pt3dadd(hillock.L, 0, 0, axon.ais_diam_um, sec=hillock)
    hillock.connect(soma(1.0))
    _insert(hillock, chans.soma, chans.passive)

    ais = h.Section(name="ais")
    ais.L, ais.diam = axon.ais_len_um, axon.ais_diam_um
    ais.nseg = max(3, int(ais.L / 2) | 1)
    ais.connect(hillock(1.0))
    _insert(ais, chans.ais, chans.passive)

    myelin, nodes = [], []
    prev = ais
    for k in range(axon.n_myelin):
        m = h.Section(name=f"myelin_{k}")
        m.L, m.diam, m.nseg = axon.myelin_len_um, axon.axon_diam_um, 5
        m.connect(prev(1.0))
        _insert(m, chans.myelin, {**chans.passive, "cm": chans.passive["cm"] / 50.0,
                                   "g_pas": chans.passive["g_pas"] / 50.0})
        n = h.Section(name=f"node_{k}")
        n.L, n.diam, n.nseg = axon.node_len_um, axon.axon_diam_um, 1
        n.connect(m(1.0))
        _insert(n, chans.node, chans.passive)
        myelin.append(m)
        nodes.append(n)
        prev = n
    return {"h": h, "soma": soma, "dends": dends, "hillock": hillock, "ais": ais, "myelin": myelin, "nodes": nodes}


def spike_features(t: np.ndarray, v_soma: np.ndarray, dvdt_threshold: float = 20.0) -> Dict[str, float]:
    """Threshold (mV, first crossing of dV/dt >= threshold), peak, amplitude and max dV/dt from a trace.

    Pure NumPy so it can be unit-tested without NEURON.
    """
    t = np.asarray(t, float)
    v = np.asarray(v_soma, float)
    dvdt = np.gradient(v, t)
    idx = np.where(dvdt >= dvdt_threshold)[0]
    if len(idx) == 0:
        return {"threshold_mv": float("nan"), "peak_mv": float(v.max()), "amplitude_mv": float("nan"),
                "max_dvdt": float(dvdt.max()), "spiked": 0.0}
    i0 = int(idx[0])
    i_peak = i0 + int(np.argmax(v[i0:]))
    return {"threshold_mv": float(v[i0]), "peak_mv": float(v[i_peak]), "amplitude_mv": float(v[i_peak] - v[i0]),
            "max_dvdt": float(dvdt[i0:i_peak + 1].max()) if i_peak >= i0 else float(dvdt.max()), "spiked": 1.0}


def run_step(model: Dict, amp_na: float, delay_ms: float = 50.0, dur_ms: float = 100.0, tstop_ms: float = 200.0,
             dt_ms: float = 0.025, v_init: float = -70.0) -> Dict[str, np.ndarray]:
    """Inject a somatic current step and return time and somatic voltage arrays (needs NEURON)."""
    h = model["h"]
    stim = h.IClamp(model["soma"](0.5))
    stim.delay, stim.dur, stim.amp = delay_ms, dur_ms, amp_na
    t_vec, v_vec = h.Vector(), h.Vector()
    t_vec.record(h._ref_t)
    v_vec.record(model["soma"](0.5)._ref_v)
    h.dt = dt_ms
    h.finitialize(v_init)
    h.continuerun(tstop_ms)
    return {"t": np.array(t_vec), "v": np.array(v_vec)}


def rheobase(model: Dict, lo_na: float = 0.0, hi_na: float = 2.0, tol_na: float = 0.005, **kw) -> Optional[float]:
    """Bisection for the smallest step current that evokes a somatic spike (needs NEURON)."""
    def spikes(amp: float) -> bool:
        tr = run_step(model, amp, **kw)
        return spike_features(tr["t"], tr["v"])["spiked"] > 0

    if not spikes(hi_na):
        return None
    while hi_na - lo_na > tol_na:
        mid = 0.5 * (lo_na + hi_na)
        if spikes(mid):
            hi_na = mid
        else:
            lo_na = mid
    return float(hi_na)
