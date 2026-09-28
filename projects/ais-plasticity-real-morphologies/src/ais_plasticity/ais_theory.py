"""Resistive-coupling theory of AIS excitability and a plasticity-demand solver.

Following the analysis of Brette (2013) and Goethals & Brette (2020), when the
soma is a large current sink relative to the AIS, the somatic voltage threshold
for spike initiation at a point-like AIS a distance ``delta`` from the soma is

    V_th = V_half - k_a * ln( r_a * delta * G_Na / k_a * (E_Na - V_half)/1 ) + const

i.e. it decreases logarithmically with the axial resistance R_a = r_a * delta
between soma and AIS and with the total AIS sodium conductance G_Na. The exact
constants depend on the Na activation curve; this module keeps the functional
form with explicit, documented parameters so that simulated thresholds can be
used to calibrate them. For an extended AIS of length L starting at delta, the
effective resistive distance is delta + L/2 in the simplest approximation, and
G_Na = g_Na * pi * d * L.

Nothing here replaces the NEURON simulations - the point of the project is to
test how far this theory carries across real morphologies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq


@dataclass
class AISParams:
    axon_diam_um: float = 1.0
    ra_ohm_cm: float = 150.0
    g_na_s_per_cm2: float = 0.4      # AIS Na peak conductance density (S/cm^2), order of published values
    v_half_mv: float = -30.0         # Na activation half-point (mV)
    k_a_mv: float = 6.0              # Na activation slope (mV)
    e_na_mv: float = 60.0
    offset_mv: float = 0.0           # calibration constant fitted to simulations


def axial_resistance_per_um(p: AISParams) -> float:
    """Axial resistance per micrometre of axon (Ohm/um)."""
    d_cm = p.axon_diam_um * 1e-4
    r_a_per_cm = p.ra_ohm_cm / (np.pi * (d_cm / 2) ** 2)  # Ohm/cm
    return float(r_a_per_cm * 1e-4)


def total_na_conductance(length_um: float, p: AISParams) -> float:
    """Total AIS Na conductance (S) for an AIS of the given length."""
    area_cm2 = np.pi * (p.axon_diam_um * 1e-4) * (length_um * 1e-4)
    return float(p.g_na_s_per_cm2 * area_cm2)


def threshold_mv(delta_um: float, length_um: float, p: AISParams = AISParams()) -> float:
    """Somatic voltage threshold (mV) from the resistive-coupling relation.

    Uses the effective distance delta + L/2 and total conductance g_Na * pi d L.
    Returns ``nan`` for non-positive geometry.
    """
    if delta_um <= 0 and length_um <= 0:
        return float("nan")
    eff = max(delta_um + 0.5 * length_um, 1e-3)
    r_a = axial_resistance_per_um(p) * eff                       # Ohm
    g_na = total_na_conductance(max(length_um, 1e-3), p)         # S
    drive = r_a * g_na * (p.e_na_mv - p.v_half_mv) / p.k_a_mv    # dimensionless
    return float(p.v_half_mv - p.k_a_mv * np.log(max(drive, 1e-12)) + p.offset_mv)


def calibrate_offset(delta_um: float, length_um: float, simulated_threshold_mv: float,
                     p: AISParams = AISParams()) -> AISParams:
    """Return a copy of ``p`` whose offset makes the theory match one simulated threshold."""
    base = threshold_mv(delta_um, length_um, AISParams(**{**p.__dict__, "offset_mv": 0.0}))
    return AISParams(**{**p.__dict__, "offset_mv": float(simulated_threshold_mv - base)})


def somatic_ap_amplitude_proxy(delta_um: float, length_um: float, c_eff_pf: float, g_in_ns: float,
                               p: AISParams = AISParams(), t_ais_ms: float = 0.3) -> float:
    """Crude proxy for somatic AP amplitude (mV above rest): the depolarisation the AIS can impose.

    The AIS at potential ~E_Na drives current (E_Na - V_s)/R_a into a soma with
    conductance g_in and effective capacitance c_eff for roughly ``t_ais_ms``. The
    quasi-static solution gives amplitude = (E_Na - V_rest) * g_a / (g_a + g_in + c_eff/t),
    with g_a = 1/R_a. This captures the qualitative dependence on load that
    Hamada et al. (2016) describe (larger trees -> smaller somatic AP for the same AIS).
    """
    v_rest_mv = -70.0
    eff = max(delta_um + 0.5 * length_um, 1e-3)
    g_a = 1.0 / (axial_resistance_per_um(p) * eff)
    g_in = g_in_ns * 1e-9
    g_c = c_eff_pf * 1e-12 / (t_ais_ms * 1e-3)
    return float((p.e_na_mv - v_rest_mv) * g_a / (g_a + g_in + g_c))


def demand_distance(target_threshold_mv: float, length_um: float, p: AISParams = AISParams(),
                    lo_um: float = 0.5, hi_um: float = 200.0) -> Optional[float]:
    """AIS distance (um) at which the theory reaches ``target_threshold_mv``; None if unreachable."""
    f = lambda d: threshold_mv(d, length_um, p) - target_threshold_mv  # noqa: E731
    try:
        if f(lo_um) * f(hi_um) > 0:
            return None
        return float(brentq(f, lo_um, hi_um))
    except ValueError:
        return None


def demand_length(target_threshold_mv: float, delta_um: float, p: AISParams = AISParams(),
                  lo_um: float = 1.0, hi_um: float = 150.0) -> Optional[float]:
    """AIS length (um) at which the theory reaches ``target_threshold_mv``; None if unreachable."""
    f = lambda L: threshold_mv(delta_um, L, p) - target_threshold_mv  # noqa: E731
    try:
        if f(lo_um) * f(hi_um) > 0:
            return None
        return float(brentq(f, lo_um, hi_um))
    except ValueError:
        return None


def amplitude_demand_distance(target_amp_mv: float, length_um: float, c_eff_pf: float, g_in_ns: float,
                              p: AISParams = AISParams(), lo_um: float = 0.5, hi_um: float = 200.0
                              ) -> Optional[float]:
    """AIS distance at which the amplitude proxy equals ``target_amp_mv`` (None if unreachable).

    Because the proxy *decreases* with distance while the threshold *decreases*
    too, the two set points pull the AIS in opposite directions; the pair
    (threshold demand, amplitude demand) is the morphology's plasticity demand.
    """
    f = lambda d: somatic_ap_amplitude_proxy(d, length_um, c_eff_pf, g_in_ns, p) - target_amp_mv  # noqa: E731
    try:
        if f(lo_um) * f(hi_um) > 0:
            return None
        return float(brentq(f, lo_um, hi_um))
    except ValueError:
        return None


def plasticity_demand(load: Dict[str, float], reference: Dict[str, float], p: AISParams = AISParams(),
                      ref_delta_um: float = 30.0, ref_length_um: float = 40.0) -> Dict[str, Optional[float]]:
    """Distance/length changes needed for a morphology with ``load`` to match a reference cell's set points.

    ``load`` and ``reference`` are :func:`ais_plasticity.dendritic_load.load_metrics`
    dictionaries. The reference cell with the reference AIS geometry defines the
    set points (threshold from the theory, amplitude from the proxy); the solver
    returns the geometry the new morphology needs and the demanded change.
    """
    c_ref, g_ref = reference["c_eff_1000hz_pf"], reference["input_conductance_ns"]
    c_new, g_new = load["c_eff_1000hz_pf"], load["input_conductance_ns"]
    target_amp = somatic_ap_amplitude_proxy(ref_delta_um, ref_length_um, c_ref, g_ref, p)
    target_thr = threshold_mv(ref_delta_um, ref_length_um, p)
    d_amp = amplitude_demand_distance(target_amp, ref_length_um, c_new, g_new, p)
    l_thr = demand_length(target_thr, d_amp, p) if d_amp is not None else None
    return {
        "target_threshold_mv": target_thr,
        "target_amplitude_mv": target_amp,
        "distance_for_amplitude_um": d_amp,
        "length_for_threshold_um": l_thr,
        "delta_distance_um": (d_amp - ref_delta_um) if d_amp is not None else None,
        "delta_length_um": (l_thr - ref_length_um) if l_thr is not None else None,
        "load_ratio": c_new / c_ref if c_ref > 0 else float("nan"),
    }
