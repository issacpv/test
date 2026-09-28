"""Extracellular forward model: probe geometries, line-source potentials, propagating-spike currents.

The line-source approximation (Holt & Koch, 1999; as used in LFPy, Hagen et al., 2018) gives
the potential at an electrode from a segment carrying uniform transmembrane current I over its
length L, in a homogeneous isotropic medium of conductivity sigma. The phenomenological current
template lets large factorial sweeps run without NEURON: a somatic Na/K spike current whose
dendritic return currents are delayed and attenuated with path distance. Full simulations of
the Allen all-active models (LFPy/BMTK) should replace it for the final results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

SIGMA_DEFAULT = 0.3  # S/m


# ------------------------------------------------------------------ probe geometries
@dataclass
class ProbeGeometry:
    name: str
    xyz: np.ndarray      # (n_sites, 3) in um; x across shank width, y along shank (depth), z normal to shank
    pitch_y: float


def neuropixels_1(n_rows: int = 40, y0: float = 0.0) -> ProbeGeometry:
    """Neuropixels 1.0: checkerboard, 4 columns (x = 11, 27, 43, 59 um), 20-um row pitch."""
    cols = np.array([11.0, 27.0, 43.0, 59.0])
    xs, ys = [], []
    for r in range(n_rows):
        y = y0 + r * 20.0
        for x in (cols[[0, 2]] if r % 2 == 0 else cols[[1, 3]]):
            xs.append(x)
            ys.append(y)
    xyz = np.column_stack([np.array(xs) - cols.mean(), np.array(ys), np.zeros(len(xs))])
    return ProbeGeometry("NP1.0", xyz, 20.0)


def neuropixels_2(n_rows: int = 50, y0: float = 0.0) -> ProbeGeometry:
    """Neuropixels 2.0 (one shank): 2 columns 32 um apart, 15-um row pitch."""
    xs, ys = [], []
    for r in range(n_rows):
        for x in (-16.0, 16.0):
            xs.append(x)
            ys.append(y0 + r * 15.0)
    return ProbeGeometry("NP2.0", np.column_stack([xs, ys, np.zeros(len(xs))]), 15.0)


def neuropixels_ultra(n_rows: int = 48, y0: float = 0.0) -> ProbeGeometry:
    """Neuropixels Ultra: 8 columns, 6-um pitch in both directions (48 x 8 = 384 sites)."""
    xs, ys = [], []
    for r in range(n_rows):
        for c in range(8):
            xs.append((c - 3.5) * 6.0)
            ys.append(y0 + r * 6.0)
    return ProbeGeometry("NPUltra", np.column_stack([xs, ys, np.zeros(len(xs))]), 6.0)


PROBES = {"NP1.0": neuropixels_1, "NP2.0": neuropixels_2, "NPUltra": neuropixels_ultra}


# ------------------------------------------------------------------ line-source potentials
def line_source_matrix(starts: np.ndarray, ends: np.ndarray, electrodes: np.ndarray,
                       sigma: float = SIGMA_DEFAULT, min_dist: float = 1.0) -> np.ndarray:
    """Mapping matrix M (n_electrodes x n_segments): V = M @ I, with I in nA and V in uV.

    Uses the closed-form line-source expression: for a segment of length L along direction u,
    electrode at longitudinal offset h (from the segment end) and perpendicular distance r,
    phi = I / (4 pi sigma L) * ln( (sqrt(h^2 + r^2) - h) / (sqrt(l^2 + r^2) - l) ), l = h + L.
    Units: I [nA], sigma [S/m], lengths [um] -> phi [uV] with the 1e3 factor below.
    """
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    electrodes = np.asarray(electrodes, float)
    d = ends - starts
    L = np.linalg.norm(d, axis=1)
    L_safe = np.where(L < 1e-6, 1e-6, L)
    u = d / L_safe[:, None]
    # vector from segment end to electrode
    rel = electrodes[:, None, :] - ends[None, :, :]            # (E, S, 3)
    h = np.einsum("esk,sk->es", rel, u)                          # longitudinal offset from end
    perp = rel - h[..., None] * u[None, :, :]
    r2 = np.maximum(np.sum(perp ** 2, axis=-1), min_dist ** 2)
    l_ = h + L_safe[None, :]
    num = np.sqrt(h ** 2 + r2) - h
    den = np.sqrt(l_ ** 2 + r2) - l_
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.log(np.maximum(num, 1e-12) / np.maximum(den, 1e-12)) / (4 * np.pi * sigma * L_safe[None, :])
    # point-source fallback for degenerate (zero-length) segments
    point = 1.0 / (4 * np.pi * sigma * np.sqrt(np.sum((electrodes[:, None, :] - 0.5 * (starts + ends)[None]) ** 2, -1).clip(min_dist ** 2)))
    M = np.where(L[None, :] < 1e-6, point, M)
    # nA / (S/m * um) = 1e-9 A / (S * 1e-6) = 1e-3 V = 1e3 uV
    return M * 1e3


def point_source_matrix(positions: np.ndarray, electrodes: np.ndarray, sigma: float = SIGMA_DEFAULT,
                        min_dist: float = 1.0) -> np.ndarray:
    d = np.sqrt(np.sum((np.asarray(electrodes)[:, None, :] - np.asarray(positions)[None]) ** 2, -1)).clip(min_dist)
    return 1e3 / (4 * np.pi * sigma * d)


# ------------------------------------------------------------------ current templates
def somatic_spike_current(t_ms: np.ndarray, t0: float = 1.0, na_width: float = 0.25, k_width: float = 0.6,
                          k_delay: float = 0.3, amplitude_na: float = -1.0, k_ratio: float = 0.6) -> np.ndarray:
    """Biphasic transmembrane current (nA): inward Na (negative) then outward K (positive)."""
    na = amplitude_na * np.exp(-0.5 * ((t_ms - t0) / na_width) ** 2)
    k = -amplitude_na * k_ratio * np.exp(-0.5 * ((t_ms - t0 - k_delay) / k_width) ** 2)
    return na + k


def propagating_spike_currents(
    starts: np.ndarray, ends: np.ndarray, radii: np.ndarray, types: np.ndarray, path_dist: np.ndarray,
    t_ms: np.ndarray, soma_current_nA: float = -2.0, dend_velocity_um_per_ms: float = 300.0,
    dend_space_constant_um: float = 150.0, axon_velocity_um_per_ms: float = 500.0, axon_gain: float = 0.5,
    dend_gain: float = 0.3, soma_type: int = 1, axon_type: int = 2,
) -> np.ndarray:
    """Phenomenological compartment currents (n_segments x n_time, nA), charge-balanced.

    The soma carries the spike current template. Dendritic compartments carry a delayed
    (back-propagation with finite velocity), attenuated (exponential in path distance) copy
    scaled by membrane area, representing capacitive/return currents; the axon carries a
    forward-propagating, less attenuated copy. Return currents are then adjusted so the total
    membrane current sums to zero at each time point (current conservation), which is what
    makes distant sites see a dipole-like field.
    """
    n = len(starts)
    length = np.linalg.norm(ends - starts, axis=1)
    area = 2 * np.pi * radii * length
    base = somatic_spike_current(t_ms)
    I = np.zeros((n, len(t_ms)))
    dt = t_ms[1] - t_ms[0]
    soma_mask = types == soma_type
    axon_mask = types == axon_type
    dend_mask = ~(soma_mask | axon_mask)
    if soma_mask.any():
        w = area[soma_mask] / area[soma_mask].sum()
        I[soma_mask] = soma_current_nA * np.outer(w, base)
    else:
        I[0] = soma_current_nA * base
    for mask, vel, gain in ((dend_mask, dend_velocity_um_per_ms, dend_gain), (axon_mask, axon_velocity_um_per_ms, axon_gain)):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        delay_bins = np.round(path_dist[idx] / vel / dt).astype(int)
        atten = np.exp(-path_dist[idx] / dend_space_constant_um)
        amp = soma_current_nA * gain * atten * area[idx] / max(area[soma_mask].sum() if soma_mask.any() else 1.0, 1e-9)
        for k, i in enumerate(idx):
            shifted = np.roll(base, delay_bins[k])
            if delay_bins[k] > 0:
                shifted[: delay_bins[k]] = 0.0
            I[i] = amp[k] * shifted
    # current conservation: distribute the residual over non-soma compartments weighted by area
    resid = I.sum(0)
    non_soma = np.nonzero(~soma_mask)[0]
    if len(non_soma):
        w = area[non_soma] / area[non_soma].sum()
        I[non_soma] -= np.outer(w, resid)
    else:
        I -= resid / n
    return I


def simulate_eap(
    starts: np.ndarray, ends: np.ndarray, radii: np.ndarray, types: np.ndarray, path_dist: np.ndarray,
    probe: ProbeGeometry, soma_offset: Tuple[float, float, float] = (0.0, 0.0, 30.0),
    t_ms: Optional[np.ndarray] = None, sigma: float = SIGMA_DEFAULT, currents: Optional[np.ndarray] = None,
    **current_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatiotemporal EAP (n_sites x n_time, uV) for a morphology placed at ``soma_offset`` from the probe.

    ``currents`` (n_segments x n_time, nA) may be supplied from a NEURON/LFPy simulation;
    otherwise the phenomenological template is used.
    """
    if t_ms is None:
        t_ms = np.arange(0, 4.0, 0.02)
    off = np.asarray(soma_offset, float)
    s = np.asarray(starts) + off
    e = np.asarray(ends) + off
    M = line_source_matrix(s, e, probe.xyz, sigma=sigma)
    if currents is None:
        currents = propagating_spike_currents(s, e, radii, types, path_dist, t_ms, **current_kwargs)
    return M @ currents, t_ms


def placement_grid(distances=(20.0, 40.0, 60.0, 80.0), depth_offsets=(0.0, 10.0), lateral=(0.0,)) -> np.ndarray:
    """Grid of soma offsets (x lateral, y depth, z distance from the shank plane)."""
    return np.array([(x, y, z) for z in distances for y in depth_offsets for x in lateral])
