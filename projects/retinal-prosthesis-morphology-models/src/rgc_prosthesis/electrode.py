"""Extracellular potentials of prosthesis electrodes and stimulus waveforms.

Disk electrode of radius ``a`` on the surface of a semi-infinite homogeneous medium (the
classic closed form, e.g. Wiley & Webster, 1982, IEEE TBME):

    V(r, z) = V0 * (2/pi) * arcsin( 2a / ( sqrt((r-a)^2 + z^2) + sqrt((r+a)^2 + z^2) ) )
    V0      = I / (4 sigma a)

with ``I`` the electrode current, ``sigma`` the conductivity and ``(r, z)`` cylindrical
coordinates about the disk axis. Units here: current uA, lengths um, sigma S/m, potential mV
(``V0_mV = 1e3 * I / (4 sigma a)`` with I in uA and a in um, because the 1e-6 factors cancel).
A point source gives ``V = 1e3 * I / (4 pi sigma R)`` in the same units.

Placement: an epiretinal electrode sits on the vitreal side at ``z_um < 0`` (retinal convention
of :mod:`rgc_prosthesis.swc_morph`); a subretinal electrode sits beyond the dendrites at
``z_um > 0``. Layered-retina or degenerate-tissue effects require a FEM potential; the
``potential_fn`` hook on :class:`Electrode` accepts any callable ``xyz -> mV per uA``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def point_source_potential(xyz_um: np.ndarray, centre_um: np.ndarray, current_uA: float, sigma: float = 0.3,
                           r_min_um: float = 1.0) -> np.ndarray:
    xyz = np.atleast_2d(np.asarray(xyz_um, dtype=float))
    d = np.linalg.norm(xyz - np.asarray(centre_um, dtype=float)[None, :], axis=1)
    return 1e3 * current_uA / (4 * np.pi * sigma * np.maximum(d, r_min_um))


def disk_electrode_potential(xyz_um: np.ndarray, centre_um: np.ndarray, radius_um: float, current_uA: float,
                             sigma: float = 0.3, axis: int = 2) -> np.ndarray:
    """Potential (mV) of a disk electrode (axis along ``axis``) in a semi-infinite medium."""
    xyz = np.atleast_2d(np.asarray(xyz_um, dtype=float))
    rel = xyz - np.asarray(centre_um, dtype=float)[None, :]
    z = np.abs(rel[:, axis])
    lateral = np.delete(rel, axis, axis=1)
    r = np.linalg.norm(lateral, axis=1)
    a = float(radius_um)
    denom = np.sqrt((r - a) ** 2 + z ** 2) + np.sqrt((r + a) ** 2 + z ** 2)
    v0 = 1e3 * current_uA / (4 * sigma * a)
    return v0 * (2 / np.pi) * np.arcsin(np.clip(2 * a / np.maximum(denom, 1e-9), -1, 1))


@dataclass
class Pulse:
    """Stimulus time course. ``shape``: cathodic, anodic, biphasic (cathodic-first), biphasic_anodic_first."""

    width_ms: float = 0.1
    shape: str = "biphasic"
    delay_ms: float = 0.05
    interphase_ms: float = 0.0

    def waveform(self, t: np.ndarray) -> np.ndarray:
        w = np.zeros_like(t, dtype=float)
        p1 = (t >= self.delay_ms) & (t < self.delay_ms + self.width_ms)
        t2 = self.delay_ms + self.width_ms + self.interphase_ms
        p2 = (t >= t2) & (t < t2 + self.width_ms)
        if self.shape == "cathodic":
            w[p1] = -1.0
        elif self.shape == "anodic":
            w[p1] = 1.0
        elif self.shape == "biphasic":
            w[p1], w[p2] = -1.0, 1.0
        elif self.shape == "biphasic_anodic_first":
            w[p1], w[p2] = 1.0, -1.0
        else:
            raise ValueError(self.shape)
        return w

    @property
    def duration_ms(self) -> float:
        return self.delay_ms + 2 * self.width_ms + self.interphase_ms


@dataclass
class Electrode:
    """An electrode with a unit-current potential function.

    Parameters
    ----------
    centre_um : (x, y, z) of the electrode centre (retinal convention: epiretinal z < 0).
    radius_um : disk radius (``geometry="disk"``) or ignored for ``"point"``.
    sigma : bulk conductivity (S/m); 0.3 S/m is a common homogeneous-retina value, 1-1.5 S/m for vitreous.
    potential_fn : optional custom ``xyz -> mV per uA`` (e.g. FEM interpolation) overriding geometry.
    """

    centre_um: np.ndarray
    radius_um: float = 50.0
    geometry: str = "disk"
    sigma: float = 0.3
    potential_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def unit_potential(self, xyz_um: np.ndarray) -> np.ndarray:
        """Potential (mV) per 1 uA of (positive) electrode current."""
        if self.potential_fn is not None:
            return np.asarray(self.potential_fn(xyz_um), dtype=float)
        if self.geometry == "disk":
            return disk_electrode_potential(xyz_um, self.centre_um, self.radius_um, 1.0, self.sigma)
        if self.geometry == "point":
            return point_source_potential(xyz_um, self.centre_um, 1.0, self.sigma)
        raise ValueError(self.geometry)

    @classmethod
    def epiretinal(cls, height_um: float = 30.0, radius_um: float = 50.0, xy_um=(0.0, 0.0), **kw) -> "Electrode":
        """Disk on the vitreal side, ``height_um`` above the ganglion-cell layer (negative z)."""
        return cls(np.array([xy_um[0], xy_um[1], -abs(height_um)]), radius_um, "disk", **kw)

    @classmethod
    def subretinal(cls, depth_um: float = 150.0, radius_um: float = 50.0, xy_um=(0.0, 0.0), **kw) -> "Electrode":
        """Disk beyond the outer retina, ``depth_um`` below the ganglion-cell layer (positive z)."""
        return cls(np.array([xy_um[0], xy_um[1], abs(depth_um)]), radius_um, "disk", **kw)
