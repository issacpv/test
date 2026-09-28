"""DBS lead geometries and extracellular potentials from discretised contact surfaces.

The starter model is a homogeneous, isotropic, infinite medium (default sigma = 0.2 S/m). Each
active contact is discretised into ``n_theta x n_z`` point sources on the cylindrical contact
surface that share the contact current equally. The potential of a point source of current
``I`` (mA) at distance ``r`` (mm) is

    phi(r) = 1e3 * I / (4 pi sigma r)      [mV]

Superposition over sources gives the lead field. Voltage-controlled stimulation is approximated
by scaling the unit-current field so that the *mean* potential on the active contact surface
equals the set voltage (an ideal equipotential contact would need a boundary-element solve; the
error of this approximation is a few percent for ring contacts and is one of the conductor-model
factors in the variance decomposition). An encapsulation layer is represented by an effective
conductivity scaling. FEM potentials exported from Lead-DBS / OSS-DBS can be used instead via
:class:`FieldFromNifti`.

Coordinates: lead axis = z (mm), lead tip at z = 0, contacts stacked upward; x, y radial.
Lead dimensions were transcribed from manufacturer documentation; verify against Lead-DBS
``templates/electrode_models`` before publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LeadSpec:
    """Geometry of a cylindrical DBS lead.

    Attributes
    ----------
    name : identifier.
    diameter_mm : lead body diameter.
    contact_length_mm : axial length of each contact level.
    spacing_mm : insulation gap between levels.
    tip_length_mm : distance from the tip to the lower edge of level 0.
    segmented_levels : which levels are split into ``n_segments`` arcs (others are rings).
    n_segments : number of arcs on a segmented level.
    segment_arc_deg : angular width of each arc.
    """

    name: str
    diameter_mm: float
    contact_length_mm: float
    spacing_mm: float
    tip_length_mm: float
    n_levels: int
    segmented_levels: Tuple[int, ...] = ()
    n_segments: int = 3
    segment_arc_deg: float = 90.0

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    def level_z(self, level: int) -> Tuple[float, float]:
        z0 = self.tip_length_mm + level * (self.contact_length_mm + self.spacing_mm)
        return z0, z0 + self.contact_length_mm

    @property
    def z_span(self) -> Tuple[float, float]:
        return self.level_z(0)[0], self.level_z(self.n_levels - 1)[1]


LEADS: Dict[str, LeadSpec] = {
    "medtronic_3389": LeadSpec("medtronic_3389", 1.27, 1.5, 0.5, 1.5, 4),
    "medtronic_3387": LeadSpec("medtronic_3387", 1.27, 1.5, 1.5, 1.5, 4),
    "bsci_vercise_cartesia": LeadSpec("bsci_vercise_cartesia", 1.3, 1.5, 0.5, 1.5, 4, segmented_levels=(1, 2)),
    "abbott_6172": LeadSpec("abbott_6172", 1.29, 1.5, 0.5, 1.5, 4, segmented_levels=(1, 2)),
}


def point_source_potential(xyz: np.ndarray, src: np.ndarray, currents_mA: np.ndarray, sigma: float = 0.2,
                           r_min_mm: float = 0.05) -> np.ndarray:
    """Superposed point-source potentials (mV) at ``xyz`` (n x 3, mm) from sources ``src`` (m x 3, mm)."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    src = np.atleast_2d(np.asarray(src, dtype=float))
    d = np.linalg.norm(xyz[:, None, :] - src[None, :, :], axis=2)
    d = np.maximum(d, r_min_mm)
    return 1e3 * (np.asarray(currents_mA, dtype=float)[None, :] / (4.0 * np.pi * sigma * d)).sum(axis=1)


def contact_surface_points(spec: LeadSpec, level: int, segment: Optional[int] = None, n_theta: int = 24,
                           n_z: int = 5) -> np.ndarray:
    """Point-source positions (mm) discretising one contact (ring or one arc of a segmented level)."""
    z0, z1 = spec.level_z(level)
    zs = z0 + (np.arange(n_z) + 0.5) / n_z * (z1 - z0)
    if segment is None or level not in spec.segmented_levels:
        thetas = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    else:
        centre = 2 * np.pi * segment / spec.n_segments
        half = np.deg2rad(spec.segment_arc_deg) / 2
        n_arc = max(4, int(round(n_theta * spec.segment_arc_deg / 360.0)))
        thetas = centre + (np.arange(n_arc) + 0.5) / n_arc * 2 * half - half
    r = spec.radius_mm
    pts = [(r * np.cos(t), r * np.sin(t), z) for z in zs for t in thetas]
    return np.asarray(pts, dtype=float)


@dataclass
class LeadField:
    """Potential field of one stimulation configuration.

    Parameters
    ----------
    spec : lead geometry.
    active : list of ``(level, segment_or_None, weight)``; weights are fractions of the total current
        (or, in voltage mode, all active contacts are assumed at the same set voltage).
    sigma : bulk conductivity (S/m).
    encapsulation_factor : multiplies the effective conductivity (e.g. 0.7 for a low-conductivity
        capsule; a coarse stand-in for an explicit encapsulation layer).
    mode : ``"current"`` (amplitude in mA) or ``"voltage"`` (amplitude in V).
    """

    spec: LeadSpec
    active: Sequence[Tuple[int, Optional[int], float]] = field(default_factory=lambda: [(1, None, 1.0)])
    sigma: float = 0.2
    encapsulation_factor: float = 1.0
    mode: str = "current"
    n_theta: int = 24
    n_z: int = 5
    _src: np.ndarray = field(init=False, repr=False)
    _cur: np.ndarray = field(init=False, repr=False)
    _volt_scale: float = field(init=False, repr=False, default=1.0)

    def __post_init__(self) -> None:
        srcs, curs = [], []
        total_w = sum(w for _, _, w in self.active)
        for level, seg, w in self.active:
            pts = contact_surface_points(self.spec, level, seg, self.n_theta, self.n_z)
            srcs.append(pts)
            curs.append(np.full(len(pts), (w / total_w) / len(pts)))
        self._src = np.vstack(srcs)
        self._cur = np.concatenate(curs)
        if self.mode == "voltage":
            # unit current -> mean surface potential; scale so that 1 V set gives that mean
            surf = self.unit_potential(self._src)
            self._volt_scale = 1e3 / float(np.mean(surf))  # mV per V
        elif self.mode != "current":
            raise ValueError("mode must be 'current' or 'voltage'")

    @property
    def sigma_eff(self) -> float:
        return self.sigma * self.encapsulation_factor

    def unit_potential(self, xyz: np.ndarray) -> np.ndarray:
        """Potential (mV) for 1 mA total current."""
        return point_source_potential(xyz, self._src, self._cur, self.sigma_eff)

    def potential(self, xyz: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """Potential (mV) at ``xyz`` for ``amplitude`` (mA in current mode, V in voltage mode)."""
        base = self.unit_potential(xyz)
        return base * amplitude * (self._volt_scale if self.mode == "voltage" else 1.0)

    def efield_magnitude(self, xyz: np.ndarray, amplitude: float = 1.0, h: float = 0.02) -> np.ndarray:
        """|E| in V/mm by central differences of the potential."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        grads = []
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            grads.append((self.potential(xyz + e, amplitude) - self.potential(xyz - e, amplitude)) / (2 * h))
        g = np.stack(grads, axis=1) * 1e-3  # mV/mm -> V/mm
        return np.linalg.norm(g, axis=1)

    def contact_centre(self) -> np.ndarray:
        return self._src.mean(axis=0)


class FieldFromNifti:
    """Potential field interpolated from a NIfTI volume (e.g. OSS-DBS / Lead-DBS export, in mV or V).

    Requires ``nibabel``. Coordinates are world (mm) coordinates of the volume's affine.
    """

    def __init__(self, path: str, units: str = "V") -> None:
        import nibabel as nib  # local import: optional dependency
        from scipy.ndimage import map_coordinates

        img = nib.load(path)
        self._data = np.asarray(img.dataobj, dtype=float)
        self._inv = np.linalg.inv(img.affine)
        self._map = map_coordinates
        self._scale = 1e3 if units == "V" else 1.0

    def potential(self, xyz: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        hom = np.column_stack([xyz, np.ones(len(xyz))])
        vox = (self._inv @ hom.T)[:3]
        return self._map(self._data, vox, order=1, mode="nearest") * self._scale * amplitude
