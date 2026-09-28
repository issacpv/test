"""SWC morphology parsing and a compartmental cable model of quasi-static polarization.

The physics implemented here is the *quasi-uniform field approximation*: over the
spatial extent of a single neuron (<~2 mm) the extracellular field produced by
tES / TMS / distant DBS contacts is treated as spatially uniform, so the
extracellular potential at position ``x`` is

    phi_e(x) = -E . x                                        (mV, x in mm)

The steady-state intracellular deviation from rest ``v`` of a passive, branched
cable driven by that extracellular potential solves

    (L + D) v = -L phi_e                                     (Eq. 1)

where ``L`` is the axial (core-conductor) conductance Laplacian in siemens and
``D = diag(g_m * A_i)`` the membrane leak conductance of each compartment. The
right-hand side ``-L phi_e`` is the discrete *activating function* (Rattay,
IEEE Trans Biomed Eng 1986): a compartment is driven only where the second
spatial derivative of ``phi_e`` along the neurite is non-zero, i.e. at bends,
branch points and terminals. Eq. 1 is linear in ``E``, so we solve it once per
unit field direction and scale afterwards -- that linearity is what makes
population sweeps over hundreds of morphologies tractable in numpy.

Units are CGS-with-mV, the convention used by NEURON:

===============  =========================
quantity         unit
===============  =========================
length           cm (SWC micrometres / 1e4)
diameter         cm
axial resistivity Ra, ohm.cm
specific membrane resistance Rm, ohm.cm^2
specific capacitance Cm, uF/cm^2
potential        mV
field            mV/mm == V/m
===============  =========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

__all__ = [
    "SWCNeuron",
    "CableModel",
    "PassiveParams",
    "load_swc",
    "SWC_TYPE_NAMES",
    "UM_PER_CM",
]

UM_PER_CM = 1.0e4
UM_PER_MM = 1.0e3

#: Standard SWC structure identifiers (Cannon et al. 1998 convention as used by
#: NeuroMorpho.Org's CNG-standardised files).
SWC_TYPE_NAMES: dict[int, str] = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "custom",
    6: "neurite",
    7: "glia_process",
}

CompartmentClass = Literal["soma", "axon", "dendrite", "other"]


@dataclass(frozen=True)
class PassiveParams:
    """Passive membrane and cytoplasm parameters.

    Defaults are the values conventionally used for cortical pyramidal cell
    models of transcranial stimulation (e.g. the Blue-Brain-derived models
    adapted by Aberra et al. 2018, J Neural Eng). ``Rm`` is the *specific*
    membrane resistance; the per-compartment leak conductance is ``A_i / Rm``.

    Attributes:
        Ra: Axial resistivity in ohm.cm.
        Rm: Specific membrane resistance in ohm.cm^2.
        Cm: Specific membrane capacitance in uF/cm^2.
        soma_Rm: Optional distinct specific membrane resistance for soma
            compartments; ``None`` reuses ``Rm``.
        myelin_Rm_factor: Multiplier applied to ``Rm`` on axonal compartments to
            crudely emulate myelination (internodal membrane is ~100-1000x less
            leaky). Set to 1.0 to disable.
    """

    Ra: float = 150.0
    Rm: float = 30_000.0
    Cm: float = 1.0
    soma_Rm: float | None = None
    myelin_Rm_factor: float = 1.0

    @property
    def tau_m_ms(self) -> float:
        """Membrane time constant Rm * Cm in milliseconds."""
        # ohm.cm^2 * uF/cm^2 = ohm.uF = 1e-6 s = 1e-3 ms
        return self.Rm * self.Cm * 1e-3


@dataclass
class SWCNeuron:
    """A parsed SWC reconstruction.

    Attributes:
        sample_id: (n,) int array of original SWC sample ids.
        xyz_um: (n, 3) float array of coordinates in micrometres.
        radius_um: (n,) float array of radii in micrometres.
        type_id: (n,) int array of SWC structure identifiers.
        parent: (n,) int array of *row indices* of each node's parent, -1 for roots.
        name: Identifier, typically the NeuroMorpho ``neuron_name``.
        metadata: Free-form provenance (species, region, cell type, archive...).
    """

    sample_id: np.ndarray
    xyz_um: np.ndarray
    radius_um: np.ndarray
    type_id: np.ndarray
    parent: np.ndarray
    name: str = "unnamed"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.sample_id)
        if self.xyz_um.shape != (n, 3):
            raise ValueError(f"xyz_um must be (n, 3), got {self.xyz_um.shape}")
        for arr, label in ((self.radius_um, "radius_um"), (self.type_id, "type_id"), (self.parent, "parent")):
            if arr.shape != (n,):
                raise ValueError(f"{label} must be ({n},), got {arr.shape}")

    @property
    def n_nodes(self) -> int:
        return len(self.sample_id)

    def compartment_class(self) -> np.ndarray:
        """Return a (n,) object array of coarse compartment classes."""
        out = np.empty(self.n_nodes, dtype=object)
        out[:] = "other"
        out[self.type_id == 1] = "soma"
        out[self.type_id == 2] = "axon"
        out[np.isin(self.type_id, (3, 4))] = "dendrite"
        return out

    def total_length_um(self) -> float:
        """Summed length of all parent->child segments, micrometres."""
        child = np.flatnonzero(self.parent >= 0)
        if child.size == 0:
            return 0.0
        seg = self.xyz_um[child] - self.xyz_um[self.parent[child]]
        return float(np.linalg.norm(seg, axis=1).sum())

    def terminal_mask(self) -> np.ndarray:
        """(n,) bool mask of leaf nodes (no children) that are not the root."""
        has_child = np.zeros(self.n_nodes, dtype=bool)
        par = self.parent[self.parent >= 0]
        has_child[par] = True
        return (~has_child) & (self.parent >= 0)

    def principal_axis(self, classes: Sequence[CompartmentClass] = ("dendrite",)) -> np.ndarray:
        """Unit vector of the dominant orientation of the selected compartments.

        For pyramidal cells run with ``("dendrite",)`` this recovers the
        somato-dendritic axis, the axis that governs polarization sign under tDCS
        (cf. the awake-mouse Purkinje result, bioRxiv 2023.02.18.529047). The
        estimate is the leading singular vector of the mean-centred coordinates,
        sign-fixed to point away from the soma.

        Args:
            classes: Compartment classes to include.

        Returns:
            (3,) unit vector.
        """
        cls = self.compartment_class()
        sel = np.isin(cls.astype(str), np.asarray(classes, dtype=str))
        if sel.sum() < 3:
            sel = np.ones(self.n_nodes, dtype=bool)
        pts = self.xyz_um[sel]
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis = vt[0]
        soma = self.soma_centre_um()
        if np.dot(pts.mean(axis=0) - soma, axis) < 0:
            axis = -axis
        norm = np.linalg.norm(axis)
        return axis / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    def soma_centre_um(self) -> np.ndarray:
        """Centroid of soma nodes, falling back to the root node."""
        soma = self.type_id == 1
        if soma.any():
            return self.xyz_um[soma].mean(axis=0)
        root = np.flatnonzero(self.parent < 0)
        idx = root[0] if root.size else 0
        return self.xyz_um[idx]


def load_swc(path: str | Path, name: str | None = None, **metadata: object) -> SWCNeuron:
    """Parse an SWC file into an :class:`SWCNeuron`.

    Robust to the quirks of real NeuroMorpho.Org CNG files: ``#`` comment
    headers, blank lines, whitespace or comma delimiters, non-contiguous and
    non-monotonic sample ids, and multiple roots (parent ``-1``). Nodes whose
    declared parent id is absent from the file are demoted to roots rather than
    raising, because a handful of archive files are truncated.

    Args:
        path: Path to the ``.swc`` file.
        name: Neuron identifier; defaults to the file stem with ``.CNG`` stripped.
        **metadata: Provenance recorded on the returned object.

    Returns:
        The parsed morphology.

    Raises:
        ValueError: If the file contains no parsable data rows.
    """
    path = Path(path)
    rows: list[tuple[int, int, float, float, float, float, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    (
                        int(float(parts[0])),
                        int(float(parts[1])),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        int(float(parts[6])),
                    )
                )
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no parsable SWC rows in {path}")

    sample_id = np.array([r[0] for r in rows], dtype=np.int64)
    type_id = np.array([r[1] for r in rows], dtype=np.int64)
    xyz_um = np.array([(r[2], r[3], r[4]) for r in rows], dtype=float)
    radius_um = np.array([r[5] for r in rows], dtype=float)
    parent_id = np.array([r[6] for r in rows], dtype=np.int64)

    id_to_row = {int(sid): i for i, sid in enumerate(sample_id)}
    parent = np.full(len(rows), -1, dtype=np.int64)
    for i, pid in enumerate(parent_id):
        if pid >= 0:
            parent[i] = id_to_row.get(int(pid), -1)

    stem = path.name
    for suffix in (".swc", ".SWC"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem.endswith(".CNG"):
        stem = stem[:-4]

    return SWCNeuron(
        sample_id=sample_id,
        xyz_um=xyz_um,
        radius_um=np.maximum(radius_um, 1e-3),
        type_id=type_id,
        parent=parent,
        name=name or stem,
        metadata=dict(metadata),
    )


class CableModel:
    """Passive compartmental cable model built from an SWC morphology.

    One compartment per SWC node. Segment ``i`` (for non-root ``i``) is the
    frustum from ``parent(i)`` to ``i``; its length and mean diameter set both
    the axial conductance to the parent and the membrane area attributed to
    ``i``. Root nodes get the area of a sphere of their own radius, which keeps
    soma area realistic for the single-point-soma files that dominate
    NeuroMorpho.

    Example:
        >>> import numpy as np
        >>> from morph_stim.swc_cable import CableModel, SWCNeuron
        >>> n = 40
        >>> neuron = SWCNeuron(
        ...     sample_id=np.arange(1, n + 1),
        ...     xyz_um=np.c_[np.zeros(n), np.zeros(n), np.arange(n) * 10.0],
        ...     radius_um=np.full(n, 0.5),
        ...     type_id=np.r_[1, np.full(n - 1, 3)],
        ...     parent=np.r_[-1, np.arange(n - 1)],
        ... )
        >>> model = CableModel(neuron)
        >>> v = model.steady_state_polarization(np.array([0.0, 0.0, 1.0]))
        >>> bool(v[0] * v[-1] < 0)  # opposite ends polarize with opposite sign
        True
    """

    def __init__(self, neuron: SWCNeuron, params: PassiveParams | None = None) -> None:
        """Assemble geometry and the conductance matrices.

        Args:
            neuron: Parsed morphology.
            params: Passive parameters; defaults to :class:`PassiveParams`.
        """
        self.neuron = neuron
        self.params = params or PassiveParams()
        self._build_geometry()
        self._build_matrices()

    # ---------------------------------------------------------------- geometry
    def _build_geometry(self) -> None:
        nrn = self.neuron
        n = nrn.n_nodes
        self.n = n
        self.xyz_cm = nrn.xyz_um / UM_PER_CM
        self.diam_cm = 2.0 * nrn.radius_um / UM_PER_CM
        self.cls = nrn.compartment_class()

        child = np.flatnonzero(nrn.parent >= 0)
        par = nrn.parent[child]
        seg_len = np.linalg.norm(self.xyz_cm[child] - self.xyz_cm[par], axis=1)
        # Degenerate duplicate points occur in real files; floor the length so the
        # axial resistance stays finite.
        seg_len = np.maximum(seg_len, 1e-8)
        seg_diam = 0.5 * (self.diam_cm[child] + self.diam_cm[par])

        self.seg_child = child
        self.seg_parent = par
        self.seg_len_cm = seg_len
        self.seg_diam_cm = seg_diam

        # Membrane area: segment lateral area on the child, sphere on roots.
        area = np.zeros(n)
        np.add.at(area, child, np.pi * seg_diam * seg_len)
        roots = np.flatnonzero(nrn.parent < 0)
        area[roots] += np.pi * self.diam_cm[roots] ** 2
        self.area_cm2 = np.maximum(area, 1e-14)

        # Axial conductance of each segment: two half-segment resistances in series.
        # R_half = 4 * Ra * (len/2) / (pi * d^2)
        r_axial = 4.0 * self.params.Ra * seg_len / (np.pi * seg_diam**2)
        self.seg_g_axial_S = 1.0 / np.maximum(r_axial, 1e-12)

    def _rm_per_compartment(self) -> np.ndarray:
        p = self.params
        rm = np.full(self.n, p.Rm, dtype=float)
        if p.soma_Rm is not None:
            rm[self.cls.astype(str) == "soma"] = p.soma_Rm
        if p.myelin_Rm_factor != 1.0:
            rm[self.cls.astype(str) == "axon"] *= p.myelin_Rm_factor
        return rm

    def _build_matrices(self) -> None:
        n = self.n
        i = self.seg_child
        j = self.seg_parent
        g = self.seg_g_axial_S
        # Laplacian L: L[i,i] += g, L[j,j] += g, L[i,j] -= g, L[j,i] -= g
        rows = np.concatenate([i, j, i, j])
        cols = np.concatenate([i, j, j, i])
        vals = np.concatenate([g, g, -g, -g])
        self.L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        self.g_leak_S = self.area_cm2 / self._rm_per_compartment()
        self.cap_uF = self.area_cm2 * self.params.Cm
        self._A = (self.L + diags(self.g_leak_S)).tocsc()

    # ------------------------------------------------------------------ solving
    def extracellular_potential_mV(self, e_field: np.ndarray) -> np.ndarray:
        """Extracellular potential at every compartment for a uniform field.

        Args:
            e_field: (3,) field vector in mV/mm (equivalently V/m).

        Returns:
            (n,) potential in mV, gauge-fixed to zero at the soma centroid.
        """
        e_field = np.asarray(e_field, dtype=float).reshape(3)
        xyz_mm = self.neuron.xyz_um / UM_PER_MM
        ref_mm = self.neuron.soma_centre_um() / UM_PER_MM
        return -(xyz_mm - ref_mm) @ e_field

    def activating_function(self, e_field: np.ndarray) -> np.ndarray:
        """Discrete activating function ``-L phi_e`` in nanoamperes.

        Args:
            e_field: (3,) field vector in mV/mm.

        Returns:
            (n,) equivalent injected current per compartment, nA.
        """
        phi = self.extracellular_potential_mV(e_field)
        # L is in S, phi in mV  ->  S*mV = mA; convert to nA.
        return -(self.L @ phi) * 1e6

    def steady_state_polarization(self, e_field: np.ndarray) -> np.ndarray:
        """Solve Eq. 1 for the steady-state membrane polarization.

        Args:
            e_field: (3,) field vector in mV/mm. Because the system is linear,
                passing a unit vector yields polarization *per V/m*.

        Returns:
            (n,) membrane potential deviation from rest in mV. Positive is
            depolarizing.
        """
        phi = self.extracellular_potential_mV(e_field)
        rhs = -(self.L @ phi)
        return np.asarray(spsolve(self._A, rhs), dtype=float)

    def polarization_per_unit_field(self, directions: np.ndarray) -> np.ndarray:
        """Polarization for a batch of unit field directions.

        Exploits linearity: solving for the three cartesian unit fields gives a
        (n, 3) sensitivity matrix, and any direction is a linear combination.
        This makes a 200-direction orientation sweep cost three sparse solves
        instead of 200.

        Args:
            directions: (k, 3) array of field direction vectors, not necessarily
                normalised; magnitudes are interpreted as V/m.

        Returns:
            (k, n) polarization in mV.
        """
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        if directions.shape[1] != 3:
            raise ValueError(f"directions must be (k, 3), got {directions.shape}")
        basis = np.stack(
            [self.steady_state_polarization(np.eye(3)[d]) for d in range(3)], axis=1
        )  # (n, 3)
        self._sensitivity_mV_per_Vpm = basis
        return directions @ basis.T

    @property
    def sensitivity_mV_per_Vpm(self) -> np.ndarray:
        """(n, 3) cached polarization sensitivity to the cartesian unit fields."""
        if not hasattr(self, "_sensitivity_mV_per_Vpm"):
            self.polarization_per_unit_field(np.eye(3))
        return self._sensitivity_mV_per_Vpm

    # ------------------------------------------------------------- descriptors
    def polarization_length_um(self, e_field_dir: np.ndarray) -> float:
        """Effective polarization length ``max|v| / |E|`` expressed in micrometres.

        For a semi-infinite straight cable aligned with the field this equals the
        electrotonic length constant ``lambda = sqrt(d*Rm/(4*Ra))``; for a real
        branched morphology it is the empirical analogue and is the quantity
        reported experimentally as "mV per V/m" (Bikson et al. 2004, J Physiol
        report ~0.2 mV/(V/m) somatic polarization in rat hippocampal slices).

        Args:
            e_field_dir: (3,) direction; normalised internally.

        Returns:
            Effective length in micrometres.
        """
        d = np.asarray(e_field_dir, dtype=float).reshape(3)
        norm = np.linalg.norm(d)
        if norm == 0:
            return 0.0
        v = self.steady_state_polarization(d / norm)  # mV per V/m == mV per mV/mm
        return float(np.abs(v).max() * UM_PER_MM)

    def summary(self) -> dict[str, float | str | int]:
        """Morphological and electrotonic descriptors used as mixed-model covariates."""
        nrn = self.neuron
        cls = self.cls.astype(str)
        term = nrn.terminal_mask()
        soma = nrn.soma_centre_um()
        radial = np.linalg.norm(nrn.xyz_um - soma, axis=1)
        axis = nrn.principal_axis()
        along = (nrn.xyz_um - soma) @ axis
        return {
            "name": nrn.name,
            "n_nodes": int(self.n),
            "total_length_um": float(nrn.total_length_um()),
            "n_terminals": int(term.sum()),
            "n_axon_terminals": int((term & (cls == "axon")).sum()),
            "max_radial_extent_um": float(radial.max()),
            "somatodendritic_span_um": float(along.max() - along.min()),
            "mean_diam_um": float(2.0 * nrn.radius_um.mean()),
            "membrane_area_um2": float(self.area_cm2.sum() * UM_PER_CM**2),
            "has_axon": bool((cls == "axon").any()),
            "frac_axon_nodes": float((cls == "axon").mean()),
        }


def build_models(
    neurons: Iterable[SWCNeuron], params: PassiveParams | None = None
) -> list[CableModel]:
    """Convenience wrapper: build a :class:`CableModel` for each morphology."""
    return [CableModel(n, params) for n in neurons]
