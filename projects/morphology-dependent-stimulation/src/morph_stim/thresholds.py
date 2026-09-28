"""Orientation sweeps and activation-threshold estimation under uniform fields.

Given a :class:`~morph_stim.swc_cable.CableModel`, this module answers the
question the project is built around: *how strong a field, pointing which way, is
needed to bring this particular morphology to firing threshold?*

Two approximations are made explicit, because they are the price of scaling to
hundreds of morphologies without NEURON:

1. **Quasi-uniform field.** The field is spatially constant over the neuron.
   This is well justified for tES and TMS at the scale of one cortical neuron,
   and for DBS only at distances >~2 mm from the contact (see README risks).
2. **Polarization threshold.** A compartment fires when its steady-state
   depolarization reaches ``delta_v_th`` above rest. This replaces integrating
   Hodgkin-Huxley-type channels and is exact only in the long-pulse limit; the
   :class:`Waveform` low-pass correction below extends it approximately to TMS
   and DBS pulse widths. Absolute thresholds from this surrogate should be
   treated as ordinal, and calibrated against NEURON on a subset
   (:mod:`morph_stim.neuron_driver`) before being reported as mV or V/m.

The quantity that the population analysis actually consumes is the *relative*
spread of thresholds across morphologies, which is far more robust to (1) and
(2) than any absolute value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .swc_cable import CableModel

__all__ = [
    "Waveform",
    "MODALITIES",
    "fibonacci_directions",
    "threshold_for_directions",
    "orientation_sweep",
    "sweep_population",
]

ExcitableSite = Literal["axon_terminal", "any_terminal", "soma", "all"]


@dataclass(frozen=True)
class Waveform:
    """Temporal characteristics of a stimulation waveform.

    The membrane acts as a first-order low-pass filter with time constant
    ``tau_m = Rm * Cm``, so a brief pulse produces less polarization than the
    steady-state solution. ``attenuation`` returns the fraction of steady-state
    polarization reached.

    Attributes:
        name: Label, e.g. ``"tDCS"``.
        kind: ``"dc"``, ``"sinusoid"`` or ``"pulse"``.
        frequency_hz: Frequency for ``"sinusoid"``.
        pulse_width_ms: Effective pulse width for ``"pulse"``.
    """

    name: str
    kind: Literal["dc", "sinusoid", "pulse"]
    frequency_hz: float | None = None
    pulse_width_ms: float | None = None

    def attenuation(self, tau_m_ms: float) -> float:
        """Fraction of the steady-state polarization reached by this waveform.

        Args:
            tau_m_ms: Membrane time constant in milliseconds.

        Returns:
            A factor in (0, 1].

        Raises:
            ValueError: If required parameters for ``kind`` are missing.
        """
        if tau_m_ms <= 0:
            raise ValueError("tau_m_ms must be positive")
        if self.kind == "dc":
            return 1.0
        if self.kind == "sinusoid":
            if self.frequency_hz is None:
                raise ValueError(f"{self.name}: sinusoid needs frequency_hz")
            omega_tau = 2.0 * np.pi * self.frequency_hz * (tau_m_ms * 1e-3)
            return float(1.0 / np.sqrt(1.0 + omega_tau**2))
        if self.kind == "pulse":
            if self.pulse_width_ms is None:
                raise ValueError(f"{self.name}: pulse needs pulse_width_ms")
            return float(1.0 - np.exp(-self.pulse_width_ms / tau_m_ms))
        raise ValueError(f"unknown waveform kind {self.kind!r}")


#: Representative waveforms for the four modalities under study. Pulse widths are
#: the conventional clinical values: ~0.2 ms for the dominant phase of a
#: biphasic TMS pulse, 60 us for standard DBS.
MODALITIES: dict[str, Waveform] = {
    "tDCS": Waveform("tDCS", "dc"),
    "tACS_10Hz": Waveform("tACS_10Hz", "sinusoid", frequency_hz=10.0),
    "tACS_40Hz": Waveform("tACS_40Hz", "sinusoid", frequency_hz=40.0),
    "tACS_1kHz": Waveform("tACS_1kHz", "sinusoid", frequency_hz=1000.0),
    "TMS": Waveform("TMS", "pulse", pulse_width_ms=0.2),
    "DBS": Waveform("DBS", "pulse", pulse_width_ms=0.06),
}


def fibonacci_directions(n: int = 128, hemisphere: bool = False) -> np.ndarray:
    """Near-uniformly distributed unit vectors on the sphere.

    A spherical Fibonacci lattice avoids the polar clustering of a naive
    theta/phi grid, so the minimum-threshold direction is found with far fewer
    solves.

    Args:
        n: Number of directions.
        hemisphere: If ``True``, return directions with ``z >= 0`` only. Useful
            when the response is known to be polarity-antisymmetric and the
            opposite polarity is handled separately.

    Returns:
        (n, 3) array of unit vectors.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    i = np.arange(n, dtype=float) + 0.5
    golden = np.pi * (1.0 + 5.0**0.5)
    if hemisphere:
        z = 1.0 - i / n
    else:
        z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    theta = golden * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _site_mask(model: CableModel, site: ExcitableSite) -> np.ndarray:
    """Boolean mask of compartments treated as spike-initiation sites."""
    nrn = model.neuron
    cls = model.cls.astype(str)
    term = nrn.terminal_mask()
    if site == "axon_terminal":
        mask = term & (cls == "axon")
        if not mask.any():  # reconstruction lacks an axon: fall back, and flag it
            mask = term
    elif site == "any_terminal":
        mask = term
    elif site == "soma":
        mask = cls == "soma"
    elif site == "all":
        mask = np.ones(model.n, dtype=bool)
    else:
        raise ValueError(f"unknown site {site!r}")
    if not mask.any():
        mask = np.ones(model.n, dtype=bool)
    return mask


def threshold_for_directions(
    model: CableModel,
    directions: np.ndarray,
    *,
    waveform: Waveform | str = "tDCS",
    delta_v_th: float = 12.0,
    site: ExcitableSite = "axon_terminal",
    bidirectional: bool = True,
) -> np.ndarray:
    """Threshold field magnitude for each candidate field direction.

    Threshold is ``delta_v_th / (attenuation * max_depolarization_per_Vpm)``
    taken over the excitable compartments. With ``bidirectional=True`` both field
    polarities are tried and the easier one kept, which is the right convention
    for TMS and tACS (the coil or the AC cycle supplies both signs) but *not*
    for tDCS, where polarity is fixed by montage.

    Args:
        model: Cable model of one morphology.
        directions: (k, 3) field directions; normalised internally.
        waveform: A :class:`Waveform` or a key of :data:`MODALITIES`.
        delta_v_th: Depolarization from rest required to fire, in mV.
        site: Which compartments are treated as excitable.
        bidirectional: Whether to allow field reversal.

    Returns:
        (k,) threshold magnitudes in V/m. Entries are ``inf`` where no excitable
        compartment depolarizes at all.

    Raises:
        ValueError: If ``delta_v_th <= 0``.
    """
    if delta_v_th <= 0:
        raise ValueError("delta_v_th must be positive")
    wf = MODALITIES[waveform] if isinstance(waveform, str) else waveform
    atten = wf.attenuation(model.params.tau_m_ms)

    dirs = np.atleast_2d(np.asarray(directions, dtype=float))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(norms == 0, 1.0, norms)

    sens = model.sensitivity_mV_per_Vpm  # (n, 3), mV per V/m
    mask = _site_mask(model, site)
    v = dirs @ sens[mask].T  # (k, m) polarization per V/m at excitable sites

    peak = v.max(axis=1)
    if bidirectional:
        peak = np.maximum(peak, (-v).max(axis=1))
    peak = peak * atten

    with np.errstate(divide="ignore"):
        thr = np.where(peak > 0, delta_v_th / peak, np.inf)
    return thr


def orientation_sweep(
    model: CableModel,
    *,
    waveform: Waveform | str = "tDCS",
    n_directions: int = 128,
    delta_v_th: float = 12.0,
    site: ExcitableSite = "axon_terminal",
    bidirectional: bool = True,
    column_axis: np.ndarray | None = None,
) -> dict[str, object]:
    """Full orientation sweep for one morphology.

    Args:
        model: Cable model.
        waveform: Modality waveform or :data:`MODALITIES` key.
        n_directions: Number of sampled field directions.
        delta_v_th: Firing threshold above rest, mV.
        site: Excitable compartment definition.
        bidirectional: Allow field reversal.
        column_axis: (3,) cortical column normal (pia-to-white-matter). If given,
            thresholds for the field parallel and perpendicular to the column are
            reported; this is the axis that a head model supplies per cortical
            location, and the one that makes the result translatable to dosing.
            Defaults to the morphology's own somatodendritic axis.

    Returns:
        Dict with keys ``threshold_min_Vpm``, ``threshold_median_Vpm``,
        ``threshold_max_Vpm``, ``best_direction``, ``anisotropy_ratio``,
        ``threshold_along_column_Vpm``, ``threshold_across_column_Vpm``,
        ``column_alignment`` (cosine between best direction and column axis),
        ``polarization_length_um`` and ``waveform``.
    """
    wf = MODALITIES[waveform] if isinstance(waveform, str) else waveform
    dirs = fibonacci_directions(n_directions)
    thr = threshold_for_directions(
        model,
        dirs,
        waveform=wf,
        delta_v_th=delta_v_th,
        site=site,
        bidirectional=bidirectional,
    )
    finite = np.isfinite(thr)
    best_idx = int(np.argmin(thr)) if finite.any() else 0

    axis = (
        model.neuron.principal_axis()
        if column_axis is None
        else np.asarray(column_axis, dtype=float).reshape(3)
    )
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    perp = _orthogonal_unit(axis)
    thr_axis = threshold_for_directions(
        model, np.stack([axis, perp]), waveform=wf, delta_v_th=delta_v_th,
        site=site, bidirectional=bidirectional,
    )

    t_min = float(thr[best_idx])
    t_max = float(thr[finite].max()) if finite.any() else float("inf")
    return {
        "name": model.neuron.name,
        "waveform": wf.name,
        "threshold_min_Vpm": t_min,
        "threshold_median_Vpm": float(np.median(thr[finite])) if finite.any() else float("inf"),
        "threshold_max_Vpm": t_max,
        "best_direction": dirs[best_idx].tolist(),
        "anisotropy_ratio": float(t_max / t_min) if t_min > 0 and np.isfinite(t_max) else float("inf"),
        "threshold_along_column_Vpm": float(thr_axis[0]),
        "threshold_across_column_Vpm": float(thr_axis[1]),
        "column_alignment": float(abs(np.dot(dirs[best_idx], axis))),
        "polarization_length_um": model.polarization_length_um(axis),
        "n_directions": int(n_directions),
        "delta_v_th_mV": float(delta_v_th),
        "site": site,
        "axon_present": bool((model.cls.astype(str) == "axon").any()),
    }


def _orthogonal_unit(axis: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector orthogonal to ``axis``."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    perp = seed - np.dot(seed, axis) * axis
    return perp / max(np.linalg.norm(perp), 1e-12)


def sweep_population(
    models: Sequence[CableModel],
    *,
    waveforms: Sequence[Waveform | str] = ("tDCS", "TMS"),
    n_directions: int = 128,
    delta_v_th: float = 12.0,
    site: ExcitableSite = "axon_terminal",
    metadata_keys: Sequence[str] = ("species", "brain_region", "cell_type", "archive", "layer"),
) -> list[dict[str, object]]:
    """Run :func:`orientation_sweep` over a population of morphologies.

    Morphological descriptors from :meth:`CableModel.summary` and selected
    provenance fields are merged into each row, giving a tidy table ready for
    :mod:`morph_stim.popstats`.

    Args:
        models: Cable models.
        waveforms: Waveforms or :data:`MODALITIES` keys to evaluate.
        n_directions: Directions per sweep.
        delta_v_th: Firing threshold above rest, mV.
        site: Excitable compartment definition.
        metadata_keys: Keys lifted from ``neuron.metadata`` onto each row.

    Returns:
        One dict per (morphology, waveform) pair.
    """
    rows: list[dict[str, object]] = []
    for model in models:
        summary = model.summary()
        meta = {k: model.neuron.metadata.get(k) for k in metadata_keys}
        for wf in waveforms:
            row = orientation_sweep(
                model,
                waveform=wf,
                n_directions=n_directions,
                delta_v_th=delta_v_th,
                site=site,
            )
            row.update(summary)
            row.update(meta)
            rows.append(row)
    return rows
