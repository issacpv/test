"""Optional NEURON driver for calibrating the numpy surrogate against real spikes.

The numpy cable model in :mod:`morph_stim.swc_cable` gives *steady-state passive
polarization*; turning that into a firing threshold requires the polarization
surrogate described in :mod:`morph_stim.thresholds`. This module closes the loop
on a subset of morphologies by running the same geometry in NEURON
(Hines & Carnevale 1997, Neural Computation) with active channels and an
``extracellular`` mechanism driven by ``e_extracellular``, then bisecting on field
amplitude to find the true threshold.

NEURON is an optional dependency. Import this module freely: it degrades to
:func:`neuron_available` returning ``False`` and every driver call raising a clear
:class:`RuntimeError`, so CI and the population sweep never depend on it.

Workflow:
    1. Run the numpy sweep over the full population.
    2. Sample ~30-50 morphologies stratified by species, class and numpy threshold
       decile.
    3. Run :func:`neuron_threshold` on those.
    4. Regress NEURON threshold on numpy threshold; report the calibration slope,
       R^2 and residual spread in the paper. The population claims stand on the
       *rank* correlation being high, not on the surrogate being unbiased.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .swc_cable import UM_PER_MM, SWCNeuron

__all__ = ["neuron_available", "NeuronConfig", "neuron_threshold", "calibration_report"]

LOG = logging.getLogger(__name__)


def neuron_available() -> bool:
    """Return ``True`` if the ``neuron`` Python module can be imported."""
    try:
        import neuron  # noqa: F401
    except Exception:  # noqa: BLE001 - NEURON import can fail many ways
        return False
    return True


def _require_neuron() -> Any:
    """Import and return the NEURON ``h`` object.

    Raises:
        RuntimeError: If NEURON is not installed.
    """
    try:
        from neuron import h
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "NEURON is not available. Install it (`pip install neuron`) or use the "
            "numpy surrogate in morph_stim.thresholds, which needs no NEURON."
        ) from exc
    h.load_file("stdrun.hoc")
    return h


@dataclass(frozen=True)
class NeuronConfig:
    """Simulation settings for the NEURON driver.

    Attributes:
        temperature_c: ``celsius``; channel kinetics are strongly temperature
            dependent, so this must match whatever the channel models were fit at.
        v_init: Initial membrane potential, mV.
        dt_ms: Integration step.
        tstop_ms: Total simulated time.
        stim_onset_ms: When the field turns on.
        stim_dur_ms: How long it stays on. For TMS use ~0.2; for tDCS use a value
            several times ``tau_m`` so the response is at steady state.
        spike_threshold_mV: Crossing level counted as a spike.
        mechanism: ``"hh"`` for a quick smoke test, or the name of a compiled
            mod-file suite (e.g. the Blue-Brain / Aberra channel set) once
            ``nrnivmodl`` has been run in the working directory.
        n_bisect: Bisection iterations on field amplitude.
        amp_bracket_Vpm: Initial (low, high) bracket for the threshold search.
    """

    temperature_c: float = 37.0
    v_init: float = -70.0
    dt_ms: float = 0.005
    tstop_ms: float = 20.0
    stim_onset_ms: float = 2.0
    stim_dur_ms: float = 0.2
    spike_threshold_mV: float = -10.0
    mechanism: str = "hh"
    n_bisect: int = 12
    amp_bracket_Vpm: tuple[float, float] = (1.0, 5000.0)


def _build_sections(h: Any, neuron: SWCNeuron, cfg: NeuronConfig) -> list[Any]:
    """Create one NEURON section per SWC node, wired by the parent array.

    Sections are single-segment: the SWC sampling (typically 1-5 um) is already
    finer than the electrotonic length constant, so further ``nseg`` refinement
    buys little and costs a great deal in a population run.
    """
    sections: list[Any] = []
    for i in range(neuron.n_nodes):
        sec = h.Section(name=f"c{i}")
        sec.nseg = 1
        sec.diam = max(2.0 * float(neuron.radius_um[i]), 0.05)
        sec.Ra = 150.0
        sec.cm = 1.0
        sec.insert("extracellular")
        sec.insert(cfg.mechanism)
        sections.append(sec)

    for i in range(neuron.n_nodes):
        p = int(neuron.parent[i])
        if p < 0:
            sections[i].L = max(2.0 * float(neuron.radius_um[i]), 0.1)
            continue
        seg = neuron.xyz_um[i] - neuron.xyz_um[p]
        sections[i].L = max(float(np.linalg.norm(seg)), 0.1)
        sections[i].connect(sections[p](1.0), 0.0)
    return sections


def _apply_field(
    h: Any, sections: Sequence[Any], neuron: SWCNeuron, e_field_Vpm: np.ndarray
) -> None:
    """Set ``e_extracellular`` on every section for a uniform field.

    ``e_extracellular`` is in mV and positions in NEURON are unitless here, so we
    compute ``phi_e = -E . x`` with ``E`` in V/m (== mV/mm) and ``x`` in mm,
    matching :meth:`CableModel.extracellular_potential_mV`.
    """
    e = np.asarray(e_field_Vpm, dtype=float).reshape(3)
    ref_mm = neuron.soma_centre_um() / UM_PER_MM
    xyz_mm = neuron.xyz_um / UM_PER_MM
    phi = -(xyz_mm - ref_mm) @ e
    for sec, value in zip(sections, phi):
        for seg in sec:
            seg.e_extracellular = float(value)


def neuron_threshold(
    neuron: SWCNeuron,
    direction: np.ndarray,
    cfg: NeuronConfig | None = None,
) -> dict[str, float]:
    """Bisect on field amplitude to find the true activation threshold in NEURON.

    Args:
        neuron: Morphology to simulate.
        direction: (3,) field direction; normalised internally.
        cfg: Simulation settings.

    Returns:
        Dict with ``threshold_Vpm`` (``inf`` if the upper bracket never fires),
        ``bracket_low``, ``bracket_high`` and ``n_sections``.

    Raises:
        RuntimeError: If NEURON is unavailable.
    """
    cfg = cfg or NeuronConfig()
    h = _require_neuron()
    h.celsius = cfg.temperature_c
    h.dt = cfg.dt_ms

    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    sections = _build_sections(h, neuron, cfg)

    def fires(amp: float) -> bool:
        _apply_field(h, sections, neuron, d * amp)
        # Gate the field in time by scaling e_extracellular with a step: NEURON
        # has no built-in ramp for this, so run pre-stim, stim and post-stim legs.
        h.finitialize(cfg.v_init)
        _apply_field(h, sections, neuron, np.zeros(3))
        h.continuerun(cfg.stim_onset_ms)
        _apply_field(h, sections, neuron, d * amp)
        h.continuerun(cfg.stim_onset_ms + cfg.stim_dur_ms)
        _apply_field(h, sections, neuron, np.zeros(3))
        peak = max(float(sec(0.5).v) for sec in sections)
        h.continuerun(cfg.tstop_ms)
        peak = max(peak, max(float(sec(0.5).v) for sec in sections))
        return peak > cfg.spike_threshold_mV

    lo, hi = cfg.amp_bracket_Vpm
    if not fires(hi):
        LOG.warning("%s: no spike at %.0f V/m", neuron.name, hi)
        return {"threshold_Vpm": float("inf"), "bracket_low": lo, "bracket_high": hi,
                "n_sections": float(len(sections))}
    if fires(lo):
        return {"threshold_Vpm": lo, "bracket_low": lo, "bracket_high": lo,
                "n_sections": float(len(sections))}
    for _ in range(cfg.n_bisect):
        mid = 0.5 * (lo + hi)
        if fires(mid):
            hi = mid
        else:
            lo = mid
    return {"threshold_Vpm": 0.5 * (lo + hi), "bracket_low": lo, "bracket_high": hi,
            "n_sections": float(len(sections))}


def calibration_report(
    numpy_thresholds: Sequence[float], neuron_thresholds: Sequence[float]
) -> dict[str, float]:
    """Compare surrogate and NEURON thresholds on a calibration subset.

    Args:
        numpy_thresholds: Surrogate thresholds, V/m.
        neuron_thresholds: NEURON thresholds, V/m.

    Returns:
        Dict with ``n``, ``spearman_rho``, ``log_slope``, ``log_intercept``,
        ``log_r2`` and ``median_fold_error``.

    Raises:
        ValueError: If fewer than three finite paired values are supplied.
    """
    a = np.asarray(numpy_thresholds, dtype=float)
    b = np.asarray(neuron_thresholds, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 3:
        raise ValueError(f"need >=3 finite pairs, got {int(ok.sum())}")
    la, lb = np.log10(a[ok]), np.log10(b[ok])

    ra = np.argsort(np.argsort(la)).astype(float)
    rb = np.argsort(np.argsort(lb)).astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])

    slope, intercept = np.polyfit(la, lb, 1)
    pred = slope * la + intercept
    ss_res = float(((lb - pred) ** 2).sum())
    ss_tot = float(((lb - lb.mean()) ** 2).sum())
    return {
        "n": int(ok.sum()),
        "spearman_rho": rho,
        "log_slope": float(slope),
        "log_intercept": float(intercept),
        "log_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "median_fold_error": float(np.median(10.0 ** np.abs(lb - la))),
    }
