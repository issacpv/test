"""fluid_resp: waveform-derived pulse-pressure variation vs fluid-bolus response in MIMIC.

Modules
-------
abp_beats   Beat detection, per-beat pressures, signal-quality mask, synthetic ABP generator.
ppv         Windowed PPV / SPV, respiratory rate from PP modulation.
boluses     Bolus identification from ``inputevents``, pre/post response windows, sham windows, prerequisites.
evaluate    AUROC with cluster bootstrap, gray zone, attributable response, stratified AUROC.
"""

from . import abp_beats, boluses, evaluate, ppv  # noqa: F401

__all__ = ["abp_beats", "boluses", "evaluate", "ppv"]
__version__ = "0.1.0"
