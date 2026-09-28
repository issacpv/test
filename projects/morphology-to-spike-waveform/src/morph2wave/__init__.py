"""morph2wave: from dendritic morphology to extracellular spike waveforms on high-density probes.

Modules
-------
swc
    SWC parsing, morphology features (soma, stems, shell surface areas, polarity, AIS origin)
    and compartmentalisation for the forward model.
eap_forward
    Probe geometries (Neuropixels 1.0 / 2.0 / Ultra), line-source extracellular potentials,
    and a phenomenological propagating-spike current template for fast factorial sweeps.
waveform_features
    Single-channel and spatiotemporal EAP features (trough-to-peak, footprint, decay, velocity).
regression
    Cross-validated morphology -> waveform regression, permutation importance and a factorial
    variance partition (morphology x channel set x placement).
"""
from . import eap_forward, regression, swc, waveform_features  # noqa: F401

__all__ = ["swc", "eap_forward", "waveform_features", "regression"]
__version__ = "0.1.0"
