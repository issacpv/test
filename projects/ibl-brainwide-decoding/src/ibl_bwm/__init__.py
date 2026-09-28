"""ibl_bwm: cross-lab, cross-region decoding benchmark on the IBL Brain-wide Map (+ Allen Visual Coding).

Modules
-------
loaders       session containers, ONE / NWB loaders (lazy imports), Beryl region assignment, synthetic sessions
binning       trial-aligned binned spike tensors and target extraction
decoding      cross-validated logistic / ridge decoders, shuffle and pseudo-session nulls, latents, yield curves
hierarchical  random-effects meta-analysis, lab/subject variance components (ICC), leave-one-lab-out fragility
"""
from . import binning, decoding, hierarchical, loaders  # noqa: F401

__version__ = "0.1.0"
