"""nm_fingerprint: provenance detectability and harmonisation for neuron reconstructions.

Modules
-------
swc_features : SWC parsing, morphometrics, sampling-fingerprint features, resampling.
fingerprint  : grouped-CV provenance classifiers, detectability index, permutation nulls,
               feature importance, per-record provenance risk.
harmonize    : parametric ComBat for feature tables, kBET-style local mixing test,
               effect-size preservation checks.
synthetic    : synthetic SWC trees with controllable tracing artefacts.
"""
from .swc_features import read_swc, write_swc, morphometrics, sampling_fingerprint, feature_vector, resample_swc
from .fingerprint import detectability, archive_level_permutation_null, provenance_risk, feature_subset_contrast
from .harmonize import combat, kbet_rejection_rate, effect_size_preservation
from .synthetic import make_tree

__all__ = [
    "read_swc", "write_swc", "morphometrics", "sampling_fingerprint", "feature_vector", "resample_swc",
    "detectability", "archive_level_permutation_null", "provenance_risk", "feature_subset_contrast",
    "combat", "kbet_rejection_rate", "effect_size_preservation",
    "make_tree",
]
__version__ = "0.1.0"
