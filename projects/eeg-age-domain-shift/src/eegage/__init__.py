"""eegage: age as a continuous domain variable for scalp-EEG seizure detection.

Modules
-------
ages       age parsing (TUH EDF headers, CHB-MIT SUBJECT-INFO, Siena subject_info), bins and log-age distance
features   spectral window features and an age-conditioned normative scaler (label-free adaptation)
transfer   age-bin transfer matrix, decay-vs-distance fit and data-value curves
drift      MMD / centroid drift between age bins, age probes on embeddings, age-importance weights
"""
from . import ages, drift, features, transfer  # noqa: F401

__all__ = ["ages", "features", "transfer", "drift"]
__version__ = "0.1.0"
