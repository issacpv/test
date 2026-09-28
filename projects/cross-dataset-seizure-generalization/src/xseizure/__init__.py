"""xseizure: cross-dataset scalp-EEG seizure detection benchmark utilities.

Modules
-------
io_edf        unified EDF loading + montage harmonization to a common bipolar set
splits        patient-wise / record-wise / window-wise splitters and leakage audit helpers
features      spectral and Riemannian (tangent-space) baseline features + logistic baseline
scoring       event-based scorers: OVLP, TAES (NEDC-style) and SzCORE-style, plus sample metrics
domain_shift  MMD, PSD divergence and Riemannian covariance distance between cohorts
"""
from . import domain_shift, features, io_edf, scoring, splits  # noqa: F401

__all__ = ["io_edf", "splits", "features", "scoring", "domain_shift"]
__version__ = "0.1.0"
