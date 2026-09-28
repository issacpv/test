"""ecg_xgen: cross-dataset / population-shift benchmarking of 12-lead ECG classifiers.

Modules
-------
loaders   multi-dataset WFDB / HDF5 loading with resampling and lead-order harmonisation
labels    SNOMED-CT label harmoniser with explicit mapping policies (strict / lenient / superclass)
models    handcrafted-feature baseline (numpy / sklearn) and an optional 1D-ResNet (PyTorch)
shift     domain-shift diagnostics (domain-classifier AUC, MMD, spectral fingerprints, density-ratio weights)
metrics   subgroup AUROC/parity with stratified bootstrap CIs and label-noise-aware evaluation
"""
from . import labels, loaders, metrics, models, shift

__all__ = ["labels", "loaders", "metrics", "models", "shift"]
__version__ = "0.1.0"
