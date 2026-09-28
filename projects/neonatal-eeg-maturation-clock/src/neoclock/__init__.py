"""neoclock: an open neonatal EEG maturation clock with bias-corrected deviation scores.

Modules
-------
features_neural  maturational EEG features: band powers, spectral slope, rEEG, burst/interburst structure, synchrony
clock            age regression with subject-grouped CV, delta bias correction (Smith et al., 2019), ICC, MAE by bin
normative        per-feature Gaussian centiles vs PMA, z-scores and a multivariate deviation score
outcomes         trend / correlation tests of deviation vs HIE grade and seizure burden with permutation nulls
"""
from . import clock, features_neural, normative, outcomes  # noqa: F401

__all__ = ["features_neural", "clock", "normative", "outcomes"]
__version__ = "0.1.0"
