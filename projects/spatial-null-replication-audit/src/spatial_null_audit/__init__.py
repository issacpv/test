"""spatial_null_audit: re-analysis of published brain-map correlations under multiple spatial nulls.

Modules
-------
claims   Claim-registry schema, loading/validation, practice-survey summaries, robustness index.
nulls    From-scratch spatial null models on parcellated maps: naive permutation, hemisphere-aware
         spin test (nearest-neighbour or one-to-one), projection-corrected spin, Moran spectral
         randomisation, variogram-matched surrogates; autocorrelation diagnostics; GRF simulator.
audit    Per-claim audit driver (observed r, p under each null, robustness index) and the
         smoothness-matched false-positive calibration experiment.
"""

from . import audit, claims, nulls

__all__ = ["audit", "claims", "nulls"]
__version__ = "0.1.0"
