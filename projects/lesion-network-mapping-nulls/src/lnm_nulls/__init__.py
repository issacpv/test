"""lnm_nulls — null models for lesion network mapping, evaluated with known ground truth.

Modules
-------
lesion_sampling : lesion features, synthetic spheres, location shuffling, anatomically matched resampling.
lnm             : parcel-level functional lesion-network maps, structural disconnection maps, symptom association.
nulls           : label permutation, synthetic-lesion nulls, bias atlases, prior-leakage R², Moran spectral surrogates.
simulation      : ground-truth simulation engine returning type-I error / power / specificity per null family.
"""

from . import lesion_sampling, lnm, nulls, simulation

__all__ = ["lesion_sampling", "lnm", "nulls", "simulation"]
__version__ = "0.1.0"
