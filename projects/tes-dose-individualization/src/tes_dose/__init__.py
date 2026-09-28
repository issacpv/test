"""tes_dose: individual E-field dose-response analysis for transcranial electrical stimulation.

Modules
-------
openneuro      OpenNeuro GraphQL registry search for tES datasets (paged).
efield         Analytical sphere E-field toy model, ROI dose metrics, current scaling.
spatial_nulls  Spin-test and subject-permutation ("montage") null models.
dose_response  Per-dataset dose-response regression and random-effects pooling.
"""

from . import dose_response, efield, openneuro, spatial_nulls

__all__ = ["dose_response", "efield", "openneuro", "spatial_nulls"]
__version__ = "0.1.0"
