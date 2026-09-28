"""amyloid_mri: audit of structural-MRI models for amyloid-PET positivity.

Modules
-------
oasis_tables    OASIS-3 clinical / PET (Centiloid) / MR-session loaders and the subject table builder.
roi_features    FreeSurfer stats parsing, ICV normalization, residualization, ComBat harmonization.
models          Covariate baseline, ROI elastic-net / GBM models, nested subject-grouped CV, calibration
                and bootstrap metrics.
decision_curve  Net benefit, screening policies at fixed sensitivity, PET-scans-avoided cost model.
"""
from . import decision_curve, models, oasis_tables, roi_features  # noqa: F401

__all__ = ["oasis_tables", "roi_features", "models", "decision_curve"]
__version__ = "0.1.0"
