"""tms_target: joint connectivity x E-field target scoring with rigorous spatial nulls.

Modules:
    connectome: HCP normative connectome loading (parcellated and dense CIFTI),
        Fisher-z averaging, seed-based and field-weighted connectivity maps,
        structural-connectivity weighting.
    efield: SimNIBS/ROAST E-field loading (NIfTI, Gmsh .msh), ROI dosing metrics,
        focality, parcellation and E-field-based dose normalization.
    nulls: spin-test and variogram-matched surrogate maps, and the spatial
        correlation test that makes a map comparison interpretable.
    scoring: network dose (field-weighted connectivity), target scoring models and
        the normative-versus-individualized evaluation.

Typical use:
    >>> import numpy as np
    >>> from tms_target import network_dose, score_targets
    >>> conn = np.eye(4) + 0.1
    >>> field = np.array([80.0, 20.0, 5.0, 1.0])
    >>> scores = score_targets(conn, field)
    >>> scores.network_dose_linear.shape
    (4,)
"""

from .connectome import (
    Connectome,
    average_connectomes,
    dense_seed_connectivity,
    fisher_z,
    inverse_fisher_z,
    load_parcel_coords,
    load_parcellated_connectome,
    seed_connectivity,
    structural_to_weights,
)
from .efield import (
    EFieldMap,
    field_weighted_centroid,
    focality,
    load_efield_msh,
    load_efield_nifti,
    normalize_to_motor_threshold,
    parcellate_field,
    roi_dose_metrics,
)
from .nulls import (
    NullResult,
    effective_dof,
    random_rotation,
    spatial_correlation_test,
    spin_surrogates,
    variogram,
    variogram_surrogates,
)
from .scoring import (
    ScoreModel,
    TargetScore,
    compare_normative_vs_individual,
    fit_score_model,
    grouped_cv_predictions,
    linear_weighting,
    network_dose,
    score_targets,
    threshold_weighting,
)

__version__ = "0.1.0"

__all__ = [
    "Connectome",
    "EFieldMap",
    "NullResult",
    "ScoreModel",
    "TargetScore",
    "average_connectomes",
    "compare_normative_vs_individual",
    "dense_seed_connectivity",
    "effective_dof",
    "field_weighted_centroid",
    "fisher_z",
    "fit_score_model",
    "focality",
    "grouped_cv_predictions",
    "inverse_fisher_z",
    "linear_weighting",
    "load_efield_msh",
    "load_efield_nifti",
    "load_parcel_coords",
    "load_parcellated_connectome",
    "network_dose",
    "normalize_to_motor_threshold",
    "parcellate_field",
    "random_rotation",
    "roi_dose_metrics",
    "score_targets",
    "seed_connectivity",
    "spatial_correlation_test",
    "spin_surrogates",
    "structural_to_weights",
    "threshold_weighting",
    "variogram",
    "variogram_surrogates",
    "__version__",
]
