"""bci_transfer: calibration-transfer benchmark for motor-imagery BCIs.

Modules
-------
covariance      : SPD geometry (sqrt/log/exp, AIRM & log-Euclidean means, tangent space),
                  Euclidean alignment, Riemannian re-centering, MDM classifier.
learning_curves : calibration-budget protocol, inverse-power learning-curve fits,
                  trials-to-target and calibration-savings estimands.
mixed_effects   : long-table construction, mixed models, paired effect sizes,
                  negative-transfer rate, dataset-level meta-regression.
synthetic       : synthetic multi-subject two-class SPD data with subject shifts.
"""
from .covariance import (
    covariances,
    sqrtm_spd,
    invsqrtm_spd,
    logm_spd,
    expm_spd,
    airm_distance,
    riemannian_mean,
    log_euclidean_mean,
    tangent_space,
    euclidean_alignment,
    riemannian_alignment,
    MDM,
)
from .learning_curves import calibration_curve, fit_learning_curve, trials_to_fraction, calibration_savings, auc_learning_curve
from .mixed_effects import paired_effect_size, negative_transfer_rate, fit_transfer_mixed_model, meta_regression, holm
from .synthetic import make_subject, make_multi_subject_dataset

__all__ = [
    "covariances", "sqrtm_spd", "invsqrtm_spd", "logm_spd", "expm_spd", "airm_distance", "riemannian_mean",
    "log_euclidean_mean", "tangent_space", "euclidean_alignment", "riemannian_alignment", "MDM",
    "calibration_curve", "fit_learning_curve", "trials_to_fraction", "calibration_savings", "auc_learning_curve",
    "paired_effect_size", "negative_transfer_rate", "fit_transfer_mixed_model", "meta_regression", "holm",
    "make_subject", "make_multi_subject_dataset",
]
__version__ = "0.1.0"
