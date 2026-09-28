"""motion_causal: causal decomposition of head-motion contributions to FC-behaviour prediction.

Modules
-------
fc_features    : framewise displacement, FC matrices, vectorisation
predict        : grouped cross-validated ridge / CPM with per-run predictions
decomposition  : within-subject artifact sensitivity (FE / 2SLS), covariance decomposition,
                 negative-control-exposure contrast, robustness values
nulls          : motion-matched permutation null, bootstrap helpers
simulate       : generative model with known artifact and trait paths
"""

from .fc_features import framewise_displacement, fisher_z_fc, vectorize_upper, load_hcp_movement_regressors
from .predict import RunPredictions, crossval_run_predictions, fit_ridge, cpm_select_edges
from .decomposition import (
    artifact_sensitivity,
    decompose_association,
    negative_control_exposure,
    robustness_value,
)
from .nulls import motion_matched_permutation, permutation_pvalue, bootstrap_ci
from .simulate import simulate_motion_cohort

__all__ = [
    "framewise_displacement",
    "fisher_z_fc",
    "vectorize_upper",
    "load_hcp_movement_regressors",
    "RunPredictions",
    "crossval_run_predictions",
    "fit_ridge",
    "cpm_select_edges",
    "artifact_sensitivity",
    "decompose_association",
    "negative_control_exposure",
    "robustness_value",
    "motion_matched_permutation",
    "permutation_pvalue",
    "bootstrap_ci",
    "simulate_motion_cohort",
]
