"""atlas_stability: atlas-robustness of connectome-based phenotype prediction.

Modules
-------
parcellate : parcellation of vertex/voxel series, FC, projection of node values to a common space
predict    : ridge / CPM with grouped nested CV, out-of-fold predictions, Haufe patterns
robustness : accuracy dispersion, prediction and finding concordance, ARI, selection inflation
simulate   : vertex-level generator with a ground-truth parcellation and phenotype-coupled edges
"""

from .parcellate import (
    parcellate_timeseries,
    fisher_z_fc,
    vectorize_upper,
    node_to_space,
    edge_weights_to_node_strength,
    majority_network_assignment,
    aggregate_nodes_to_networks,
    random_contiguous_parcellation,
)
from .predict import CVResult, ridge_cv_predict, cpm_cv_predict, prediction_accuracy, haufe_pattern
from .robustness import (
    accuracy_dispersion,
    prediction_concordance,
    finding_concordance,
    atlas_robustness_index,
    posthoc_selection_inflation,
)
from .simulate import simulate_vertex_dataset

__all__ = [
    "parcellate_timeseries",
    "fisher_z_fc",
    "vectorize_upper",
    "node_to_space",
    "edge_weights_to_node_strength",
    "majority_network_assignment",
    "aggregate_nodes_to_networks",
    "random_contiguous_parcellation",
    "CVResult",
    "ridge_cv_predict",
    "cpm_cv_predict",
    "prediction_accuracy",
    "haufe_pattern",
    "accuracy_dispersion",
    "prediction_concordance",
    "finding_concordance",
    "atlas_robustness_index",
    "posthoc_selection_inflation",
    "simulate_vertex_dataset",
]
