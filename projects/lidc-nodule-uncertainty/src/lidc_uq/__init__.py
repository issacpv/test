"""lidc_uq: multi-rater aware calibration and disagreement prediction for lung nodules.

Modules:
    annotations: multi-rater LIDC-IDRI aggregation -- soft labels, disagreement
        metrics, consensus masks, and an optional pylidc loader.
    features: numpy/scipy image features chosen to predict reader disagreement
        (boundary sharpness, attenuation heterogeneity, threshold proximity), with
        an optional pyradiomics backend.
    calibration: soft-label training routes, calibration metrics and the Brier
        decomposition, grouped cross-validation, and the soft-versus-majority
        comparison.
    disagreement: disagreement prediction and its operational evaluation as a
        second-read triage tool.
    lungrads: Lung-RADS v2022 category assignment, reader-level category
        disagreement, and screening operating-point analysis.

Typical use:
    >>> from lidc_uq import disagreement_metrics, soft_label_from_ratings
    >>> ratings = [2, 3, 4, 5]
    >>> round(soft_label_from_ratings(ratings, scheme="split_3"), 3)
    0.625
    >>> round(disagreement_metrics(ratings)["binary_disagreement"], 3)
    0.75
"""

from .annotations import (
    MALIGNANCY_SCALE,
    SEMANTIC_FEATURES,
    NoduleConsensus,
    RaterAnnotation,
    build_nodule_table,
    consensus_mask,
    disagreement_metrics,
    load_pylidc_nodules,
    segmentation_disagreement,
    soft_label_from_ratings,
)
from .calibration import (
    CalibrationReport,
    SoftLabelTrainer,
    brier_decomposition,
    calibration_metrics,
    compare_label_schemes,
    expand_soft_labels,
    grouped_cv_soft,
    reliability_curve,
)
from .disagreement import (
    DISAGREEMENT_TARGETS,
    DisagreementModel,
    compare_to_size_baseline,
    fit_disagreement_model,
    recall_at_k,
    selective_prediction_curve,
    triage_curve,
)
from .features import (
    SIMPLE_FEATURE_NAMES,
    extract_pyradiomics_features,
    extract_simple_features,
    feature_frame,
    resample_to_isotropic,
)
from .lungrads import (
    POSITIVE_CATEGORIES,
    SOLID_THRESHOLDS_MM,
    assign_lung_rads,
    assign_lung_rads_batch,
    category_disagreement,
    management_from_category,
    operating_point_analysis,
)

__version__ = "0.1.0"

__all__ = [
    "CalibrationReport",
    "DISAGREEMENT_TARGETS",
    "DisagreementModel",
    "MALIGNANCY_SCALE",
    "NoduleConsensus",
    "POSITIVE_CATEGORIES",
    "RaterAnnotation",
    "SEMANTIC_FEATURES",
    "SIMPLE_FEATURE_NAMES",
    "SOLID_THRESHOLDS_MM",
    "SoftLabelTrainer",
    "assign_lung_rads",
    "assign_lung_rads_batch",
    "brier_decomposition",
    "build_nodule_table",
    "calibration_metrics",
    "category_disagreement",
    "compare_label_schemes",
    "compare_to_size_baseline",
    "consensus_mask",
    "disagreement_metrics",
    "expand_soft_labels",
    "extract_pyradiomics_features",
    "extract_simple_features",
    "feature_frame",
    "fit_disagreement_model",
    "grouped_cv_soft",
    "load_pylidc_nodules",
    "management_from_category",
    "operating_point_analysis",
    "recall_at_k",
    "reliability_curve",
    "resample_to_isotropic",
    "segmentation_disagreement",
    "selective_prediction_curve",
    "soft_label_from_ratings",
    "triage_curve",
    "__version__",
]
