"""cpm_fair: fairness audit of connectome-based predictive modelling.

Modules
-------
fc_loader        Load HCP parcellated time series / CIFTI, build FC matrices,
                 merge unrestricted + restricted subject tables, define subgroups.
predictors       CPM and ridge-on-FC regressors, family-aware cross-validation,
                 within-fold confound residualisation.
subgroup_metrics Per-subgroup accuracy, calibration and error-structure metrics,
                 family-cluster bootstrap and permutation tests for gaps.
reweighting      Inverse-frequency weights, balanced subsampling and a
                 group-DRO ridge baseline.
mediation        Mediation-by-motion analysis and a sequential decomposition
                 of the subgroup performance gap (sample size, motion, label
                 reliability, residual).
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import fc_loader, mediation, predictors, reweighting, subgroup_metrics  # noqa: E402,F401

__all__ = [
    "fc_loader",
    "predictors",
    "subgroup_metrics",
    "reweighting",
    "mediation",
    "__version__",
]
