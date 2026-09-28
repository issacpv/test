"""dmri_charts: model-comparison lifespan normative charts for diffusion microstructure.

Modules
-------
shells     : gradient tables, shell identification, protocol emulation by subsampling
models     : weighted linear DTI and DKI fitters, multi-compartment signal simulator
normative  : location-scale spline normative model (centiles, z-scores, age of peak)
compare    : model-comparison metrics (age sensitivity, ICC, protocol-transfer bias, AUC)
"""

from .shells import read_bvals_bvecs, identify_shells, subsample_protocol, protocol_summary
from .models import fit_dti, fit_dki, simulate_multicompartment_signal, fibonacci_sphere, DTIResult, DKIResult
from .normative import NormativeModel, age_of_peak, bootstrap_age_of_peak, simulate_lifespan_dataset
from .compare import age_explained_variance, icc_2_1, protocol_transfer_bias, deviation_auc, rank_models

__all__ = [
    "read_bvals_bvecs",
    "identify_shells",
    "subsample_protocol",
    "protocol_summary",
    "fit_dti",
    "fit_dki",
    "simulate_multicompartment_signal",
    "fibonacci_sphere",
    "DTIResult",
    "DKIResult",
    "NormativeModel",
    "age_of_peak",
    "bootstrap_age_of_peak",
    "simulate_lifespan_dataset",
    "age_explained_variance",
    "icc_2_1",
    "protocol_transfer_bias",
    "deviation_auc",
    "rank_models",
]
