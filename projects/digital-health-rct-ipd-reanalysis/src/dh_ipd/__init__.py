"""dh_ipd - individual-participant-data re-analysis of digital-health RCTs.

Modules
-------
- :mod:`dh_ipd.registry` - ClinicalTrials.gov v2 API client: find digital-health
  RCTs, extract their IPD-sharing statements, classify the named repository.
- :mod:`dh_ipd.engagement` - ITT, naive per-protocol, Wald/CACE and
  principal-score stratified engagement effects with bootstrap CIs.
- :mod:`dh_ipd.missing` - attrition sensitivity analyses: complete case, MAR
  multiple imputation, jump-to-reference, delta-adjusted tipping points.
- :mod:`dh_ipd.hte` - heterogeneity of treatment effect with internal-external
  cross-validation across trials.
- :mod:`dh_ipd.simulate` - a multi-trial IPD simulator with engagement
  confounding, informative dropout and effect modification.
"""
from .engagement import bootstrap_estimates, cace_wald, itt, naive_per_protocol, principal_score_effects
from .hte import fit_interaction_model, internal_external_cv, predict_benefit
from .missing import complete_case, multiple_imputation, rubin_combine, tipping_point
from .registry import build_params, classify_repository, fetch_studies, flatten_study, is_digital_intervention, sharing_summary, studies_to_frame
from .simulate import simulate_ipd

__all__ = [
    "itt",
    "naive_per_protocol",
    "cace_wald",
    "principal_score_effects",
    "bootstrap_estimates",
    "complete_case",
    "multiple_imputation",
    "tipping_point",
    "rubin_combine",
    "fit_interaction_model",
    "predict_benefit",
    "internal_external_cv",
    "build_params",
    "fetch_studies",
    "flatten_study",
    "classify_repository",
    "is_digital_intervention",
    "studies_to_frame",
    "sharing_summary",
    "simulate_ipd",
]

__version__ = "0.1.0"
