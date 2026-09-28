"""cxr_label_audit — auditing report-derived (weak) chest X-ray labels against image-level truth.

Modules
-------
report_labeler     : transparent rule-based CheXpert-style labeler + report-structure features.
agreement          : noise matrices, kappa/PABAK, subgroup-stratified disagreement, differential-noise models.
noise_simulation   : subgroup-specific label-noise injection, noise-rate estimation, fairness/AUC impact.
local_llm_labeler  : loopback-only LLM labeler wrapper (PhysioNet-compliant), prompt + JSON parsing.
"""

from . import agreement, local_llm_labeler, noise_simulation, report_labeler

__all__ = ["report_labeler", "agreement", "noise_simulation", "local_llm_labeler"]
__version__ = "0.1.0"
