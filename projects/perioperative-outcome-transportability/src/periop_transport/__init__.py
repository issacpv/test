"""periop_transport: transportability of preoperative outcome models across health systems.

Modules
-------
cohorts   surgical-encounter definitions (MIMIC-IV multiverse), mortality / ICU / KDIGO-AKI labels
features  harmonised preoperative feature ontology, last-value-before-surgery labs, Charlson
shift     covariate / label / concept shift decomposition, recalibration, few-shot curves
metrics   discrimination, calibration, net benefit, subgroup gaps, bootstrap
"""

from . import cohorts, features, metrics, shift

__all__ = ["cohorts", "features", "metrics", "shift"]
__version__ = "0.1.0"
