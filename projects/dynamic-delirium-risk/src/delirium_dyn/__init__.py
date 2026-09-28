"""delirium_dyn: sedation-aware, multi-state dynamic delirium risk from MIMIC-IV / eICU assessments.

Modules
-------
assessments   Parse CAM-ICU / RASS chart rows, build 12-h window states, simulate synthetic stays.
landmark      Landmark datasets with three label schemes; delirium/coma-free days.
features      Sedative-exposure, RASS-trajectory and assessment-process features.
models        Landmark classifiers, transition-intensity regression, IPAW, sedation-leakage ablation.
"""

from . import assessments, features, landmark, models  # noqa: F401

STATES = ("normal", "delirium", "coma", "unscreened", "discharged", "dead")
__all__ = ["assessments", "features", "landmark", "models", "STATES"]
__version__ = "0.1.0"
