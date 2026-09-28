"""nursing_risk: dynamic hospital-acquired pressure-injury risk from nursing assessments in MIMIC-IV.

Modules
-------
items     d_items resolution by label regex with verification; Braden total reconstruction
labels    ICD and nursing-documentation HAPI labels; label-definition multiverse and agreement
dynamic   landmark dataset construction with leakage guards; landmark supermodel and static baseline
evaluate  dynamic AUROC / calibration by landmark, stay-level bootstrap, observation-process null, treatment-aware summaries
"""

from . import dynamic, evaluate, items, labels

__all__ = ["dynamic", "evaluate", "items", "labels"]
__version__ = "0.1.0"
