"""rad_delay: radiology report turnaround and ED disposition on MIMIC-IV-Note + MIMIC-IV-ED.

Modules
-------
linkage   exam table (modality, TAT, text markers), report-to-ED-stay linkage, post-exit flags
features  timing features (hour, night, weekend, shift distance) and CXR severity from CheXpert
analysis  descriptives, adjusted models, manual 2SLS, regression discontinuity in time, negative controls
"""

from . import analysis, features, linkage

__all__ = ["analysis", "features", "linkage"]
__version__ = "0.1.0"
