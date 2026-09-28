"""pulse_contour: open pulse-contour stroke-volume estimation validated against three reference standards.

Modules
-------
beats       Windkessel synthetic ABP with known SV, slope-sum onset detector, per-beat features
estimators  Liljestrand-Zander, Herd, systolic-area, 2-element Windkessel and fitted impedance-corrected SV; calibration
agreement   Bland-Altman (repeated measures), percentage error, four-quadrant and polar trend concordance, cluster bootstrap
references  echo LVOT parsing, MIMIC d_items lookup for thermodilution CO, VitalDB track download, time alignment
"""
from . import agreement, beats, estimators, references

__all__ = ["agreement", "beats", "estimators", "references"]
__version__ = "0.1.0"
