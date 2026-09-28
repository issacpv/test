"""echo_uq: shift-aware conformal uncertainty for video-based ejection fraction.

Modules
-------
video      AVI -> array loading, clip sampling, normalisation, synthetic beating-LV videos, image descriptors
simpson    LV volume / EF from segmentation masks (Simpson's method of disks, single-plane and biplane),
           ED/ES selection and beat-to-beat EF series from mask-area curves
conformal  split, locally adaptive and weighted (covariate-shift) conformal intervals + coverage/width metrics
shift      dataset-shift diagnostics (domain-classifier AUC, MMD, descriptor shift cards, weight ESS)
"""
from . import conformal, shift, simpson, video

__all__ = ["conformal", "shift", "simpson", "video"]
__version__ = "0.1.0"
