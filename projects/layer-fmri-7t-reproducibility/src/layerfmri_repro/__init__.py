"""layerfmri_repro — reproducibility of laminar fMRI profiles across labs, contrasts and pipelines.

Modules
-------
layering        : equidistant / equivolume cortical depth from distance maps; annulus phantom.
profiles        : laminar profile extraction, normalization, shape features, similarity, draining-vein model.
reproducibility : ICC, variance components (lab / subject / residual), chance-corrected index, SNR degradation.
"""

from . import layering, profiles, reproducibility

__all__ = ["layering", "profiles", "reproducibility"]
__version__ = "0.1.0"
