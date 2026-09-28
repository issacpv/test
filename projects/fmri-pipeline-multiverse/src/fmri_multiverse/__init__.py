"""fmri_multiverse: dataset-scale multiverse analysis of task fMRI on OpenNeuro.

Modules
-------
fetch      Dataset registry, OpenNeuro GraphQL / S3 / openneuro-py / DataLad
           fetchers, fMRIPrep-derivative discovery and dataset characteristics.
confounds  fMRIPrep confound strategies (Ciric 2017 / Wang 2024 families),
           implemented on the confounds TSV with an optional nilearn path.
glm        Pipeline specifications, multiverse enumeration, HRF/design
           matrices, NumPy first-level GLM, nilearn wrappers, second level.
speccurve  Specification curves, pipeline-robustness score, variance
           decomposition, bootstrap/sign-flip multiverse inference, fragility model.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import confounds, fetch, glm, speccurve  # noqa: E402,F401

__all__ = ["fetch", "confounds", "glm", "speccurve", "__version__"]
