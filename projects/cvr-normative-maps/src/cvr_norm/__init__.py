"""cvr_norm — breath-hold cerebrovascular reactivity (CVR) mapping and lifespan normative charts.

Modules
-------
physio      : end-tidal CO2 extraction, RVT, breath-hold compliance, task regressors.
cvr_model   : lag-optimized voxelwise CVR amplitude / delay GLM (+ simulator).
normative   : heteroscedastic spline normative model with covariates; centiles and z-scores.
reliability : ICC(2,1), within-subject CoV, minimal detectable change, Dice.
"""

from . import cvr_model, normative, physio, reliability

__all__ = ["physio", "cvr_model", "normative", "reliability"]
__version__ = "0.1.0"
