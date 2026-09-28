"""icu_transport: multi-site transportability, shift decomposition and fairness for ICU models.

Modules
-------
cohorts      DuckDB SQL templates + shared feature ontology for MIMIC-IV, eICU-CRD, HiRID, AUMCdb
shift        covariate / label / concept shift estimation and performance-gap decomposition
calibration  ECE, calibration slope/intercept, decision curves, few-shot recalibration
fairness     subgroup parity metrics with bootstrap CIs and permutation nulls
"""

from . import calibration, cohorts, fairness, shift

__all__ = ["calibration", "cohorts", "fairness", "shift"]
__version__ = "0.1.0"
