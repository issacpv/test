"""ed_triage: mis-triage equity audit and temporally validated triage prediction on MIMIC-IV-ED.

Modules
-------
cohort         DuckDB SQL + pandas cohort builders reproducing Xie et al. (2022) outcomes with equity extensions
text_features  chief-complaint normalisation, TF-IDF, optional local transformer embeddings
mistriage      outcome-anchored under/over-triage definitions, subgroup rates, adjusted odds ratios
validation     temporal (anchor_year_group) splits, drift metrics, calibration and decision curves
"""
from . import cohort, mistriage, text_features, validation

__all__ = ["cohort", "mistriage", "text_features", "validation"]
__version__ = "0.1.0"
