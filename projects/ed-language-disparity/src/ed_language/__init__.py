"""ed_language: language-related disparities in ED triage, process and outcomes on MIMIC-IV-ED.

Modules
-------
cohort         language / race harmonisation, visit table with outcomes
text_features  chief-complaint informativeness, language-barrier and interpreter regexes
disparity      adjusted odds, under-triage by group, exact matching, mediation, model fairness
"""

from . import cohort, disparity, text_features

__all__ = ["cohort", "disparity", "text_features"]
__version__ = "0.1.0"
