"""oai_prog: knee-OA progression with decomposed uncertainty and competing risks.

Modules
-------
cohort       OAI knee-level cohort, competing-risk event coding, reader disagreement, grouped folds, simulator.
uncertainty  Ensemble uncertainty decomposition, calibration, Mondrian conformal sets, subgroup metrics.
survival     Aalen-Johansen cumulative incidence, cause-specific concordance, IPCW Brier score.
"""

from . import cohort, survival, uncertainty

__all__ = ["cohort", "survival", "uncertainty"]
__version__ = "0.1.0"
