"""atrophy_subtypes: longitudinal audit of subtype-and-stage models of regional atrophy.

Modules
-------
longitudinal          Sessions -> subject timelines; amyloid status over time; groups (A-CN, A+CI, A-CI).
zscoring              Control-referenced z-scores with age/sex/ICV regression (abnormal = positive).
ebm                   Compact z-score event-based model with EM subtyping (SuStaIn-like), simulation,
                      pySuStaIn hook.
longitudinal_metrics  Stage monotonicity, subtype stability (with permutation / marginal nulls),
                      stage-vs-clinical mixed models, Holm correction.
"""
from . import ebm, longitudinal, longitudinal_metrics, zscoring  # noqa: F401

__all__ = ["longitudinal", "zscoring", "ebm", "longitudinal_metrics"]
__version__ = "0.1.0"
