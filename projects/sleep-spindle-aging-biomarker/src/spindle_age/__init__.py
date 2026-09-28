"""spindle_age: harmonized cross-cohort SO-spindle coupling pipeline and normative modeling.

Modules
-------
io_edf     EDF loading, per-cohort channel harmonization, NSRR/Sleep-EDF hypnogram parsing
detect     YASA spindle/SO detection wrappers with a dependency-free fallback detector
coupling   phase-amplitude and event-locked SO-spindle coupling metrics with surrogate nulls
normative  B-spline quantile-regression normative curves (GAMLSS-like) and centile scoring
transport  site harmonization, leave-one-cohort-out evaluation, paired feature-set comparison
"""
from . import coupling, detect, io_edf, normative, transport  # noqa: F401

__all__ = ["io_edf", "detect", "coupling", "normative", "transport"]
__version__ = "0.1.0"
