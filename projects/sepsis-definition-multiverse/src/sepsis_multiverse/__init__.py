"""sepsis_multiverse: factorial multiverse of Sepsis-3 operationalisations.

Modules
-------
sofa
    Hourly SOFA component scores with configurable carry-forward and missingness handling.
suspected_infection
    Culture/antibiotic pairing rules with configurable windows; first suspicion time per stay.
multiverse
    ``SepsisSpec`` (all decisions), grid enumeration, sepsis labelling per spec, cohort
    geometry (Jaccard, onset shifts) and main-effects variance decomposition.
benchmark
    Early-warning label construction, leakage-safe feature windows and the
    definition-transfer matrix / model-ranking stability.
"""
from . import benchmark, multiverse, sofa, suspected_infection  # noqa: F401

__all__ = ["sofa", "suspected_infection", "multiverse", "benchmark"]
__version__ = "0.1.0"
