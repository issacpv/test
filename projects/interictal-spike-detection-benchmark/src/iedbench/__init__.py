"""iedbench: cross-modality interictal epileptiform discharge (IED) detection benchmark utilities.

Modules
-------
events      event representation, tolerance-based matching, precision/recall/F1, FP per minute, threshold sweeps
detectors   envelope (Janca-style), matched-filter and morphology-classifier baseline detectors
morphology  IFCN-criteria-inspired morphology features of candidate events (Kural et al., 2020) and spatial field
downstream  spike-rate per channel, SOZ ranking AUROC, detector rank agreement, vigilance-stratified rates
"""
from . import detectors, downstream, events, morphology  # noqa: F401

__all__ = ["events", "detectors", "morphology", "downstream"]
__version__ = "0.1.0"
