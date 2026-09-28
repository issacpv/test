"""pet_multiverse: kinetic-modelling multiverse for reference-tissue PET.

Modules
-------
tacs         BIDS-PET timing, tracer classification, TAC extraction/loading
models       SRTM, SRTM2, Logan reference, MRTM/MRTM2, SUVR + forward simulator
multiverse   specification grid, runner, specification curves, variance decomposition
reliability  ICC, within-subject CV, Bland-Altman for test-retest designs
"""

__all__ = ["tacs", "models", "multiverse", "reliability"]
__version__ = "0.1.0"
