"""QT-Dose: real-world dose-response and time-course of QT-interval prolongation.

Estimate, within patients, the change in rate-corrected QT (QTc) per unit dose
of QT-prolonging drugs from MIMIC-IV-ECG intervals linked to eMAR administration
records, with a multiverse over correction formulae and measurement choices, and
concordance against CredibleMeds and FAERS.

Modules
-------
corrections : heart-rate correction formulae (Bazett/Fridericia/... + fitted alpha)
exposure    : normalise eMAR administrations to canonical drug/dose events
linkage     : pair pre-dose (baseline) and post-dose ECGs per administration
models      : within-patient dose-response and time-course (hysteresis) models
external    : CredibleMeds / FAERS concordance and disproportionality
"""

from . import corrections, exposure, linkage, models

__all__ = ["corrections", "exposure", "linkage", "models"]
__version__ = "0.1.0"
