"""deid_audit: aggregate, local-only audit of residual identifier *patterns* in de-identified notes.

Modules
-------
sections   MIMIC-IV-Note / MIMIC-III template section splitter and placeholder (``___`` / ``[** **]``) statistics
patterns   regex detectors for HIPAA Safe Harbor identifier classes; findings carry category/section/length only
estimate   rates per 10k notes with Wilson CIs, capture-recapture (Lincoln-Petersen, Chapman), planted-PHI recall
           calibration, small-cell suppression
audit      per-note count rows (regex + optional offline spaCy NER), group summaries, CLI

Nothing in this package returns, stores or prints matched text.
"""
from . import audit, estimate, patterns, sections

__all__ = ["audit", "estimate", "patterns", "sections"]
__version__ = "0.1.0"
