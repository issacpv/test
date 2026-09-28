"""art_damping: arterial-line damping at scale from recorded fast-flush tests and routine beats.

Modules
-------
transfer  second-order catheter-transducer model, Gardner adequacy classification, synthetic true ABP and flush tests
flush     detection of recorded fast-flush square waves and estimation of fn / zeta from the ringing
sqi       flush-free beat morphology features (ringing, dP/dt, notch, HF power) and a damping classifier
impact    NIBP-ABP discrepancy, hypotension-threshold enrichment and event-rate ratios by damping class
"""
from . import flush, impact, sqi, transfer

__all__ = ["flush", "impact", "sqi", "transfer"]
__version__ = "0.1.0"
