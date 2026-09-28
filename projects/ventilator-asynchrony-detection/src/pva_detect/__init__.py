"""pva_detect: patient-ventilator asynchrony from waveforms, and its footprint in charted ICU data.

Modules
-------
lungsim     single-compartment lung + ventilator state machine that generates labelled asynchronies
breaths     breath segmentation, per-breath features, rule-based ineffective-effort / double-trigger detectors
surrogates  low-resolution (2-min / hourly) surrogate features of asynchrony burden and their calibration
cohort      MIMIC-IV / eICU / HiRID charted-ventilator extraction templates and exposure-outcome analysis
"""
from . import breaths, cohort, lungsim, surrogates

__all__ = ["breaths", "cohort", "lungsim", "surrogates"]
__version__ = "0.1.0"
