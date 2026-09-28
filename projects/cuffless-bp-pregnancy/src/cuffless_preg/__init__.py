"""cuffless_preg: pregnancy-shift audit of cuffless blood-pressure estimation.

Modules
-------
signals    beat detection (ECG R-peaks, PPG feet, ABP beats), PAT/PTT, SQI, morphology
cohort     ICD-based pregnancy / HDP flags and matched-control selection
ptt_model  per-subject and mixed-effects PAT-BP calibration, Bramwell-Hill simulator
metrics    ISO 81060-2, BHS and IEEE 1708-style evaluation with subject bootstrap
"""

from . import cohort, metrics, ptt_model, signals

__all__ = ["cohort", "metrics", "ptt_model", "signals"]
__version__ = "0.1.0"
