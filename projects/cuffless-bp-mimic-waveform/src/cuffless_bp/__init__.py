"""cuffless_bp: leakage-free evaluation of cuffless blood-pressure estimation.

Modules
-------
wfdb_loader  streaming 10-s ECG/PPG/ABP windows from PhysioNet (MIMIC-III/IV waveforms), VitalDB, PulseDB
sqi          signal-quality indices for PPG, ECG and ABP windows
beats        R-peak / PPG fiducial / ABP beat detection and PAT/PTT feature extraction
evaluation   subject-independent splits, leakage audit, AAMI/ISO 81060-2, IEEE 1708, BHS, change tracking
models       Moens-Korteweg PTT baseline, feature ridge, small 1D-CNN and hybrid (torch optional)
"""

from . import beats, evaluation, models, sqi, wfdb_loader

__all__ = ["beats", "evaluation", "models", "sqi", "wfdb_loader"]
__version__ = "0.1.0"
