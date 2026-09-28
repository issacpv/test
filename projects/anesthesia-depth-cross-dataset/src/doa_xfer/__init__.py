"""doa_xfer: cross-agent / cross-dataset depth-of-anaesthesia evaluation beyond BIS.

Modules
-------
vitaldb_io     VitalDB case selection (agent arms), track loading/alignment, EEG epoching, frontal channel picks
eeg_features   spectral features incl. aperiodic exponent (FOOOF-lite), SEF95, burst-suppression ratio, alpha/delta
targets        BIS lag alignment, age-adjusted MAC fraction, normalized propofol Ce, anaesthesia phase labels
transfer_eval  prediction probability Pk, Lin's CCC, Bland-Altman, agent-transfer tables, age strata, nulls
"""
from . import eeg_features, targets, transfer_eval, vitaldb_io  # noqa: F401

__all__ = ["vitaldb_io", "eeg_features", "targets", "transfer_eval"]
__version__ = "0.1.0"
