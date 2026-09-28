"""cardiac_twin: physiology-anchored ECG-echo digital twin from unpaired corpora.

Modules
-------
data          PTB-XL / EchoNet / MIMIC loaders, ECG-echo pairing, patient-level splits.
ecg_features  Synthetic 12-lead generator and feature extraction (HR, QRS, axis, voltage).
twin_model    Interpretable cardiac state, forward models, unpaired alignment, counterfactuals.
evaluation    Regression/classification metrics, cluster bootstrap, calibration, discordance.
"""

from . import data, ecg_features, evaluation, twin_model

__all__ = ["data", "ecg_features", "evaluation", "twin_model"]
__version__ = "0.1.0"
