"""szforecast: open-data EEG + ECG seizure forecasting benchmark utilities.

Modules
-------
ecg_hrv        R-peak detection, RR cleaning and windowed HRV features from a raw ECG channel
windows        pre-ictal / inter-ictal window labelling (SPH / SOP), patient-wise splits, seizure-time surrogates
forecast_eval  firing-power alarms, sensitivity / time-in-warning / FPR, chance level and improvement over chance
fusion         per-modality logistic baselines, late/early fusion, covariate residualization
"""
from . import ecg_hrv, forecast_eval, fusion, windows  # noqa: F401

__all__ = ["ecg_hrv", "windows", "forecast_eval", "fusion"]
__version__ = "0.1.0"
