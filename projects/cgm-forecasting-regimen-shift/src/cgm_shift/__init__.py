"""CGM-Regimen-Shift: glucose forecasting across insulin-delivery regimens.

Modules
-------
loaders  : harmonise CGM/insulin/carb streams to a common 5-min grid + windowing
regimen  : label segments as open-loop / hybrid-closed-loop / DIY-loop
models   : forecasting baselines (persistence, AR, ridge) with exogenous inputs
clinical : hypoglycemia warning sensitivity, lead time, error-grid, calibration
shift    : cross-regimen transfer diagnostics and controller-feedback decomposition
audit    : CGM data-quality artefact detection (gaps, interpolation, calibration)
"""

from . import loaders, regimen, models, clinical, shift, audit

__all__ = ["loaders", "regimen", "models", "clinical", "shift", "audit"]
__version__ = "0.1.0"
