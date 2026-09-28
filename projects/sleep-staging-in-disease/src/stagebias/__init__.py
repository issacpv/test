"""stagebias: automated sleep-staging error as a measurement-error problem for sleep epidemiology.

Modules
-------
hypnogram          Parse NSRR profusion XML and EDF+ annotations into 30-s epoch stages and event masks.
features           Per-epoch spectral features and a transparent gradient-boosting stager with posteriors.
error_structure    Confusion matrices by stratum, kappa, N3% bias, event-locked and differential error tests.
measurement_error  Cox/logistic outcome models with error-prone exposures; regression calibration and SIMEX.
"""

from . import error_structure, features, hypnogram, measurement_error  # noqa: F401

__all__ = ["hypnogram", "features", "error_structure", "measurement_error"]
__version__ = "0.1.0"
