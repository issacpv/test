"""hemo_aging — resting-state hemodynamic timing (blood-arrival lag, HRF shape) as an aging biomarker.

Modules
-------
lag_mapping     : sLFO cross-correlation lag mapping with iterative reference refinement.
hrf_estimation  : point-process resting-state HRF estimation, HRF parameters, Wiener deconvolution.
normative       : heteroscedastic spline normative model, centiles, hemodynamic-age delta.
stats           : ICC(2,1), partial correlation, attenuation ratio, bootstrap mediation.
"""

from . import hrf_estimation, lag_mapping, normative, stats

__all__ = ["lag_mapping", "hrf_estimation", "normative", "stats"]
__version__ = "0.1.0"
