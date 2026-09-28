"""Echo-View-Shift: cross-vendor echocardiographic view-classification robustness.

Modules
-------
views       : canonical view set + per-dataset label crosswalk, clinical danger weights
acquisition : controlled vendor counterfactuals (histogram match, sector, frame rate)
shift       : per-view drop, vendor discriminability, Shapley decomposition
metrics     : clinically weighted confusion metric and downstream EF-impact
"""

from . import views, acquisition, shift, metrics

__all__ = ["views", "acquisition", "shift", "metrics"]
__version__ = "0.1.0"
