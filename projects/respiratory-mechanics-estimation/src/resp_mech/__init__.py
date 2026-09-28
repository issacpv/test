"""resp_mech: respiratory mechanics estimation from charted ICU ventilator data.

Modules
-------
single_compartment  Equation-of-motion algebra, mechanical power, breath simulation and least-squares fit.
state_space         Kalman filter / RTS smoother for time-varying elastance and resistance from charted rows.
charting            Label mapping, pivoting, unit harmonisation, flow proxy and ventilation-episode detection.
"""

from . import charting, single_compartment, state_space

__all__ = ["charting", "single_compartment", "state_space"]
__version__ = "0.1.0"
