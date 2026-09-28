"""dbs_popvta: population variability of DBS activation thresholds with empirical axon morphologies.

Modules
-------
lead_field    DBS lead geometries, point-source contact fields, E-field magnitude, optional NIfTI fields
axon_model    CRRSS/Sweeney myelinated axon on a 3-D path, implicit tree solver, thresholds by bisection
morph_paths   SWC axon-path extraction, descriptors, scaling and random placement around a lead
population    threshold sweeps, probabilistic VTA radius, factorial (Sobol/ANOVA) variance decomposition
"""

from .axon_model import CRRSSParams, MyelinatedAxon, Pulse, activating_function, resample_path, threshold_current
from .lead_field import LEADS, LeadField, LeadSpec, point_source_potential
from .morph_paths import axon_paths, load_swc, path_descriptors, place_path, straight_fiber
from .population import (
    activation_probability_curve,
    efield_isoline_radius,
    probabilistic_vta_radius,
    threshold_sweep,
    variance_decomposition,
)

__all__ = [
    "LEADS", "LeadSpec", "LeadField", "point_source_potential",
    "CRRSSParams", "MyelinatedAxon", "Pulse", "activating_function", "resample_path", "threshold_current",
    "load_swc", "axon_paths", "path_descriptors", "place_path", "straight_fiber",
    "threshold_sweep", "activation_probability_curve", "probabilistic_vta_radius", "efield_isoline_radius",
    "variance_decomposition",
]

__version__ = "0.1.0"
