"""rgc_prosthesis: type-resolved population models of RGC activation by retinal prostheses.

Modules
-------
swc_morph          SWC parsing, compartmentalisation, morphometrics, synthetic axon/AIS generation
electrode          disk/point electrode potentials (epiretinal / subretinal), pulse waveforms
rgc_model          Fohlmeister-Miller-type multicompartment RGC, implicit tree solver, thresholds
population_stats   type summaries, morphological-determinant mixed models, selectivity, validation
"""

from .electrode import Electrode, Pulse, disk_electrode_potential, point_source_potential
from .population_stats import (
    activation_site,
    compare_with_empirical,
    morphological_determinants,
    selectivity_auc,
    type_threshold_summary,
    variance_fractions,
)
from .rgc_model import RGCCable, RGCParams, threshold_current
from .swc_morph import Compartments, compartmentalize, load_swc, morphometrics, synthesize_axon

__all__ = [
    "load_swc", "compartmentalize", "Compartments", "morphometrics", "synthesize_axon",
    "Electrode", "Pulse", "disk_electrode_potential", "point_source_potential",
    "RGCParams", "RGCCable", "threshold_current",
    "type_threshold_summary", "morphological_determinants", "variance_fractions", "selectivity_auc",
    "compare_with_empirical", "activation_site",
]

__version__ = "0.1.0"
