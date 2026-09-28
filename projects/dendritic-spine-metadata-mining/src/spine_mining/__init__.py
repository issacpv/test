"""spine_mining: method-annotated dendritic spine density database from literature + NeuroMorpho.

Modules
-------
swc_spines       : SWC parsing, spine-like structure detection, per-file audit and density.
text_extraction  : regex grammar for spine-density statements, unit normalisation, context tags.
europepmc        : Europe PMC REST client (search with cursorMark, full-text XML) and XML -> text.
meta_regression  : random-effects meta-regression of log density on method/biology moderators
                   with lab clustering, permutation null for the method R^2.
"""
from .swc_spines import read_swc, detect_spines, audit_file, audit_tree
from .text_extraction import extract_statements, normalise_density, tag_context
from .europepmc import EuropePMC, xml_to_text
from .meta_regression import meta_regression, method_r2_permutation, prepare_effects

__all__ = [
    "read_swc", "detect_spines", "audit_file", "audit_tree",
    "extract_statements", "normalise_density", "tag_context",
    "EuropePMC", "xml_to_text",
    "meta_regression", "method_r2_permutation", "prepare_effects",
]
__version__ = "0.1.0"
