"""nm_scaling: batch-corrected, phylogenetically-aware scaling laws of dendritic morphology.

Modules
-------
api_client   NeuroMorpho.org REST API v1 client (paginated metadata, morphometry, SWC download)
swc          SWC parser and morphometrics (length, branch points, Sholl, hull volume, dimensionality)
mixed_models Allometric mixed-effects models with laboratory (archive) random effects
phylo_gls    Newick parsing, Brownian covariance, Pagel's lambda and phylogenetic GLS
"""

from .swc import SWCTree, morphometrics, parse_swc_text, read_swc, sholl_profile  # noqa: F401

__all__ = ["SWCTree", "morphometrics", "parse_swc_text", "read_swc", "sholl_profile"]
__version__ = "0.1.0"
