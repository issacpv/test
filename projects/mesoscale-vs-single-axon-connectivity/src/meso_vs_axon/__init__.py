"""meso_vs_axon: single-neuron axon projections vs. Allen mesoscale (bulk tracer) connectivity.

Modules
-------
allen_connectivity  allensdk MouseConnectivityCache wrapper -> structure-level projection matrix / vectors
ccf_assign          SWC axon nodes/terminals -> CCFv3 structure assignment -> per-neuron target vectors
concordance         Jaccard, weighted rank correlation, AUROC/precision@k, pooled recovery curves, nulls
heterogeneity       Projection heterogeneity index, independent-sampling null, bootstrap CI, rarefaction
"""

from .concordance import concordance_table, jaccard, weighted_rank_correlation  # noqa: F401
from .heterogeneity import heterogeneity_index, independent_sampling_null  # noqa: F401

__all__ = ["concordance_table", "jaccard", "weighted_rank_correlation", "heterogeneity_index",
           "independent_sampling_null"]
__version__ = "0.1.0"
