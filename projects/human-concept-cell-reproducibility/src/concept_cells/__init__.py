"""concept_cells: criterion-multiverse reproducibility audit of human concept cells.

Modules
-------
nwb_loader   : NWB -> SessionData container; trial-window spike counts.
selectivity  : five concept-cell criteria, permutation / circular-shift nulls,
               split-half cross-validated selectivity.
prevalence   : hierarchical prevalence estimates, cluster bootstrap, heterogeneity,
               criterion agreement.
synthetic    : synthetic sessions with known selective units, drift and blocking.
"""
from .nwb_loader import SessionData, spike_counts, binned_counts
from .selectivity import (
    anova_criterion,
    kruskal_criterion,
    binwise_ranksum_criterion,
    response_strength_criterion,
    poisson_glm_criterion,
    permutation_pvalue,
    split_half_selectivity,
    run_multiverse,
)
from .prevalence import prevalence_table, cluster_bootstrap_ci, cochran_q, criterion_agreement
from .synthetic import make_synthetic_session

__all__ = [
    "SessionData", "spike_counts", "binned_counts",
    "anova_criterion", "kruskal_criterion", "binwise_ranksum_criterion",
    "response_strength_criterion", "poisson_glm_criterion", "permutation_pvalue",
    "split_half_selectivity", "run_multiverse",
    "prevalence_table", "cluster_bootstrap_ci", "cochran_q", "criterion_agreement",
    "make_synthetic_session",
]
__version__ = "0.1.0"
