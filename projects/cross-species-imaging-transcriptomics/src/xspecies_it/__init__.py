"""xspecies_it: cross-species (mouse MERFISH vs human AHBA) imaging-transcriptomics replication.

Modules
-------
regional_expression
    Aggregate cell-level MERFISH expression to atlas regions; cell-type composition
    matrices; decomposition of regional gene maps into composition and within-type parts.
nulls
    3-D variogram-matched spatial surrogates, gene-ensemble (matched random gene set)
    nulls and a panel-aware enrichment test.
cross_species
    Ortholog alignment, per-gene association profiles, cross-species replication
    statistics and the conjunction (double-null) replication criterion.
"""
from . import cross_species, nulls, regional_expression  # noqa: F401

__all__ = ["regional_expression", "nulls", "cross_species"]
__version__ = "0.1.0"
