"""gwas_ct: cell-type and spatial-domain enrichment of GWAS using the Allen Brain Cell Atlas.

Modules
-------
abc_loader   manifest-driven ABC Atlas download + anndata/pseudobulk helpers
specificity  EWCE-style specificity, top-decile gene sets, hierarchy aggregation, spatial domains
magma_io     MAGMA / S-LDSC input writers and output parsers
scdrs_lite   per-cell polygenic scoring with expression-matched control gene sets
enrichment   bootstrap / permutation enrichment tests, FDR, estimator concordance
"""
from . import abc_loader, enrichment, magma_io, scdrs_lite, specificity  # noqa: F401

__version__ = "0.1.0"
