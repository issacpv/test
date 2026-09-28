"""synth_transport: does synthetic EHR training data transport across hospitals?

Modules
-------
ontology    Shared encounter-level feature specification for MIMIC-IV, eICU-CRD and Synthea, the
            Synthea CSV loader, and the common feature-matrix builder (with observation-process block).
generators  Dependency-free Gaussian-copula generator, negative/positive-control generators, and
            import-guarded adapters for SDV (CTGAN/TVAE) and synthcity (DDPM).
fidelity    Marginal / correlation / propensity (pMSE) / MMD fidelity and DCR privacy proxies.
transport   TSTR matrix, synthetic-gap vs site-gap decomposition with bootstrap, subgroup metrics.
"""

from . import fidelity, generators, ontology, transport

__all__ = ["fidelity", "generators", "ontology", "transport"]
__version__ = "0.1.0"
