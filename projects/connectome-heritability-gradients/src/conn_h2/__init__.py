"""conn_h2: twin heritability of cortical functional organisation.

Modules
-------
gradients     Diffusion-map embedding of FC (own implementation + BrainSpace
              wrapper), Procrustes alignment, gradient dispersion.
features      Region-wise SC-FC coupling (correlation and multilinear
              communication models) and dynamic-FC state occupancy.
heritability  Twin-pair construction from HCP restricted tables, Falconer,
              DeFries-Fulker regression, maximum-likelihood ACE, bootstrap CIs,
              reliability correction, regional heritability maps.
spatial_nulls Spin permutations for parcellated maps, spatial correlation
              tests, random-gene-set nulls and an abagen wrapper.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import features, gradients, heritability, spatial_nulls  # noqa: E402,F401

__all__ = ["gradients", "features", "heritability", "spatial_nulls", "__version__"]
