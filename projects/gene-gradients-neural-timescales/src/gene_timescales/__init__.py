"""gene_timescales: gene-expression gradients vs single-neuron timescales and tuning.

Modules
-------
timescales
    Spike-count autocorrelation, exponential timescale fits (with offset and two-timescale
    variants), and tuning metrics (OSI/DSI, latency, adaptation, preferred TF).
gene_maps
    Regional gene-expression matrices from ISH energies or MERFISH cells, PLS/ridge
    prediction of area-level targets with nested CV, and composition-partial R^2.
nulls
    3-D variogram surrogates, hierarchy-stratified permutations, gene-ensemble nulls, BH-FDR.
"""
from . import gene_maps, nulls, timescales  # noqa: F401

__all__ = ["timescales", "gene_maps", "nulls"]
__version__ = "0.1.0"
