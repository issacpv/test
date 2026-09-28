"""lamlat: laminar latent dynamics in Neuropixels recordings of mouse visual cortex.

Modules
-------
io_allen    Allen cache tables, depth-from-surface, unit/area selection.
layers      CSD computation, L4 sink anchoring, depth -> layer assignment, boundary jitter.
latent      Spike-count matrices, cross-validated FA dimensionality, subspace overlap, reduced-rank regression,
            depth-shuffle nulls and unit-count matching.
partition   Ridge encoding models and stimulus-vs-behaviour variance partitioning.
"""

from . import io_allen, latent, layers, partition  # noqa: F401

__all__ = ["io_allen", "layers", "latent", "partition"]
__version__ = "0.1.0"
