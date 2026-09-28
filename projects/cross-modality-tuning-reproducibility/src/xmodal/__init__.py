"""xmodal: decomposing two-photon vs Neuropixels differences in visual tuning.

Modules
-------
tuning         Trial x condition responses; OSI/DSI (vector and ratio), preferences, lifetime sparseness,
               responsiveness.
forward_model  Spikes -> GCaMP fluorescence (kernel, supralinearity, saturation, noise) and AR(1) deconvolution.
reliability    Split-half / test-retest reliability, Spearman-Brown, correction for attenuation.
multiverse     Distribution-shift statistics, propensity reweighting, sequential gap decomposition, grids.
"""

from . import forward_model, multiverse, reliability, tuning  # noqa: F401

__all__ = ["tuning", "forward_model", "reliability", "multiverse"]
__version__ = "0.1.0"
