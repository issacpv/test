"""morph2ephys: predicting intrinsic electrophysiology from dendritic morphology.

Modules
-------
allen_fetch     allensdk Cell Types fetcher (cells, ephys/morphology feature tables, SWC, NWB)
swc_graph       SWC parser, SWC -> graph converter (torch_geometric-ready, numpy fallback), morphometrics
ephys_features  Long-square ephys feature extractor (R_in, sag, tau, rheobase, AP width, adaptation) + simulator
baseline        Ridge baseline with grouped CV, permutation nulls, transfer metrics, within-type partial R2
gnn             Plain-PyTorch message-passing regressor on neuron graphs (optional torch import)
"""

from .swc_graph import NeuronGraph, morphometric_vector, parse_swc_text, read_swc, swc_to_graph  # noqa: F401

__all__ = ["NeuronGraph", "morphometric_vector", "parse_swc_text", "read_swc", "swc_to_graph"]
__version__ = "0.1.0"
