"""npx_drift: cross-area, cross-dataset representational-drift benchmark for Neuropixels data.

Modules
-------
loaders        DANDI/NWB streaming, AllenSDK and IBL adapters, SessionData container, synthetic sessions
quality        unit quality filtering, quality strata, stability covariates, area-group mapping
responses      stimulus-aligned spike binning and trial x unit response matrices, time blocks
drift_metrics  PV correlation vs lag, RDM stability, tuning correlation, cross-time decoding
nulls          time-shuffle, Poisson rate-matched and circular-shift null models
"""
from . import drift_metrics, loaders, nulls, quality, responses  # noqa: F401

__all__ = ["loaders", "quality", "responses", "drift_metrics", "nulls"]
__version__ = "0.1.0"
