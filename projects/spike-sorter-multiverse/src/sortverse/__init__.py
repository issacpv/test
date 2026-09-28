"""sortverse: a spike-sorter x curation multiverse with downstream scientific claims as outcomes.

Modules
-------
raw_io       Byte-range arithmetic and readers for Allen raw ``spike_band.dat``; Neuropixels 1.0 geometry.
agreement    Spike-train matching, Hungarian unit pairing, consensus/orphan labelling, QC metrics.
downstream   Responsiveness, OSI/DSI, noise correlations, representational drift, waveform duration.
multiverse   Specification grids, variance decomposition, specification curves, claim stability.
"""

from . import agreement, downstream, multiverse, raw_io  # noqa: F401

__all__ = ["raw_io", "agreement", "downstream", "multiverse"]
__version__ = "0.1.0"
