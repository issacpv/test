"""morph_stim: population-level morphological determinants of stimulation thresholds.

Modules:
    swc_cable: SWC parsing and a passive compartmental cable model solving the
        quasi-static polarization of a branched morphology in a uniform field.
    thresholds: orientation sweeps and activation-threshold estimation per
        modality waveform (tDCS, tACS, TMS, DBS).
    neuromorpho: NeuroMorpho.Org REST client for building species/region/cell-type
        filtered cohorts of real reconstructions.
    popstats: mixed models, cluster bootstrap and variance partitioning over the
        resulting threshold population.
    neuron_driver: optional NEURON backend used to calibrate the numpy surrogate.

Typical use:
    >>> from morph_stim import CableModel, load_swc, orientation_sweep
    >>> # neuron = load_swc("data/neuromorpho/swc/<name>.CNG.swc")
    >>> # result = orientation_sweep(CableModel(neuron), waveform="TMS")
"""

from .popstats import (
    bootstrap_ratio_ci,
    class_separability,
    fit_threshold_mixed_model,
    prepare_frame,
    variance_components,
)
from .swc_cable import CableModel, PassiveParams, SWCNeuron, load_swc
from .thresholds import (
    MODALITIES,
    Waveform,
    fibonacci_directions,
    orientation_sweep,
    sweep_population,
    threshold_for_directions,
)

__version__ = "0.1.0"

__all__ = [
    "CableModel",
    "MODALITIES",
    "PassiveParams",
    "SWCNeuron",
    "Waveform",
    "bootstrap_ratio_ci",
    "class_separability",
    "fibonacci_directions",
    "fit_threshold_mixed_model",
    "load_swc",
    "orientation_sweep",
    "prepare_frame",
    "sweep_population",
    "threshold_for_directions",
    "variance_components",
    "__version__",
]
