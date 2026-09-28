"""circadian_bias: time-of-day structure in ICU benchmark labels and its effect on model evaluation.

Modules
-------
circular
    Circular statistics for event clock times: resultant vector, Rayleigh test, peak/trough,
    shift-change phase locking with permutation p-values, hour histograms.
label_timing
    Event tables, documentation delays (storetime - charttime), windowed labels at fixed
    prediction times, and the phase-randomisation null with label-flip rates.
clock_ablation
    Clock feature construction, full / no-clock / clock-only model comparison with
    hour-stratified discrimination and calibration.
"""
from . import circular, clock_ablation, label_timing  # noqa: F401

__all__ = ["circular", "label_timing", "clock_ablation"]
__version__ = "0.1.0"
