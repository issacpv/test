"""xspindle: harmonised, scale-free slow-oscillation / spindle detection and coupling across species.

Modules
-------
spectrum    Welch PSD, aperiodic (1/f) fit, individualised spindle-peak detection.
detect      Percentile-threshold SO and spindle detectors with duration limits in seconds or cycles.
coupling    SO phase, event-locked circular statistics, surrogates, Tort modulation index.
harmonize   Species presets, resampling, crude NREM scoring, one-call metric vector, detector multiverse.
"""

from . import coupling, detect, harmonize, spectrum  # noqa: F401

__all__ = ["spectrum", "detect", "coupling", "harmonize"]
__version__ = "0.1.0"
