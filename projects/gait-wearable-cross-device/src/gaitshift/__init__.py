"""gaitshift: cross-device / cross-placement generalisation benchmark for Parkinson's gait.

Modules
-------
harmonize  Dataset registry, unit conversion, resampling, gravity removal, orientation-invariant channels, windowing.
features   Gait features (cadence, stride regularity, harmonic ratio, freeze index, VGRF stride timing) and a synthetic gait generator.
benchmark  Leave-one-dataset-out evaluation, in-distribution reference, deployment gap, shift diagnostics.
"""

from . import benchmark, features, harmonize

__all__ = ["benchmark", "features", "harmonize"]
__version__ = "0.1.0"
