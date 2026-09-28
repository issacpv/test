"""disease_morph: lab-effect-corrected meta-analysis of disease/aging neuronal morphologies.

Modules
-------
cohort   : map NeuroMorpho ``experiment_condition`` strings to condition classes and build
           archive-matched case/control tables.
swc      : SWC parsing and morphometrics recomputed with one definition for all archives.
meta     : Hedges' g, random-effects pooling (DL / REML, Hartung-Knapp CI), heterogeneity,
           Egger regression and signature similarity with permutation nulls.
passive  : passive compartmental model (steady-state input resistance, attenuation, electrotonic length).
"""

__all__ = ["cohort", "swc", "meta", "passive"]
__version__ = "0.1.0"
