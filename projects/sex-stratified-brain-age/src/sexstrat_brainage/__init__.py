"""sexstrat_brainage: pooled vs sex-stratified brain-age modelling audit.

Modules
-------
freesurfer   FreeSurfer table assembly and head-size (TIV) corrections
models       strategy-aware brain-age estimator with age-bias correction and grouped CV
evaluation   sex-gap, spurious-sex-effect, outcome association, ICC, transport metrics
simulation   lifespan cohort generator with known ground truth
"""

__all__ = ["freesurfer", "models", "evaluation", "simulation"]
__version__ = "0.1.0"
