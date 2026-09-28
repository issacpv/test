"""vmb_validation: multimodal validation of connectome-based whole-mouse-brain models.

Modules
-------
connectome  : load / normalise / symmetrise / threshold structural connectomes (the multiverse factors).
models      : linear (OU) model with analytic covariance; Hopf (Stuart-Landau) simulation; BOLD and
              calcium forward filters; coupling sweeps.
fc_metrics  : FC, FC similarity, homotopic FC, FCD and KS distance.
nulls       : degree-preserving rewiring, distance-preserving weight permutation, weight shuffle.
parcellate  : region time series from 4-D volumes or widefield frame stacks.
"""

__all__ = ["connectome", "models", "fc_metrics", "nulls", "parcellate"]
__version__ = "0.1.0"
