"""vent_rl: OPE-reliability audit and guideline-constrained offline RL for mechanical ventilation.

Modules
-------
mdp_builder   DuckDB SQL templates (MIMIC-IV / eICU), 4-h binning, PBW, discretised action grid, trajectories
ope           WIS, PDIS, doubly-robust, tabular / linear FQE, ESS, bootstrap CIs
policies      behaviour cloning, empirical MDP, value iteration, tabular CQL-lite, policy utilities
constraints   ARDSNet lung-protective ventilation tables and action masks
reliability   synthetic ground-truth MDP, site-shift generator, estimator audit, cross-site protocol
"""
from . import constraints, mdp_builder, ope, policies, reliability  # noqa: F401

__version__ = "0.1.0"
