"""lab_leak: how much ICU outcome prediction is *lab ordering* rather than lab values.

Modules
-------
simulate     semi-synthetic cohort with tunable ordering informativeness and ordering-policy shift
ordering     order-pattern (mask) vs value feature builder, schedule annotation, thinning/standardisation
attribution  two-player Shapley value/mask attribution, thinning stress curves, mask-dropout training, intensity reweighting
sql          DuckDB extraction templates for MIMIC-IV (labevents + poe) and eICU-CRD (lab + patient)
"""
from . import attribution, ordering, simulate, sql

__all__ = ["attribution", "ordering", "simulate", "sql"]
__version__ = "0.1.0"
