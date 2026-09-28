"""wmh_amyloid: WMH x amyloid interaction on longitudinal cognition and CDR progression.

Modules
-------
cohort        : OASIS-3 ID parsing, session/PET/clinical matching, amyloid status, progression events
wmh_features  : WMH volumes, periventricular/deep split, lobar fractions, tool agreement
models        : mixed-effects three-way interaction, person-period hazard, additive interaction (RERI)
simulate      : longitudinal cohort generator with known interaction and progression hazard
"""

from .cohort import parse_oasis_id, amyloid_positive, match_nearest, build_longitudinal_table, progression_events
from .wmh_features import (
    wmh_volume_ml,
    periventricular_deep_split,
    lobar_volumes,
    normalize_wmh,
    dice,
    volume_agreement,
)
from .models import (
    fit_interaction_mixed_model,
    person_period,
    discrete_time_hazard,
    additive_interaction,
    bootstrap_additive_interaction,
)
from .simulate import simulate_wmh_cohort

__all__ = [
    "parse_oasis_id",
    "amyloid_positive",
    "match_nearest",
    "build_longitudinal_table",
    "progression_events",
    "wmh_volume_ml",
    "periventricular_deep_split",
    "lobar_volumes",
    "normalize_wmh",
    "dice",
    "volume_agreement",
    "fit_interaction_mixed_model",
    "person_period",
    "discrete_time_hazard",
    "additive_interaction",
    "bootstrap_additive_interaction",
    "simulate_wmh_cohort",
]
