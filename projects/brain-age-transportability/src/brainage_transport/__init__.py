"""brainage_transport: transportability audit of brain-age delta corrections.

Modules
-------
freesurfer_stats
    Parse FreeSurfer ``aseg.stats`` / ``?h.aparc.stats`` tables and the
    ``asegstats2table`` / ``aparcstats2table`` outputs into feature tables.
harmonize
    ComBat-style empirical-Bayes site/scanner harmonisation with separate
    ``fit`` and ``transform`` steps (needed for transport to unseen cohorts),
    plus a thin wrapper around ``neuroCombat`` when it is installed.
brainage
    Ridge and gradient-boosting brain-age baselines with leakage-safe
    cross-validation.
bias_correction
    Age-bias ("regression to the mean") corrections: Beheshti (2019),
    de Lange & Cole (2020), Cole (2018), Smith (2019) quadratic variant.
longitudinal
    Mixed-model evaluation of within-person delta change against clinical
    trajectories (CDR / MMSE) and test-retest reliability.
"""

from .freesurfer_stats import (  # noqa: F401
    parse_stats_file,
    load_aseg_features,
    load_aparc_features,
    build_feature_table,
)
from .harmonize import ComBat  # noqa: F401
from .brainage import BrainAgeModel, cross_validated_predictions  # noqa: F401
from .bias_correction import BiasCorrector, age_bias_slope  # noqa: F401
from .longitudinal import (  # noqa: F401
    fit_delta_trajectory_model,
    within_person_change,
    icc_test_retest,
    simulate_longitudinal_cohort,
)

__version__ = "0.1.0"
