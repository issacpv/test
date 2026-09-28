"""label_change: dynamic prediction of FDA safety-related labeling changes from FAERS trajectories.

Modules
-------
dailymed_spl        DailyMed web-services client, LOINC section parsing/diffing, SrLC export parser
openfda_counts      quarterly drug x PT count retrieval from openFDA (count queries only)
faers_trajectories  BCPNN/ROR disproportionality, cumulative 2x2 tables, landmark features, notoriety ratio
landmark_model      landmark dataset, pooled-logistic hazard model, time-dependent AUC, lead times
"""

from .dailymed_spl import (
    SECTION_LOINC,
    DailyMedClient,
    diff_sections,
    extract_sections,
    extract_terms,
    new_terms,
    parse_srlc_export,
)
from .faers_trajectories import (
    bcpnn_ic,
    cumulative_2x2,
    landmark_features,
    notoriety_ratio,
    quarter_index,
    ror,
)
from .landmark_model import (
    build_landmark_dataset,
    fit_pooled_logistic,
    grouped_cv_auc,
    lead_times,
    time_dependent_auc,
)
from .openfda_counts import OpenFDACounts

__all__ = [
    "SECTION_LOINC", "DailyMedClient", "extract_sections", "diff_sections", "extract_terms", "new_terms",
    "parse_srlc_export", "OpenFDACounts", "quarter_index", "bcpnn_ic", "ror", "cumulative_2x2",
    "landmark_features", "notoriety_ratio", "build_landmark_dataset", "fit_pooled_logistic",
    "time_dependent_auc", "grouped_cv_auc", "lead_times",
]

__version__ = "0.1.0"
