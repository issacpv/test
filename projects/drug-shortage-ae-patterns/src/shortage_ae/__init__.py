"""shortage_ae: drug shortages as staggered treatments on FAERS error-type reporting.

Modules
-------
openfda_client   thin, retrying openFDA client (count queries, date-windowed record paging)
shortage_panel   exposure episodes, name normalisation, MedDRA outcome groups, drug x month panel
staggered_did    Poisson event study, Callaway-Sant'Anna-style ATTs, placebo permutation
"""

from .openfda_client import OpenFDAClient, build_faers_search
from .shortage_panel import (
    DOSING_ERROR_PTS,
    MEDICATION_ERROR_PTS,
    NEGATIVE_CONTROL_PTS,
    SUPPLY_ISSUE_PTS,
    build_panel,
    daily_counts_to_monthly,
    episodes_from_openfda,
    find_substitutes,
    normalize_name,
)
from .staggered_did import (
    callaway_santanna_att,
    event_study_poisson,
    placebo_permutation,
)

__all__ = [
    "OpenFDAClient",
    "build_faers_search",
    "MEDICATION_ERROR_PTS",
    "DOSING_ERROR_PTS",
    "SUPPLY_ISSUE_PTS",
    "NEGATIVE_CONTROL_PTS",
    "normalize_name",
    "episodes_from_openfda",
    "daily_counts_to_monthly",
    "build_panel",
    "find_substitutes",
    "event_study_poisson",
    "callaway_santanna_att",
    "placebo_permutation",
]

__version__ = "0.1.0"
