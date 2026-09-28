"""scanner_rd: scanner upgrades as regression-discontinuity natural experiments.

Modules
-------
sessions   : OASIS-3 / ADNI session tables, transition detection, running variables
rdit       : local-linear regression-discontinuity-in-time with subject fixed effects
harmonize  : cross-sectional ComBat, longitudinal ComBat (lite), RD-anchored correction
simulate   : longitudinal cohort generator with known scanner effects
"""

from .sessions import (
    parse_oasis_id,
    build_session_table,
    find_transitions,
    add_event_time,
    find_paired_sessions,
)
from .rdit import RDResult, local_linear_rd, placebo_cutoffs, select_bandwidth_cv, aging_equivalent_years
from .harmonize import ComBat, LongitudinalComBatLite, rd_anchored_correction
from .simulate import simulate_cohort

__all__ = [
    "parse_oasis_id",
    "build_session_table",
    "find_transitions",
    "add_event_time",
    "find_paired_sessions",
    "RDResult",
    "local_linear_rd",
    "placebo_cutoffs",
    "select_bandwidth_cv",
    "aging_equivalent_years",
    "ComBat",
    "LongitudinalComBatLite",
    "rd_anchored_correction",
    "simulate_cohort",
]
