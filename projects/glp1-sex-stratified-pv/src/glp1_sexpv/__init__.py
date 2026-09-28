"""glp1_sexpv: sex-stratified pharmacovigilance of GLP-1 receptor agonists.

Modules
-------
openfda       rate-limited openFDA client with cursor pagination
cohort        GLP-1 agent/brand matching, indication classification,
              compounded-product flag, active comparators, report flattening
sex_signals   sex-specific 2x2 tables, ratio of RORs with LRT, PT-wide atlas
              with BH, indication-adjusted drug x sex interaction, MH pooling
denominators  MEPS/NHANES-based users by sex, Poisson rates, rate ratios,
              excess-female-reporting statistic, age standardisation
"""

from .cohort import COMPARATORS, GLP1_AGENTS, classify_indication, cohort_summary, flatten_report, is_compounded, match_agent
from .denominators import excess_female_reporting, flag_glp1_fills, poisson_rate, rate_ratio, sex_reporting_rates, standardise_by_age, users_from_fills, weighted_users_by_sex
from .openfda import OpenFDAClient, read_jsonl, write_jsonl
from .sex_signals import adjusted_sex_interaction, aggregate_counts, mh_ror_by_sex, ratio_of_ror, ror, sex_difference_atlas, sex_tables

__all__ = [
    "OpenFDAClient",
    "read_jsonl",
    "write_jsonl",
    "GLP1_AGENTS",
    "COMPARATORS",
    "classify_indication",
    "is_compounded",
    "match_agent",
    "flatten_report",
    "cohort_summary",
    "ror",
    "sex_tables",
    "ratio_of_ror",
    "sex_difference_atlas",
    "aggregate_counts",
    "adjusted_sex_interaction",
    "mh_ror_by_sex",
    "poisson_rate",
    "rate_ratio",
    "sex_reporting_rates",
    "excess_female_reporting",
    "weighted_users_by_sex",
    "flag_glp1_fills",
    "users_from_fills",
    "standardise_by_age",
]

__version__ = "0.1.0"
