"""faers_bias: denominator-aware, bias-adjusted FAERS signal detection.

Modules
-------
openfda_client       Thin, rate-limited client for the openFDA drug/event endpoint
                     (count queries, Link-header / search_after pagination).
its                  Interrupted time series (segmented Poisson / NB regression)
                     for stimulated-reporting analysis around FDA communications.
denominators         Exposure denominators from Medicare Part D and MEPS files and
                     the FAERS <-> denominator merge (reports per 10k users).
disproportionality   ROR / PRR / IC, sex-stratified signals with an interaction
                     test, and bias-adjusted (reporter-type, DSC-window, denominator)
                     disproportionality.
"""

from .openfda_client import OpenFDAClient, QUALIFICATION_LABELS, SEX_LABELS  # noqa: F401
from .disproportionality import (  # noqa: F401
    ror,
    prr,
    information_component,
    sex_stratified_ror,
    sex_interaction_test,
    bias_adjusted_ror,
    benjamini_hochberg,
)
from .its import fit_its, build_monthly_series, placebo_its  # noqa: F401
from .denominators import (  # noqa: F401
    normalize_drug_name,
    load_partd_spending_by_drug,
    aggregate_meps_pmed,
    merge_denominators,
    reporting_rate_ratio,
)

__version__ = "0.1.0"
