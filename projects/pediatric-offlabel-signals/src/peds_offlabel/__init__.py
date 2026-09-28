"""peds_offlabel: age-off-label paediatric adverse-event signals from FAERS + SPL labels.

Modules
-------
openfda           rate-limited openFDA client with cursor pagination
age               FAERS age normalisation, ICH E11 bands, report flattening
label_ages        regex extraction of labelled paediatric age floors from SPL
                  ``pediatric_use`` / ``indications_and_usage`` sections
offlabel_signals  report classification, age-band-specific below-floor vs
                  on-label signal tables, seriousness model, segmented
                  Poisson ITS for labelling-change natural experiments
"""

from .age import AGE_GROUP_CODES, BANDS, band_from_years, flatten_report, is_pediatric, pediatric_band
from .label_ages import classify_age, extract_age_floor, floor_from_label
from .offlabel_signals import age_stratum, class_shares_by_period, classify_reports, offlabel_signal_table, ror, segmented_poisson_its, seriousness_model
from .openfda import OpenFDAClient, read_jsonl, write_jsonl

__all__ = [
    "OpenFDAClient",
    "read_jsonl",
    "write_jsonl",
    "AGE_GROUP_CODES",
    "BANDS",
    "band_from_years",
    "pediatric_band",
    "is_pediatric",
    "flatten_report",
    "extract_age_floor",
    "floor_from_label",
    "classify_age",
    "classify_reports",
    "age_stratum",
    "offlabel_signal_table",
    "seriousness_model",
    "segmented_poisson_its",
    "class_shares_by_period",
    "ror",
]

__version__ = "0.1.0"
