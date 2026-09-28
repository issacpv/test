"""biosim_ae: biosimilar vs originator adverse-event profiles in FAERS + labels.

Modules
-------
openfda   rate-limited openFDA client with cursor pagination
products  US biosimilar catalogue, product attribution (brand > suffixed INN >
          originator brand > INN-only), attributability summaries
profiles  comparator-restricted ROR, JS-divergence permutation test,
          launch-aligned windows, Weber descriptors, PT panels, MH ROR
labels    SPL section retrieval, PT-in-label matching, label similarity
"""

from .labels import fetch_label, label_similarity, labelled_flags, pt_in_label, unlabelled_fraction
from .openfda import OpenFDAClient, read_jsonl, write_jsonl
from .products import FAMILIES, attributability_summary, attribute_drug_entry, attribute_report, launch_dates
from .profiles import PANELS, calendar_window, comparator_ror, jensen_shannon, launch_curve, panel_comparison, profile_distance_test, ror, stratified_ror, weber_index

__all__ = [
    "OpenFDAClient",
    "read_jsonl",
    "write_jsonl",
    "FAMILIES",
    "attribute_drug_entry",
    "attribute_report",
    "attributability_summary",
    "launch_dates",
    "PANELS",
    "ror",
    "comparator_ror",
    "jensen_shannon",
    "profile_distance_test",
    "calendar_window",
    "launch_curve",
    "weber_index",
    "panel_comparison",
    "stratified_ror",
    "fetch_label",
    "pt_in_label",
    "labelled_flags",
    "unlabelled_fraction",
    "label_similarity",
]

__version__ = "0.1.0"
