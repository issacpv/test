"""faers_ddi: sex-stratified drug-drug interaction signal detection in FAERS.

Modules
-------
openfda        rate-limited openFDA client, pagination, report flattening
ddi_tables     2 x 2 x 2 (drug A x drug B x event) report-level counts
ddi_stats      Omega shrinkage, interaction ROR, RERI, sex-specific
               interaction, empirical-Bayes hierarchical shrinkage, BH
reference_sets curated / mechanism / reporter-flagged DDI reference sets and
               AUROC-type evaluation
"""

from .ddi_stats import additive_excess, benjamini_hochberg, hierarchical_shrinkage, interaction_ror, omega, screen, sex_specific_interaction
from .ddi_tables import candidate_pairs, index_reports, pair_event_table, sex_stratified_tables, three_way_counts
from .openfda import OpenFDAClient, flatten_record, read_jsonl, write_jsonl
from .reference_sets import evaluate_scores, interacting_role_pairs, label_pairs, mechanism_pairs, roc_auc

__all__ = [
    "OpenFDAClient",
    "flatten_record",
    "read_jsonl",
    "write_jsonl",
    "index_reports",
    "three_way_counts",
    "candidate_pairs",
    "pair_event_table",
    "sex_stratified_tables",
    "omega",
    "interaction_ror",
    "additive_excess",
    "sex_specific_interaction",
    "hierarchical_shrinkage",
    "benjamini_hochberg",
    "screen",
    "interacting_role_pairs",
    "mechanism_pairs",
    "label_pairs",
    "roc_auc",
    "evaluate_scores",
]

__version__ = "0.1.0"
