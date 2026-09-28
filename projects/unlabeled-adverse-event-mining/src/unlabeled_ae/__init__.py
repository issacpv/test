"""unlabeled_ae: unlabeled FAERS signals and label-lag analysis for US drug labels.

Modules
-------
label_client     openFDA ``drug/label`` client, section parser for SPL safety
                 sections (boxed warning, warnings & precautions, adverse
                 reactions incl. the postmarketing subsection), SPL-XML parser
                 for archived DailyMed versions, DailyMed version history.
term_extractor   Dictionary-based adverse-reaction term extraction with a
                 MedDRA-like seed dictionary, MedDRA ASCII loader (licensed
                 files), negation handling and an optional scispaCy hook.
signal_scan      Disproportionality (ROR / PRR / IC) with a quarterly time scan
                 that dates the first emergence of each drug-event signal, and
                 unlabeled-signal flagging / ranking.
label_lag        Label-change dating (version diffs, FDA SrLC), signal-to-label
                 lag with Kaplan-Meier, and evaluation against the FDA quarterly
                 "potential signals of serious risks" list.
"""

from .label_client import LabelClient, LabelDoc, parse_label, parse_spl_xml_sections, split_subsections  # noqa: F401
from .term_extractor import MedDRADictionary, Mention, extract_terms, extract_from_label, labeled_terms  # noqa: F401
from .signal_scan import (  # noqa: F401
    disproportionality,
    cumulative_pair_tables,
    time_scan,
    scan_all_pairs,
    flag_unlabeled,
    rank_unlabeled,
)
from .label_lag import (  # noqa: F401
    first_labeled_dates,
    compute_label_lag,
    kaplan_meier,
    load_potential_signals,
    evaluate_against_fda,
)

__version__ = "0.1.0"
