"""device_recall: predicate-chain risk and MAUDE early warning for device recalls.

Modules
-------
openfda_device    Rate-limited client for every openFDA device endpoint (MAUDE
                  ``device/event``, ``device/recall``, ``device/enforcement``,
                  ``device/510k``, ``device/pma``, ``device/classification``,
                  ``device/udi``) plus record flatteners and the recall<->class join.
entity_linking    Manufacturer / device-name normalisation and blocked fuzzy
                  linking of MAUDE reports to premarket submissions.
predicate_graph   Predicate K-number extraction from 510(k) summaries, networkx
                  predicate graph, predicate depth and ancestor-recall exposure.
survival_text     Landmark time-to-recall dataset, MAUDE text features (TF-IDF /
                  optional sentence-transformers) and Cox models (lifelines, with a
                  statsmodels fallback) with temporal validation.
"""

from .openfda_device import DeviceClient, flatten_maude, flatten_510k, flatten_recall, flatten_enforcement, join_recall_class  # noqa: F401
from .entity_linking import normalize_firm, normalize_device_name, similarity, blocked_fuzzy_match, device_key  # noqa: F401
from .predicate_graph import (  # noqa: F401
    extract_predicates,
    build_predicate_graph,
    remove_invalid_edges,
    predicate_depth,
    ancestor_recall_exposure,
    graph_features,
    summary_pdf_url,
)
from .survival_text import (  # noqa: F401
    build_survival_table,
    maude_landmark_features,
    tfidf_features,
    fit_cox,
    concordance_index,
    temporal_split_evaluate,
)

__version__ = "0.1.0"
