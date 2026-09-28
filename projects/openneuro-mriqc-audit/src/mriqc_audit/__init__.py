"""mriqc_audit: large-scale image-quality audit across OpenNeuro.

Modules
-------
openneuro_client
    GraphQL client for https://openneuro.org/crn/graphql (dataset listing with
    cursor pagination, snapshot file lists, participants.tsv fetch) and helpers
    to map OpenNeuro's git-annex MD5E keys to MRIQC WebAPI ``md5sum`` values.
mriqc_client
    Paginated client for the MRIQC WebAPI (https://mriqc.nimh.nih.gov/api/v1)
    that flattens records into tidy tables and de-duplicates re-uploads.
normative
    Normative reference charts for IQMs by age / scanner: quantile regression
    on a spline basis and a GAMLSS-style location-scale model.
qc_sensitivity
    "QC exclusion as a hidden multiverse": sweep QC thresholds, compute
    retained n, differential exclusion and effect-size changes per cell.
"""

from .openneuro_client import OpenNeuroClient, parse_annex_key, annex_md5_index  # noqa: F401
from .mriqc_client import MRIQCClient, records_to_frame, dedupe_records  # noqa: F401
from .normative import QuantileNormativeModel, LocationScaleModel  # noqa: F401
from .qc_sensitivity import (  # noqa: F401
    QCRule, apply_rules, qc_multiverse, effect_size, specification_summary,
)

__version__ = "0.1.0"
