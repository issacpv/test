"""ai_device_taxonomy: AI-enabled device failure modes from MAUDE narratives (openFDA).

Modules
-------
openfda_client   Rate-limited openFDA client (cursor pagination, counts, bulk manifest), MDR flattening.
linkage          FDA AI-list <-> MAUDE linkage with tiered name matching and a linkage report.
taxonomy         AI-specific failure-mode taxonomy: rule labeller with evidence, weak-label classifier, kappa.
rates            Device-years, rate ratios, negative-binomial rate models, pre/post software-update counts.
"""

from . import linkage, openfda_client, rates, taxonomy  # noqa: F401

__all__ = ["linkage", "openfda_client", "rates", "taxonomy"]
__version__ = "0.1.0"
