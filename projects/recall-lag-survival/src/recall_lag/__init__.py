"""recall_lag: survival analysis of the lag between MAUDE adverse-event signals and device recalls.

Modules
-------
openfda_client   Rate-limited openFDA client (cursor pagination, counts, bulk manifest) and record flattening.
linkage          Device keys (product code x firm), recall tables, left-truncated time-to-event construction,
                 lag decomposition.
signals          Monthly counts, first serious event, Poisson CUSUM signal dates, lead-time / false-alarm curves.
survival         Kaplan-Meier (Greenwood CI, left truncation), log-rank, Cox PH (statsmodels PHReg), RMST.
"""

from . import linkage, openfda_client, signals, survival  # noqa: F401

__all__ = ["linkage", "openfda_client", "signals", "survival"]
__version__ = "0.1.0"
