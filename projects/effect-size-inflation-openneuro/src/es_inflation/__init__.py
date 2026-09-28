"""es_inflation: winner's-curse / effect-size inflation audit for OpenNeuro-linked papers.

Modules
-------
openneuro_client  GraphQL client for the OpenNeuro dataset index (+ GitHub-mirror fallback).
winners_curse     Selection-corrected effect estimators: Type M/S, conditional MLE, empirical Bayes.
effects           Effect-size conversions, funnel asymmetry, and the circular vs cross-validated
                  peak-effect estimator used in the re-analysis subset.
"""

from . import effects, openneuro_client, winners_curse

__all__ = ["effects", "openneuro_client", "winners_curse"]
__version__ = "0.1.0"
