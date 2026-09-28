"""ECG-Probe: probing and concept-erasure toolkit for ECG foundation-model embeddings.

Modules
-------
embeddings : frozen-model embedding extraction with a deterministic mock encoder
probing    : linear/MLP/MDL probes with control-task selectivity
erasure    : LEACE closed-form and INLP linear concept erasure
fairness   : subgroup performance gaps and the leakage->inequity link
stats      : permutation nulls, bootstrap CIs, DeLong AUROC test
"""

from . import embeddings, probing, erasure, fairness, stats

__all__ = ["embeddings", "probing", "erasure", "fairness", "stats"]
__version__ = "0.1.0"
