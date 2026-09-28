"""scannability: who does automated MRI QC exclude?

Modules
-------
participants  harmonize BIDS participants.tsv (age, sex, group) across datasets
iqm           load MRIQC group tables and apply exclusion rules
models        exclusion models, scannability index, elasticity, representation shift
simulate      synthetic multi-dataset corpus for tests
"""

__all__ = ["participants", "iqm", "models", "simulate"]
__version__ = "0.1.0"
