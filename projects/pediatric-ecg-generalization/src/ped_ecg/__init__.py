"""Ped-ECG-Gen: age-continuous transportability of adult-trained ECG models.

Modules
-------
pediatric_norms : Rijnbeek (2001) age-banded normal limits + z-scoring
labels          : age-comparable label crosswalk (adult SCP/SNOMED <-> pediatric)
decay           : AUROC-vs-age curve, bootstrap bands, age-permutation null
adaptation      : shift decomposition and label-efficiency of adaptation methods
"""

from . import pediatric_norms, labels, decay, adaptation

__all__ = ["pediatric_norms", "labels", "decay", "adaptation"]
__version__ = "0.1.0"
