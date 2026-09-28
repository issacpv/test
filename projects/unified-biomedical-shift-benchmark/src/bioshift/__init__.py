"""bioshift - a WILDS-style, modality-agnostic benchmark interface for distribution shift in biomedical prediction.

The package separates four concerns so that the *same* evaluation code runs on
ICU tabular data, 12-lead ECG, scalp-EEG seizure detection and echocardiographic
EF regression:

- :mod:`bioshift.spec` - typed task / domain / shift-cell definitions and the
  concrete registry composing the four sibling projects.
- :mod:`bioshift.adapters` - the data interface (``load(task, domain, split)``)
  with a synthetic adapter for dry runs and a manifest adapter for real caches.
- :mod:`bioshift.metrics` - discrimination, calibration, subgroup-gap, regression
  and event-detection metrics with group bootstrap.
- :mod:`bioshift.diagnostics` - shift diagnostics (domain-classifier AUC, MMD,
  BBSE label-shift estimate) and the covariate / label / concept gap decomposition.
- :mod:`bioshift.harness` - the run loop, leaderboard tables and the
  cross-modality shift-loss regression.
"""
from .adapters import DomainData, ManifestAdapter, SyntheticAdapter
from .harness import SklearnBinaryModel, leaderboard, negative_control, run_benchmark, run_cell, shift_loss_regression
from .metrics import evaluate
from .spec import Benchmark, DomainSpec, ShiftAxis, ShiftCell, TaskSpec, TaskType, default_benchmark

__all__ = [
    "Benchmark",
    "DomainSpec",
    "ShiftAxis",
    "ShiftCell",
    "TaskSpec",
    "TaskType",
    "default_benchmark",
    "DomainData",
    "SyntheticAdapter",
    "ManifestAdapter",
    "evaluate",
    "SklearnBinaryModel",
    "run_cell",
    "run_benchmark",
    "leaderboard",
    "negative_control",
    "shift_loss_regression",
]

__version__ = "0.1.0"
