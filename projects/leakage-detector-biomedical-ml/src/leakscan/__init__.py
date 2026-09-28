"""leakscan - static (AST-based) detection of data-leakage patterns in biomedical ML code.

Rules implemented in :mod:`leakscan.detector`:

- ``split-without-groups``: a random record/window-level split on data that shows
  subject-level structure, with no group-aware splitter anywhere in the file.
- ``window-level-random-split``: as above, with explicit sliding-window / epoch /
  segment vocabulary (the classic EEG/ECG leakage mechanism).
- ``preprocess-before-split``: a scaler / imputer / decomposition fitted on the
  full data before the split.
- ``feature-selection-before-split``: SelectKBest / RFE / correlation filtering on
  the full data before the split.
- ``resample-before-split``: SMOTE-style over/under-sampling before the split.
- ``test-set-in-training``: test-named arrays used in ``fit`` / ``validation_data``
  / ``eval_set`` / model selection.
- ``no-holdout-evaluation``: an estimator fitted and evaluated on the same
  arrays with no split anywhere.
- ``group-aware-split-present`` / ``pipeline-present``: positive evidence
  (severity ``info``).
"""
from .detector import Finding, ScanResult, scan_file, scan_path, scan_source
from .notebooks import notebook_to_source
from .report import findings_to_frame, prevalence_table, summarise_repo, to_markdown

__all__ = [
    "Finding",
    "ScanResult",
    "scan_source",
    "scan_file",
    "scan_path",
    "notebook_to_source",
    "findings_to_frame",
    "summarise_repo",
    "prevalence_table",
    "to_markdown",
]

__version__ = "0.1.0"
