"""Canonical echocardiographic view set and per-dataset label crosswalk.

Coarse canonical views are used for cross-dataset comparison; a finer set aligns
with TTE47 where compatible. ``DANGER_WEIGHTS`` encodes how clinically costly
each confusion is (used by the clinically-weighted error metric): confusing two
apical chamber views (A4C<->A2C), which corrupts EF/strain, is far worse than
confusing a view with 'other'.
"""
from __future__ import annotations

# Coarse canonical views used across all datasets.
CANONICAL_COARSE = ["A4C", "A2C", "A3C", "PLAX", "PSAX", "SUBCOSTAL", "SUPRASTERNAL", "OTHER"]

# Per-dataset raw label -> canonical coarse view.
CROSSWALK = {
    "echonet-dynamic": {"a4c": "A4C"},  # EchoNet-Dynamic is A4C-only
    "camus": {"2ch": "A2C", "4ch": "A4C", "a2c": "A2C", "a4c": "A4C"},
    "tmed2": {
        "plax": "PLAX", "psax": "PSAX", "a2c": "A2C", "a4c": "A4C",
        "a2ch": "A2C", "a4ch": "A4C",
    },
    "tte47": {
        # a representative slice of the 47-view scheme -> coarse
        "apical_4_chamber": "A4C", "apical_2_chamber": "A2C", "apical_3_chamber": "A3C",
        "plax": "PLAX", "psax_av": "PSAX", "psax_mv": "PSAX", "psax_pm": "PSAX",
        "subcostal_4c": "SUBCOSTAL", "suprasternal": "SUPRASTERNAL",
    },
}

# Clinical danger weight for confusing true view i with predicted view j.
# High weight = clinically consequential (wrong downstream measurement).
_HIGH = 3.0
_MED = 1.5
_LOW = 0.5
DANGER_PAIRS = {
    frozenset({"A4C", "A2C"}): _HIGH,
    frozenset({"A4C", "A3C"}): _MED,
    frozenset({"A2C", "A3C"}): _MED,
    frozenset({"A4C", "PLAX"}): _HIGH,
    frozenset({"A2C", "PLAX"}): _HIGH,
    frozenset({"PLAX", "PSAX"}): _LOW,
}


def canonical(dataset: str, raw_label: str) -> str:
    """Map a dataset-specific raw label to the coarse canonical view."""
    d = CROSSWALK.get(dataset, {})
    return d.get(str(raw_label).strip().lower(), "OTHER")


def danger_weight(true_view: str, pred_view: str) -> float:
    """Clinical cost of predicting ``pred_view`` when the truth is ``true_view``."""
    if true_view == pred_view:
        return 0.0
    w = DANGER_PAIRS.get(frozenset({true_view, pred_view}))
    if w is not None:
        return w
    return _LOW  # any other misclassification: low but nonzero cost
