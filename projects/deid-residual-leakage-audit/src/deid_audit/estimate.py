"""Rate estimation, capture-recapture, small-cell suppression and detector calibration.

* Note-level residual *rates* per 10,000 notes with Wilson intervals.
* Two-detector capture-recapture (Lincoln-Petersen and the bias-corrected
  Chapman estimator) to estimate the number of notes with a residual that
  *neither* detector found, from the overlap of two (approximately independent)
  detectors such as regex vs local NER.
* Recall calibration by *planting* clearly fictitious identifiers into clean
  text: observed candidate counts are divided by measured recall.
* ``suppress_small_cells`` enforces the aggregate-only reporting rule.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


@dataclass(frozen=True)
class RateEstimate:
    k: int
    n: int
    rate_per_10k: float
    low_per_10k: float
    high_per_10k: float


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (works for k = 0)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return (low, high)


def rate_per_10k(k: int, n: int) -> RateEstimate:
    lo, hi = wilson_interval(k, n)
    return RateEstimate(k, n, 1e4 * k / n if n else float("nan"), 1e4 * lo, 1e4 * hi)


@dataclass(frozen=True)
class CaptureRecapture:
    n1: int
    n2: int
    m: int
    estimate: float
    se: float
    low: float
    high: float
    method: str


def lincoln_petersen(n1: int, n2: int, m: int) -> CaptureRecapture:
    """Classic estimator N = n1 n2 / m (undefined when m = 0). Variance per Seber (1982)."""
    if m <= 0:
        return CaptureRecapture(n1, n2, m, float("inf"), float("nan"), float("nan"), float("nan"), "lincoln_petersen")
    N = n1 * n2 / m
    var = n1 * n2 * (n1 - m) * (n2 - m) / (m ** 3)
    se = math.sqrt(var)
    return CaptureRecapture(n1, n2, m, N, se, N - Z95 * se, N + Z95 * se, "lincoln_petersen")


def chapman(n1: int, n2: int, m: int) -> CaptureRecapture:
    """Bias-corrected Chapman (1951) estimator N = (n1+1)(n2+1)/(m+1) - 1; finite for m = 0."""
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    se = math.sqrt(max(var, 0.0))
    return CaptureRecapture(n1, n2, m, N, se, N - Z95 * se, N + Z95 * se, "chapman")


def capture_recapture_from_flags(flags_a: Sequence[int], flags_b: Sequence[int]) -> CaptureRecapture:
    """Chapman estimate of the number of positive notes from two detectors' note-level flags."""
    a, b = np.asarray(flags_a, bool), np.asarray(flags_b, bool)
    return chapman(int(a.sum()), int(b.sum()), int((a & b).sum()))


def suppress_small_cells(df: pd.DataFrame, count_cols: Iterable[str], min_count: int = 10) -> pd.DataFrame:
    """Replace counts in ``(0, min_count)`` with NaN so no small cell is ever reported."""
    out = df.copy()
    for c in count_cols:
        v = out[c].astype(float)
        out[c] = v.where((v == 0) | (v >= min_count), np.nan)
    return out


# --- recall calibration by planting fictitious identifiers --------------------------------

FAKE_NAMES = ("John Doe", "Jane Roe", "Alex Example", "Sam Placeholder", "Pat Fictitious")
FAKE_STREETS = ("Main Street", "Oak Avenue", "Elm Road", "Park Boulevard")
FAKE_INSTITUTIONS = ("Example General Hospital", "Placeholder Medical Center")


def plant_synthetic_phi(text: str, rng: np.random.Generator, n_per_category: int = 1,
                        categories: Sequence[str] | None = None) -> tuple[str, Counter]:
    """Insert obviously fictitious identifiers into ``text`` at random sentence boundaries.

    Uses reserved fictional ranges (555-01xx phones, example.com emails, 19xx/20xx dates) so the
    planted material can never coincide with a real person. Returns the text and the ground-truth
    counter of planted items per category.
    """
    generators: dict[str, Callable[[], str]] = {
        "phone": lambda: f"({rng.integers(200, 999)}) 555-01{rng.integers(0, 99):02d}",
        "email": lambda: f"user{rng.integers(1, 999)}@example.com",
        "ssn_like": lambda: f"{rng.integers(100, 999)}-{rng.integers(10, 99)}-{rng.integers(1000, 9999)}",
        "mrn_like": lambda: f"MRN: {rng.integers(100000, 9999999)}",
        "zip": lambda: f"MA {rng.integers(10000, 99999)}",
        "street_address": lambda: f"{rng.integers(1, 9999)} {FAKE_STREETS[rng.integers(len(FAKE_STREETS))]}",
        "date_unshifted_numeric": lambda: f"{rng.integers(1, 12)}/{rng.integers(1, 28)}/{rng.integers(1990, 2022)}",
        "date_unshifted_text": lambda: f"March {rng.integers(1, 28)}, {rng.integers(1990, 2022)}",
        "age_over_89": lambda: f"{rng.integers(90, 105)} year old",
        "name_after_title": lambda: f"Dr. {FAKE_NAMES[rng.integers(len(FAKE_NAMES))]}",
        "name_in_signature": lambda: f"Dictated by: {FAKE_NAMES[rng.integers(len(FAKE_NAMES))]}",
        "institution_name": lambda: FAKE_INSTITUTIONS[rng.integers(len(FAKE_INSTITUTIONS))],
        "provider_id": lambda: f"NPI {rng.integers(1000000000, 9999999999)}",
        "url": lambda: "https://www.example.com/portal",
    }
    cats = list(categories) if categories is not None else list(generators)
    truth: Counter = Counter()
    sentences = text.split(". ")
    for cat in cats:
        for _ in range(n_per_category):
            i = int(rng.integers(0, len(sentences)))
            sentences[i] = sentences[i] + f" {generators[cat]()}"
            truth[cat] += 1
    return ". ".join(sentences), truth


def detector_recall(scan_fn: Callable[[str], Counter], docs: Iterable[tuple[str, Counter]]) -> pd.DataFrame:
    """Per-category planted vs detected counts and recall for a ``scan_fn(text) -> Counter``."""
    planted: Counter = Counter()
    detected: Counter = Counter()
    for text, truth in docs:
        found = scan_fn(text)
        for cat, k in truth.items():
            planted[cat] += k
            detected[cat] += min(found.get(cat, 0), k)
    rows = [{"category": c, "planted": planted[c], "detected": detected[c],
             "recall": detected[c] / planted[c] if planted[c] else float("nan")} for c in planted]
    return pd.DataFrame(rows).sort_values("category", ignore_index=True)


def calibrated_rate(observed: int, recall: float, n_notes: int) -> RateEstimate:
    """Divide an observed candidate count by measured recall before computing the rate."""
    if not (0 < recall <= 1):
        raise ValueError("recall must be in (0, 1]")
    k_adj = int(round(observed / recall))
    return rate_per_10k(min(k_adj, n_notes), n_notes)
