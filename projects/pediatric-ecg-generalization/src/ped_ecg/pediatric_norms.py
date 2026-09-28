"""Age-banded pediatric ECG normal limits and z-scoring.

Approximate age-banded normal ranges (median and 2nd/98th percentile) for heart
rate and the main intervals/amplitudes, structured after Rijnbeek et al. (2001,
Eur Heart J 22:702-711) and Davignon et al. (1980). The numeric values here are
representative reference points to make the z-scoring code runnable; a study run
should transcribe the exact published tables (and cite them) before use.

The point of this module is the *method*: express a child's measured ECG in
age-normalised units so that "abnormal for age" is comparable to the adult model
input, which is the physiologically grounded domain-adaptation feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Age bands (upper bound, years, inclusive-exclusive) after common pediatric tables.
AGE_BANDS = [
    (1 / 12, "neonate"),      # < 1 month
    (0.25, "1-3mo"),
    (0.5, "3-6mo"),
    (1.0, "6-12mo"),
    (3.0, "1-3y"),
    (5.0, "3-5y"),
    (8.0, "5-8y"),
    (12.0, "8-12y"),
    (16.0, "12-16y"),
    (200.0, "adult"),
]

# Representative normal limits per band: {band: {param: (median, p2, p98)}}.
# Units: HR bpm; PR/QRS/QT ms. Values are illustrative reference points.
NORMS: dict[str, dict[str, tuple[float, float, float]]] = {
    "neonate": {"hr": (145, 90, 180), "pr": (100, 80, 140), "qrs": (70, 50, 90), "qt": (280, 240, 320)},
    "1-3mo":   {"hr": (150, 100, 190), "pr": (100, 80, 140), "qrs": (70, 50, 90), "qt": (280, 240, 320)},
    "3-6mo":   {"hr": (140, 100, 180), "pr": (105, 80, 145), "qrs": (72, 50, 95), "qt": (290, 250, 330)},
    "6-12mo":  {"hr": (135, 100, 170), "pr": (110, 85, 145), "qrs": (75, 55, 95), "qt": (300, 260, 340)},
    "1-3y":    {"hr": (120, 90, 150), "pr": (115, 90, 150), "qrs": (78, 55, 100), "qt": (310, 270, 350)},
    "3-5y":    {"hr": (105, 75, 135), "pr": (120, 95, 155), "qrs": (80, 58, 102), "qt": (320, 280, 360)},
    "5-8y":    {"hr": (95, 65, 125), "pr": (130, 100, 165), "qrs": (82, 60, 104), "qt": (330, 290, 370)},
    "8-12y":   {"hr": (85, 60, 110), "pr": (135, 105, 170), "qrs": (85, 62, 108), "qt": (340, 300, 385)},
    "12-16y":  {"hr": (78, 55, 105), "pr": (140, 110, 180), "qrs": (88, 65, 112), "qt": (350, 305, 400)},
    "adult":   {"hr": (70, 50, 95), "pr": (155, 120, 200), "qrs": (92, 70, 110), "qt": (390, 350, 440)},
}


@dataclass(frozen=True)
class NormEntry:
    median: float
    p2: float
    p98: float

    @property
    def sd_approx(self) -> float:
        """Approximate SD from a symmetric 2-98% range (~4.1 SD wide)."""
        return (self.p98 - self.p2) / 4.107


def age_band(age_years: float) -> str:
    """Return the pediatric-norm band label for an age in years."""
    for upper, name in AGE_BANDS:
        if age_years < upper:
            return name
    return "adult"


def zscore_for_age(param: str, value: float, age_years: float) -> float:
    """Z-score a measured ECG parameter against the age-appropriate normal band.

    Positive/large |z| means the value is unusual *for the child's age*.
    """
    band = age_band(age_years)
    norms = NORMS.get(band, NORMS["adult"])
    if param not in norms:
        raise KeyError(f"no norm for parameter '{param}'")
    med, p2, p98 = norms[param]
    entry = NormEntry(med, p2, p98)
    return float((value - entry.median) / (entry.sd_approx + 1e-9))


def age_normalise_frame(df, params=("hr", "pr", "qrs", "qt")):
    """Add ``<param>_z`` columns to a DataFrame with columns age_years + params."""
    out = df.copy()
    for p in params:
        if p in out.columns:
            out[f"{p}_z"] = [
                zscore_for_age(p, v, a) for v, a in zip(out[p], out["age_years"])
            ]
    return out
