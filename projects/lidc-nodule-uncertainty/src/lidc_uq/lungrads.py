"""Lung-RADS category assignment and operating-point analysis.

The clinical consequence of a nodule assessment is not a probability, it is a
**management category**. Lung-RADS v2022 (ACR; "ACR Lung-RADS v2022: Assessment
Categories and Management Recommendations", *J Am Coll Radiol*, 2024) maps nodule
type, size and growth onto categories 1-4X, and each category carries a follow-up
interval. Category 3 means a 6-month CT; 4A means a 3-month CT or PET; 4B/4X means
tissue sampling.

Why that matters for a disagreement study: the size thresholds are **sharp**, and
readers' diameter estimates differ. A nodule whose mean diameter across readers is
5.9 mm sits in category 2 (annual screening) for one reader and category 3
(6-month CT) for another. :func:`category_disagreement` quantifies exactly how often
reader disagreement crosses a management boundary, and that is a far more legible
quantity for a clinical audience than rater variance in arbitrary units.

Implemented here: baseline-screen solid, part-solid and non-solid rules by mean
diameter, plus the growth-based new/growing rules. Deliberately **not** implemented:
the S modifier, category 4X upgrade on suspicious morphology, and the airway-nodule
rules — all require findings that LIDC does not record, and inventing them would
make the downstream numbers untraceable. Anything the rules cannot determine returns
category ``"0"`` (incomplete) rather than a guess.

Sizes are **mean diameter** (the average of long and short axis), which is what
Lung-RADS v2022 specifies; using a maximum diameter instead shifts categories
upward and is a common source of non-comparable results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "NoduleType",
    "LungRadsCategory",
    "assign_lung_rads",
    "assign_lung_rads_batch",
    "category_disagreement",
    "management_from_category",
    "operating_point_analysis",
    "SOLID_THRESHOLDS_MM",
]

LOG = logging.getLogger(__name__)

NoduleType = Literal["solid", "part_solid", "non_solid", "unknown"]
LungRadsCategory = Literal["0", "1", "2", "3", "4A", "4B"]

#: Baseline-screen solid-nodule mean-diameter boundaries (mm) for Lung-RADS v2022.
SOLID_THRESHOLDS_MM: dict[str, float] = {"cat3": 6.0, "cat4A": 8.0, "cat4B": 15.0}

#: Follow-up recommendation per category, for the operating-point analysis.
_MANAGEMENT: dict[str, str] = {
    "0": "incomplete - additional imaging or prior comparison needed",
    "1": "continue annual screening (12-month LDCT)",
    "2": "continue annual screening (12-month LDCT)",
    "3": "6-month LDCT",
    "4A": "3-month LDCT or PET/CT",
    "4B": "chest CT with or without contrast, PET/CT, and/or tissue sampling",
}

#: Categories that trigger short-interval follow-up or workup, i.e. a positive screen.
POSITIVE_CATEGORIES: frozenset[str] = frozenset({"3", "4A", "4B"})


def management_from_category(category: str) -> str:
    """Return the Lung-RADS follow-up recommendation for a category.

    Args:
        category: A Lung-RADS category string.

    Returns:
        The recommendation text, or a note that the category is unrecognised.
    """
    return _MANAGEMENT.get(str(category), f"unrecognised category {category!r}")


def assign_lung_rads(
    mean_diameter_mm: float,
    *,
    nodule_type: NoduleType = "solid",
    solid_component_mm: float | None = None,
    is_baseline: bool = True,
    is_new: bool = False,
    growing: bool = False,
    benign_features: bool = False,
) -> str:
    """Assign a Lung-RADS v2022 category from size and type.

    Args:
        mean_diameter_mm: Mean of long and short axis, in mm. Non-finite or
            non-positive returns ``"0"``.
        nodule_type: ``"solid"``, ``"part_solid"``, ``"non_solid"`` or ``"unknown"``.
            ``"unknown"`` returns ``"0"``, because guessing the type would silently
            change the thresholds applied.
        solid_component_mm: Solid-component mean diameter for part-solid nodules.
            Required for part-solid categorisation above 6 mm total.
        is_baseline: Baseline screen (``True``) or an annual repeat (``False``).
        is_new: Nodule is new since the prior screen.
        growing: Nodule has grown since the prior screen.
        benign_features: Complete/central calcification or fat, i.e. a benign
            pattern, which assigns category 2 regardless of size.

    Returns:
        One of ``"0"``, ``"1"``, ``"2"``, ``"3"``, ``"4A"``, ``"4B"``.

    Raises:
        ValueError: If ``nodule_type`` is not recognised.
    """
    if nodule_type not in ("solid", "part_solid", "non_solid", "unknown"):
        raise ValueError(f"unknown nodule_type {nodule_type!r}")
    d = float(mean_diameter_mm) if mean_diameter_mm is not None else float("nan")
    if not np.isfinite(d) or d <= 0:
        return "0"
    if nodule_type == "unknown":
        return "0"
    if benign_features:
        return "2"

    if nodule_type == "non_solid":
        # Ground-glass nodules: category 2 below 30 mm, category 3 at or above.
        return "2" if d < 30.0 else "3"

    if nodule_type == "part_solid":
        if d < 6.0:
            return "2"
        if solid_component_mm is None or not np.isfinite(solid_component_mm):
            # Total >= 6 mm but the solid component is unmeasured: the category
            # genuinely cannot be determined.
            return "0"
        s = float(solid_component_mm)
        if not is_baseline and (is_new or growing):
            return "4B" if s >= 4.0 else "4A"
        if s < 6.0:
            return "3"
        if s < 8.0:
            return "4A"
        return "4B"

    # Solid nodules.
    if not is_baseline and (is_new or growing):
        if is_new:
            if d < 4.0:
                return "2"
            if d < 6.0:
                return "3"
            if d < 8.0:
                return "4A"
            return "4B"
        if d < 8.0:
            return "4A"
        return "4B"

    if d < SOLID_THRESHOLDS_MM["cat3"]:
        return "2"
    if d < SOLID_THRESHOLDS_MM["cat4A"]:
        return "3"
    if d < SOLID_THRESHOLDS_MM["cat4B"]:
        return "4A"
    return "4B"


def assign_lung_rads_batch(
    diameters_mm: Sequence[float] | np.ndarray,
    *,
    nodule_type: NoduleType = "solid",
    **kwargs: object,
) -> np.ndarray:
    """Vectorised wrapper around :func:`assign_lung_rads`.

    Args:
        diameters_mm: Mean diameters.
        nodule_type: Applied to every nodule.
        **kwargs: Forwarded to :func:`assign_lung_rads`.

    Returns:
        (n,) array of category strings.
    """
    return np.array(
        [assign_lung_rads(float(d), nodule_type=nodule_type, **kwargs) for d in np.ravel(diameters_mm)],  # type: ignore[arg-type]
        dtype=object,
    )


@dataclass
class CategoryDisagreement:
    """Reader-level Lung-RADS category disagreement for one nodule.

    Attributes:
        nodule_id: Identifier.
        categories: Category assigned from each reader's own diameter.
        n_distinct: Number of distinct categories across readers.
        crosses_positive_boundary: Whether readers disagree about whether the screen
            is positive (category 3 or above) -- the disagreement that changes
            management, and the recommended headline statistic.
        modal_category: Most frequent category, ties broken toward the higher one
            (the conservative clinical choice).
        max_category: Highest category any reader would assign.
    """

    nodule_id: str
    categories: list[str]
    n_distinct: int
    crosses_positive_boundary: bool
    modal_category: str
    max_category: str


_CATEGORY_ORDER: dict[str, int] = {"0": -1, "1": 0, "2": 1, "3": 2, "4A": 3, "4B": 4}


def category_disagreement(
    nodule_id: str,
    reader_diameters_mm: Sequence[float],
    *,
    nodule_type: NoduleType = "solid",
    **kwargs: object,
) -> CategoryDisagreement:
    """Assign a Lung-RADS category per reader and summarise the disagreement.

    Args:
        nodule_id: Identifier.
        reader_diameters_mm: One diameter estimate per reader. Non-finite entries
            are dropped.
        nodule_type: Nodule type, applied to every reader.
        **kwargs: Forwarded to :func:`assign_lung_rads`.

    Returns:
        A :class:`CategoryDisagreement`.

    Raises:
        ValueError: If no finite diameters are supplied.
    """
    d = np.asarray(reader_diameters_mm, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError(f"nodule {nodule_id}: no finite reader diameters")

    cats = [assign_lung_rads(float(v), nodule_type=nodule_type, **kwargs) for v in d]  # type: ignore[arg-type]
    positive = [c in POSITIVE_CATEGORIES for c in cats]
    counts = pd.Series(cats).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    modal = max(top, key=lambda c: _CATEGORY_ORDER.get(str(c), -1))
    return CategoryDisagreement(
        nodule_id=nodule_id,
        categories=cats,
        n_distinct=len(set(cats)),
        crosses_positive_boundary=any(positive) and not all(positive),
        modal_category=str(modal),
        max_category=max(cats, key=lambda c: _CATEGORY_ORDER.get(c, -1)),
    )


def operating_point_analysis(
    probabilities: np.ndarray,
    reference_label: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
    target_sensitivities: Sequence[float] = (0.90, 0.95, 0.98),
    prevalence: float | None = None,
) -> pd.DataFrame:
    """Sensitivity, specificity and false-positive rate across decision thresholds.

    Screening operates at high sensitivity, so the metric that matters is the
    **false-positive rate at fixed sensitivity**, not AUC. NLST reported a
    false-positive rate around 96% of positive screens at its operating point, so
    the practical question for any model is how much of that can be removed without
    losing a cancer.

    ``prevalence`` lets the reported PPV be re-expressed for a screening population
    rather than for the enriched LIDC-style cohort the model was evaluated on --
    without it, PPV from a cohort where ~30% of nodules are malignant is
    meaningless for a screening programme where ~1% are.

    Args:
        probabilities: (n,) predicted probability of malignancy.
        reference_label: (n,) binary reference label.
        thresholds: Explicit decision thresholds. When ``None``, every distinct
            predicted value is used, plus the thresholds achieving each of
            ``target_sensitivities``.
        target_sensitivities: Sensitivities to report thresholds for.
        prevalence: Target population prevalence for the adjusted PPV column.

    Returns:
        Dataframe with ``threshold``, ``n_positive``, ``sensitivity``,
        ``specificity``, ``fpr``, ``ppv``, ``npv``, ``ppv_at_prevalence`` and
        ``meets_target_sensitivity``.

    Raises:
        ValueError: If lengths disagree, a class is absent, or ``prevalence`` is
            outside (0, 1).
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(reference_label).astype(int).ravel()
    if p.size != y.size:
        raise ValueError(f"length mismatch: {p.size} vs {y.size}")
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both classes present, got {n_pos} positive and {n_neg} negative")
    if prevalence is not None and not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie in (0, 1)")

    if thresholds is None:
        candidates = sorted(set(np.round(p, 6).tolist()))
        extra = [_threshold_for_sensitivity(p, y, s) for s in target_sensitivities]
        thresholds = sorted(set(candidates + [t for t in extra if np.isfinite(t)]))

    rows: list[dict[str, object]] = []
    for t in thresholds:
        pred = p >= t
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        tn = int(((~pred) & (y == 0)).sum())
        sens = tp / n_pos
        spec = tn / n_neg
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        row: dict[str, object] = {
            "threshold": float(t),
            "n_positive": int(tp + fp),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "fpr": float(1.0 - spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "meets_target_sensitivity": bool(
                any(sens >= s - 1e-9 for s in target_sensitivities)
            ),
        }
        if prevalence is not None:
            num = sens * prevalence
            den = num + (1.0 - spec) * (1.0 - prevalence)
            row["ppv_at_prevalence"] = float(num / den) if den > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _threshold_for_sensitivity(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Highest threshold that still achieves at least ``target`` sensitivity."""
    positives = np.sort(p[y == 1])
    if positives.size == 0 or not 0.0 < target <= 1.0:
        return float("nan")
    # Keeping a fraction `target` of positives means cutting at the (1-target) quantile.
    return float(np.quantile(positives, max(0.0, 1.0 - target)))
