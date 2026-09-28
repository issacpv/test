"""Predicting inter-radiologist disagreement, and evaluating it as a triage tool.

This is the project's distinctive claim: reader disagreement is not noise to be
averaged away but a **predictable property of the image**, and a model that
predicts it is directly useful — it is a second-read triage signal. A screening
programme cannot afford a second radiologist on every scan, but it can afford one
on the 10% of nodules where a second reader is most likely to disagree with the
first.

That reframing changes the evaluation entirely. A disagreement model should not be
judged by R^2 against rater variance; it should be judged operationally:

* **Ranking quality** — if we send the top *k*% of nodules by predicted
  disagreement for a second read, what fraction of all actually-disagreed nodules
  do we catch? (:func:`triage_curve`, reported as recall at *k*%, plus the
  area under that curve.)
* **Lift over the obvious baseline** — nodule size alone already predicts
  disagreement, because borderline-size nodules straddle management thresholds.
  A disagreement model earns its keep only by beating size
  (:func:`compare_to_size_baseline`).
* **Does flagged disagreement co-locate with model error?** If the nodules readers
  disagree about are also the ones the malignancy classifier gets wrong, then
  predicted disagreement is a usable selective-prediction signal
  (:func:`selective_prediction_curve`).

The separate-model design (predict malignancy, predict disagreement, combine at the
operating point) is deliberately chosen over a single distributional model. It keeps
the two quantities separately auditable, and it means the disagreement head can be
validated against a different reference — LIDC's four readers — than the malignancy
head, which in a screening cohort is validated against biopsy or follow-up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DisagreementModel",
    "fit_disagreement_model",
    "triage_curve",
    "recall_at_k",
    "compare_to_size_baseline",
    "selective_prediction_curve",
    "DISAGREEMENT_TARGETS",
]

LOG = logging.getLogger(__name__)

#: Candidate disagreement targets from :func:`lidc_uq.annotations.disagreement_metrics`.
#: ``binary_disagreement`` is the recommended primary: it peaks when readers split
#: evenly on the decision that actually changes management.
DISAGREEMENT_TARGETS: tuple[str, ...] = (
    "binary_disagreement",
    "variance",
    "entropy",
    "range",
    "crosses_threshold",
)

TargetKind = Literal["continuous", "binary"]


@dataclass
class DisagreementModel:
    """A fitted disagreement predictor with its out-of-fold predictions.

    Attributes:
        target_name: Which disagreement metric was modelled.
        target_kind: ``"continuous"`` or ``"binary"``.
        oof_predictions: (n,) out-of-fold predicted disagreement.
        observed: (n,) observed disagreement.
        spearman: Rank correlation between predicted and observed.
        auc: AUC when the target is binary (or binarised at its median), which is
            the metric that matches the triage use-case.
        n_splits_used: Folds that produced predictions.
        feature_names: Design-matrix columns.
        fold_models: The per-fold fitted estimators, kept for inspection.
    """

    target_name: str
    target_kind: TargetKind
    oof_predictions: np.ndarray
    observed: np.ndarray
    spearman: float
    auc: float
    n_splits_used: int
    feature_names: list[str]
    fold_models: list[Any]


def fit_disagreement_model(
    x: np.ndarray,
    disagreement: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    target_name: str = "binary_disagreement",
    target_kind: TargetKind = "continuous",
    estimator: Any = None,
    n_splits: int = 5,
    seed: int = 0,
) -> DisagreementModel:
    """Fit a grouped cross-validated model of reader disagreement.

    Args:
        x: (n, k) image-feature design matrix.
        disagreement: (n,) observed disagreement values.
        groups: (n,) scan/patient identifiers. Grouping is mandatory for the same
            reason as in the malignancy model: nodules from one scan share the
            scanner, the dose and the reader panel.
        feature_names: Column names.
        target_name: Label recorded on the result.
        target_kind: ``"continuous"`` fits a regressor; ``"binary"`` fits a
            classifier on ``disagreement > 0``.
        estimator: Unfitted scikit-learn estimator; defaults to a standardised
            gradient-boosting regressor or logistic regression as appropriate.
        n_splits: Folds.
        seed: RNG seed.

    Returns:
        A :class:`DisagreementModel`.

    Raises:
        ValueError: If shapes disagree, there are fewer than 2 groups, or the
            target is constant.
    """
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(disagreement, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    if not (x.shape[0] == y.size == g.size):
        raise ValueError(f"row mismatch: x {x.shape[0]}, target {y.size}, groups {g.size}")
    n_groups = int(np.unique(g).size)
    if n_groups < 2:
        raise ValueError("need at least 2 groups for grouped CV")
    if np.nanstd(y) == 0:
        raise ValueError(f"disagreement target {target_name!r} is constant; nothing to model")
    n_splits = int(min(n_splits, n_groups))
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(x.shape[1])]

    if estimator is None:
        estimator = (
            make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=seed))
            if target_kind == "continuous"
            else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        )

    y_fit = y if target_kind == "continuous" else (y > 0).astype(int)
    oof = np.full(y.size, np.nan)
    models: list[Any] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y_fit, groups=g):
        if target_kind == "binary" and np.unique(y_fit[train_idx]).size < 2:
            LOG.warning("fold skipped: single-class disagreement target in training split")
            continue
        model = clone(estimator)
        model.fit(x[train_idx], y_fit[train_idx])
        if target_kind == "binary":
            oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        else:
            oof[test_idx] = model.predict(x[test_idx])
        models.append(model)

    binary_reference = (y > np.nanmedian(y)).astype(int) if target_kind == "continuous" else y_fit
    return DisagreementModel(
        target_name=target_name,
        target_kind=target_kind,
        oof_predictions=oof,
        observed=y,
        spearman=_spearman(oof, y),
        auc=_auc(oof, binary_reference),
        n_splits_used=len(models),
        feature_names=names,
        fold_models=models,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation over pairwise-complete entries."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with tie handling; ``nan`` if a class is absent."""
    ok = np.isfinite(scores) & np.isfinite(labels)
    s, y = scores[ok], labels[ok]
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def triage_curve(
    predicted_disagreement: np.ndarray,
    actually_disagreed: np.ndarray,
    *,
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Recall of disagreed nodules when the top *k*% are sent for a second read.

    This is the operational metric. A model with a mediocre R^2 can still be very
    useful if its top decile is enriched for disagreement, and a model with a good
    R^2 can be useless if the enrichment sits in the middle of the range.

    Args:
        predicted_disagreement: (n,) predicted disagreement scores.
        actually_disagreed: (n,) binary indicator of real disagreement, e.g. the
            ``crosses_threshold`` metric.
        fractions: Review budgets as fractions of the caseload.

    Returns:
        Dataframe with ``fraction_reviewed``, ``n_reviewed``, ``recall``,
        ``precision``, ``lift`` (recall divided by ``fraction_reviewed``, so 1.0
        means no better than reviewing at random).

    Raises:
        ValueError: If lengths disagree or no disagreement is present.
    """
    p = np.asarray(predicted_disagreement, dtype=float).ravel()
    y = np.asarray(actually_disagreed).astype(float).ravel()
    if p.size != y.size:
        raise ValueError(f"length mismatch: {p.size} vs {y.size}")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    total_positive = float(y.sum())
    if p.size == 0 or total_positive == 0:
        raise ValueError("no finite predictions, or no disagreed nodules to recall")

    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    rows: list[dict[str, float]] = []
    for frac in fractions:
        k = int(max(1, round(frac * p.size)))
        caught = float(y_sorted[:k].sum())
        rows.append(
            {
                "fraction_reviewed": float(frac),
                "n_reviewed": float(k),
                "recall": caught / total_positive,
                "precision": caught / k,
                "lift": (caught / total_positive) / frac if frac > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def recall_at_k(
    predicted_disagreement: np.ndarray, actually_disagreed: np.ndarray, *, k_fraction: float = 0.10
) -> float:
    """Recall of disagreed nodules within the top ``k_fraction`` of the ranking.

    Args:
        predicted_disagreement: (n,) scores.
        actually_disagreed: (n,) binary indicator.
        k_fraction: Review budget.

    Returns:
        Recall in [0, 1].

    Raises:
        ValueError: If ``k_fraction`` is outside (0, 1].
    """
    if not 0.0 < k_fraction <= 1.0:
        raise ValueError("k_fraction must be in (0, 1]")
    curve = triage_curve(predicted_disagreement, actually_disagreed, fractions=(k_fraction,))
    return float(curve["recall"].iloc[0])


def compare_to_size_baseline(
    predicted_disagreement: np.ndarray,
    actually_disagreed: np.ndarray,
    diameter_mm: np.ndarray,
    *,
    thresholds_mm: Sequence[float] = (6.0, 8.0, 15.0),
    k_fraction: float = 0.10,
) -> pd.DataFrame:
    """Compare the model against size-only disagreement baselines.

    Two baselines, both of which a useful model must beat:

    ``size``
        Larger nodules are annotated more variably in absolute terms.
    ``threshold_proximity``
        ``-min_t |diameter - t|``: nodules sitting right on a Lung-RADS size
        threshold are where reader disagreement changes management. **This is the
        baseline that matters**, and it is the one almost never reported. If a
        radiomics model cannot beat "how close is this to 6 mm", it has not shown
        anything.

    Args:
        predicted_disagreement: (n,) model scores.
        actually_disagreed: (n,) binary indicator.
        diameter_mm: (n,) nodule diameters.
        thresholds_mm: Size thresholds for the proximity baseline.
        k_fraction: Review budget for recall@k.

    Returns:
        One row per method with ``auc``, ``spearman`` and ``recall_at_k``.

    Raises:
        ValueError: If lengths disagree.
    """
    p = np.asarray(predicted_disagreement, dtype=float).ravel()
    y = np.asarray(actually_disagreed).astype(float).ravel()
    d = np.asarray(diameter_mm, dtype=float).ravel()
    if not (p.size == y.size == d.size):
        raise ValueError(f"length mismatch: {p.size}, {y.size}, {d.size}")

    proximity = -np.min(np.abs(d[:, None] - np.asarray(thresholds_mm)[None, :]), axis=1)
    methods = {"model": p, "size": d, "threshold_proximity": proximity}

    rows: list[dict[str, object]] = []
    for name, score in methods.items():
        try:
            rec = recall_at_k(score, y, k_fraction=k_fraction)
        except ValueError:
            rec = float("nan")
        rows.append(
            {
                "method": name,
                "auc": _auc(score, y.astype(int)),
                "spearman": _spearman(score, y),
                f"recall_at_{int(k_fraction * 100)}pct": rec,
            }
        )
    return pd.DataFrame(rows)


def selective_prediction_curve(
    malignancy_probability: np.ndarray,
    reference_label: np.ndarray,
    uncertainty_score: np.ndarray,
    *,
    coverages: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    """Accuracy of the malignancy classifier when the most uncertain cases are deferred.

    Tests whether predicted *reader* disagreement is a usable deferral signal for
    the *malignancy* model. A well-behaved curve rises monotonically as coverage
    falls; a flat curve means the disagreement signal is orthogonal to model error,
    which is itself a reportable finding.

    Args:
        malignancy_probability: (n,) predicted probability of malignancy.
        reference_label: (n,) binary reference label.
        uncertainty_score: (n,) score by which cases are deferred, highest first.
        coverages: Fractions of cases retained.

    Returns:
        Dataframe with ``coverage``, ``n_retained``, ``accuracy``, ``auc``,
        ``brier`` and ``mean_uncertainty_retained``.

    Raises:
        ValueError: If lengths disagree or a coverage is outside (0, 1].
    """
    p = np.asarray(malignancy_probability, dtype=float).ravel()
    y = np.asarray(reference_label).astype(float).ravel()
    u = np.asarray(uncertainty_score, dtype=float).ravel()
    if not (p.size == y.size == u.size):
        raise ValueError(f"length mismatch: {p.size}, {y.size}, {u.size}")
    if any(not 0.0 < c <= 1.0 for c in coverages):
        raise ValueError("coverages must lie in (0, 1]")

    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(u)
    p, y, u = p[ok], y[ok], u[ok]
    # Retain the most certain cases: ascending uncertainty.
    order = np.argsort(u, kind="mergesort")
    p, y, u = p[order], y[order], u[order]

    rows: list[dict[str, float]] = []
    for coverage in coverages:
        k = int(max(1, round(coverage * p.size)))
        pk, yk, uk = p[:k], y[:k], u[:k]
        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": float(k),
                "accuracy": float(((pk > 0.5).astype(float) == (yk > 0.5).astype(float)).mean()),
                "auc": _auc(pk, (yk > 0.5).astype(int)),
                "brier": float(np.mean((pk - yk) ** 2)),
                "mean_uncertainty_retained": float(uk.mean()),
            }
        )
    return pd.DataFrame(rows)
