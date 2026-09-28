"""Soft-label training and calibration assessment for multi-rater malignancy labels.

The question this module exists to answer: **does training on the readers' actual
label distribution, rather than on a majority vote, produce a better-calibrated
malignancy classifier?** There is a clean theoretical reason to expect yes. If four
readers split 2-2, the majority vote asserts a label the data does not support,
and the model is pushed toward a confident prediction on an intrinsically ambiguous
case. A soft target of 0.5 instead asks the model to be uncertain there -- which is
both true and, for a screening workflow, actionable.

Two training routes to soft labels are provided:

``sample_weighted``
    Duplicate each nodule as a positive with weight ``p`` and a negative with
    weight ``1 - p``. For any loss that is linear in the per-sample label (log loss
    included) this is **exactly** equivalent to minimising soft-label
    cross-entropy, so it works with any scikit-learn estimator supporting
    ``sample_weight``, including calibrated wrappers and gradient boosting.

``regression``
    Regress the soft label directly and clip to [0, 1]. Simpler, but yields a
    conditional-mean estimate that is not a proper probability and calibrates worse
    in practice. Included as a baseline because several published pipelines do it.

Calibration is assessed with expected and maximum calibration error, the Brier
score and its three-way decomposition (reliability, resolution, uncertainty), plus
reliability-curve data. The Brier decomposition matters here specifically: a
multi-rater dataset has irreducible label noise, so a floor on the Brier score
exists, and the **uncertainty** term estimates it. Comparing raw Brier scores
without that term makes ambiguous datasets look like model failures
(cf. Guo et al. 2017, *ICML*, for why confident modern classifiers are miscalibrated).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SoftLabelTrainer",
    "CalibrationReport",
    "expand_soft_labels",
    "calibration_metrics",
    "reliability_curve",
    "brier_decomposition",
    "grouped_cv_soft",
    "compare_label_schemes",
]

LOG = logging.getLogger(__name__)

TrainRoute = Literal["sample_weighted", "regression", "hard"]


def expand_soft_labels(
    x: np.ndarray, soft: np.ndarray, *, weights: np.ndarray | None = None, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand soft labels into weighted binary rows.

    Each input row with soft label ``p`` becomes at most two rows: ``(x, 1, w*p)``
    and ``(x, 0, w*(1-p))``. Degenerate weights below ``eps`` are dropped, so a
    nodule with ``p = 0`` or ``p = 1`` contributes a single row as it should.

    Args:
        x: (n, k) design matrix.
        soft: (n,) soft labels in [0, 1].
        weights: (n,) optional per-nodule weights, e.g. ``n_raters`` so that a
            4-reader nodule counts more than a 1-reader one. Strongly recommended:
            a soft label of 0.5 from two readers is far weaker evidence than the
            same value from four.
        eps: Weight below which a row is dropped.

    Returns:
        ``(x_expanded, y_expanded, weight_expanded)``.

    Raises:
        ValueError: If shapes disagree or soft labels fall outside [0, 1].
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    p = np.asarray(soft, dtype=float).ravel()
    if x.shape[0] != p.size:
        raise ValueError(f"x has {x.shape[0]} rows but soft has {p.size}")
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        raise ValueError("soft labels must lie in [0, 1]")
    p = np.clip(p, 0.0, 1.0)
    w = np.ones(p.size) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != p.size:
        raise ValueError(f"weights must have length {p.size}, got {w.size}")

    xs = np.concatenate([x, x], axis=0)
    ys = np.concatenate([np.ones(p.size), np.zeros(p.size)])
    ws = np.concatenate([w * p, w * (1.0 - p)])
    keep = ws > eps
    return xs[keep], ys[keep], ws[keep]


def _sample_weight_kwargs(estimator: Any, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Build the ``fit`` kwargs that pass ``weights`` as sample weights.

    ``Pipeline.fit`` does not accept a bare ``sample_weight``: it must be addressed
    to a named step as ``<stepname>__sample_weight``. Since a standardising pipeline
    is the natural estimator here, getting this wrong silently drops every fold, so
    the routing is handled explicitly rather than with a bare ``except TypeError``.

    Args:
        estimator: The estimator about to be fitted.
        weights: (n,) sample weights.

    Returns:
        Keyword arguments for ``estimator.fit``.
    """
    steps = getattr(estimator, "steps", None)
    if steps:
        return {f"{steps[-1][0]}__sample_weight": weights}
    return {"sample_weight": weights}


@dataclass
class SoftLabelTrainer:
    """Fit a scikit-learn classifier on multi-rater soft labels.

    Attributes:
        estimator: An unfitted scikit-learn estimator. For ``"sample_weighted"``
            it must accept ``sample_weight`` in ``fit`` and expose
            ``predict_proba``; for ``"regression"`` it must be a regressor.
        route: Training route (see module docstring).
        use_rater_weights: Multiply expansion weights by the per-nodule rater count.
    """

    estimator: Any
    route: TrainRoute = "sample_weighted"
    use_rater_weights: bool = True
    _fitted: Any = field(default=None, repr=False)

    def fit(
        self,
        x: np.ndarray,
        soft: np.ndarray,
        *,
        n_raters: np.ndarray | None = None,
        hard: np.ndarray | None = None,
    ) -> "SoftLabelTrainer":
        """Fit the estimator.

        Args:
            x: (n, k) design matrix.
            soft: (n,) soft labels in [0, 1].
            n_raters: (n,) reader counts, used as weights when
                ``use_rater_weights`` is set.
            hard: (n,) majority-vote labels, required for ``route="hard"``.

        Returns:
            ``self``.

        Raises:
            ValueError: If the route is unknown or required inputs are missing.
        """
        from sklearn.base import clone

        x = np.atleast_2d(np.asarray(x, dtype=float))
        p = np.asarray(soft, dtype=float).ravel()
        est = clone(self.estimator)

        if self.route == "sample_weighted":
            w = n_raters if (self.use_rater_weights and n_raters is not None) else None
            xs, ys, ws = expand_soft_labels(x, p, weights=w)
            if np.unique(ys).size < 2:
                raise ValueError("expanded labels are single-class; cannot fit a classifier")
            est.fit(xs, ys, **_sample_weight_kwargs(est, ws))
        elif self.route == "regression":
            est.fit(x, p)
        elif self.route == "hard":
            if hard is None:
                raise ValueError("route='hard' requires the hard label array")
            y = np.asarray(hard, dtype=int).ravel()
            if np.unique(y).size < 2:
                raise ValueError("hard labels are single-class; cannot fit a classifier")
            sw = (
                np.asarray(n_raters, dtype=float)
                if (self.use_rater_weights and n_raters is not None)
                else None
            )
            if sw is None:
                est.fit(x, y)
            else:
                est.fit(x, y, **_sample_weight_kwargs(est, sw))
        else:
            raise ValueError(f"unknown route {self.route!r}")

        self._fitted = est
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicted probability of malignancy.

        Args:
            x: (n, k) design matrix.

        Returns:
            (n,) probabilities in [0, 1].

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._fitted is None:
            raise RuntimeError("call fit() before predict_proba()")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.route == "regression":
            return np.clip(np.asarray(self._fitted.predict(x), dtype=float), 0.0, 1.0)
        proba = np.asarray(self._fitted.predict_proba(x), dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            classes = getattr(self._fitted, "classes_", np.array([0, 1]))
            pos = int(np.argmax(np.asarray(classes) == 1)) if 1 in np.asarray(classes) else 1
            return proba[:, pos]
        return proba.ravel()


@dataclass
class CalibrationReport:
    """Calibration and discrimination summary.

    Attributes:
        n: Number of scored nodules.
        auc: Area under the ROC curve against the hard reference labels.
        brier: Brier score against the *soft* labels, which is the right target
            when the reference itself is probabilistic.
        brier_vs_hard: Brier score against hard labels, for comparability with
            published numbers.
        ece: Expected calibration error (equal-count bins).
        mce: Maximum calibration error over bins.
        reliability: Reliability term of the Brier decomposition (lower is better).
        resolution: Resolution term (higher is better).
        uncertainty: Uncertainty term -- the irreducible floor set by label noise.
        curve: Reliability-curve dataframe with bin means and counts.
        n_bins: Number of calibration bins used.
    """

    n: int
    auc: float
    brier: float
    brier_vs_hard: float
    ece: float
    mce: float
    reliability: float
    resolution: float
    uncertainty: float
    curve: pd.DataFrame
    n_bins: int


def reliability_curve(
    probabilities: np.ndarray, targets: np.ndarray, *, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Binned reliability curve.

    Args:
        probabilities: (n,) predicted probabilities.
        targets: (n,) observed outcomes in [0, 1]; soft targets are allowed and are
            the more informative choice for a multi-rater dataset.
        n_bins: Number of bins.
        strategy: ``"quantile"`` for equal-count bins (the default, since it keeps
            every bin's estimate stable) or ``"uniform"`` for equal-width bins.

    Returns:
        Dataframe with ``bin``, ``n``, ``mean_predicted``, ``mean_observed``, ``gap``.

    Raises:
        ValueError: If lengths disagree, ``n_bins < 1``, or the strategy is unknown.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()
    if p.size != y.size:
        raise ValueError(f"length mismatch: {p.size} vs {y.size}")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "mean_observed", "gap"])

    if strategy == "quantile":
        edges = np.unique(np.percentile(p, np.linspace(0, 100, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(sel.sum()),
                "mean_predicted": float(p[sel].mean()),
                "mean_observed": float(y[sel].mean()),
                "gap": float(p[sel].mean() - y[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


def brier_decomposition(
    probabilities: np.ndarray, targets: np.ndarray, *, n_bins: int = 10
) -> dict[str, float]:
    """Murphy three-way decomposition ``Brier = reliability - resolution + uncertainty``.

    Args:
        probabilities: (n,) predicted probabilities.
        targets: (n,) observed outcomes in [0, 1].
        n_bins: Bins for the decomposition (equal-count).

    Returns:
        Dict with ``brier``, ``reliability``, ``resolution``, ``uncertainty`` and
        ``decomposition_residual`` (the identity's numerical gap, which binning
        makes non-zero; a large residual means too few bins or too few samples).

    Raises:
        ValueError: If lengths disagree.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()
    if p.size != y.size:
        raise ValueError(f"length mismatch: {p.size} vs {y.size}")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return {k: float("nan") for k in
                ("brier", "reliability", "resolution", "uncertainty", "decomposition_residual")}

    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    uncertainty = float(np.mean((y - base) ** 2))

    curve = reliability_curve(p, y, n_bins=n_bins)
    if curve.empty:
        return {"brier": brier, "reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": uncertainty, "decomposition_residual": float("nan")}
    n = curve["n"].to_numpy(dtype=float)
    pred = curve["mean_predicted"].to_numpy(dtype=float)
    obs = curve["mean_observed"].to_numpy(dtype=float)
    total = n.sum()
    reliability = float((n * (pred - obs) ** 2).sum() / total)
    resolution = float((n * (obs - base) ** 2).sum() / total)
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_residual": float(brier - (reliability - resolution + uncertainty)),
    }


def calibration_metrics(
    probabilities: np.ndarray,
    soft_targets: np.ndarray,
    hard_targets: np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration and discrimination report.

    Args:
        probabilities: (n,) predicted probabilities of malignancy.
        soft_targets: (n,) soft labels in [0, 1].
        hard_targets: (n,) binary labels for AUC; derived by thresholding the soft
            labels at 0.5 when omitted.
        n_bins: Calibration bins.

    Returns:
        A :class:`CalibrationReport`.

    Raises:
        ValueError: If lengths disagree or nothing is finite.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    s = np.asarray(soft_targets, dtype=float).ravel()
    if p.size != s.size:
        raise ValueError(f"length mismatch: {p.size} vs {s.size}")
    h = (s > 0.5).astype(int) if hard_targets is None else np.asarray(hard_targets, dtype=int).ravel()
    if h.size != p.size:
        raise ValueError(f"hard_targets must have length {p.size}, got {h.size}")

    ok = np.isfinite(p) & np.isfinite(s)
    if not ok.any():
        raise ValueError("no finite predictions")
    p, s, h = p[ok], s[ok], h[ok]

    curve = reliability_curve(p, s, n_bins=n_bins)
    if curve.empty:
        ece = mce = float("nan")
    else:
        n = curve["n"].to_numpy(dtype=float)
        gap = np.abs(curve["gap"].to_numpy(dtype=float))
        ece = float((n * gap).sum() / n.sum())
        mce = float(gap.max())

    decomp = brier_decomposition(p, s, n_bins=n_bins)
    return CalibrationReport(
        n=int(p.size),
        auc=_auc(p, h),
        brier=float(np.mean((p - s) ** 2)),
        brier_vs_hard=float(np.mean((p - h) ** 2)),
        ece=ece,
        mce=mce,
        reliability=decomp["reliability"],
        resolution=decomp["resolution"],
        uncertainty=decomp["uncertainty"],
        curve=curve,
        n_bins=n_bins,
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC with correct tie handling; ``nan`` if one class is absent."""
    pos = labels == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # average ranks within tied groups
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def grouped_cv_soft(
    x: np.ndarray,
    soft: np.ndarray,
    groups: np.ndarray,
    *,
    estimator: Any = None,
    route: TrainRoute = "sample_weighted",
    n_raters: np.ndarray | None = None,
    hard: np.ndarray | None = None,
    n_splits: int = 5,
    use_rater_weights: bool = True,
    eval_soft: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Grouped k-fold cross-validation with out-of-fold probabilities.

    Grouping is by **scan / patient**. A patient can contribute several nodules,
    and nodules from one scan share the scanner, reconstruction kernel, dose and
    the same four readers -- so a nodule-level split leaks all of that and inflates
    every metric. This is the single most common methodological error in the LIDC
    literature.

    Args:
        x: (n, k) design matrix.
        soft: (n,) soft labels.
        groups: (n,) scan/patient identifiers.
        estimator: Unfitted scikit-learn estimator; defaults to a standardised
            logistic regression.
        route: Training route.
        n_raters: (n,) reader counts.
        hard: (n,) majority labels, needed for ``route="hard"``.
        n_splits: Number of folds.
        use_rater_weights: Weight nodules by reader count.
        eval_soft: (n,) reference soft labels used for the calibration report,
            independent of what was trained on. Pass this whenever several arms
            with different training targets must be scored on a common reference;
            defaults to ``soft``.
        seed: RNG seed for fold assignment.

    Returns:
        Dict with ``oof_probabilities``, ``report`` (a :class:`CalibrationReport`),
        ``n_splits_used`` and ``fold_sizes``.

    Raises:
        ValueError: If shapes disagree or there are fewer groups than folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.atleast_2d(np.asarray(x, dtype=float))
    p_soft = np.asarray(soft, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    if not (x.shape[0] == p_soft.size == g.size):
        raise ValueError(f"row mismatch: x {x.shape[0]}, soft {p_soft.size}, groups {g.size}")
    n_groups = int(np.unique(g).size)
    if n_groups < 2:
        raise ValueError("need at least 2 groups for grouped CV")
    n_splits = int(min(n_splits, n_groups))

    if estimator is None:
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        )

    oof = np.full(p_soft.size, np.nan)
    fold_sizes: list[int] = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x, p_soft, groups=g):
        trainer = SoftLabelTrainer(estimator, route=route, use_rater_weights=use_rater_weights)
        try:
            trainer.fit(
                x[train_idx],
                p_soft[train_idx],
                n_raters=None if n_raters is None else np.asarray(n_raters)[train_idx],
                hard=None if hard is None else np.asarray(hard)[train_idx],
            )
        except ValueError as exc:
            LOG.warning("fold skipped: %s", exc)
            continue
        oof[test_idx] = trainer.predict_proba(x[test_idx])
        fold_sizes.append(int(test_idx.size))

    reference = p_soft if eval_soft is None else np.asarray(eval_soft, dtype=float).ravel()
    if reference.size != p_soft.size:
        raise ValueError(f"eval_soft must have length {p_soft.size}, got {reference.size}")
    report = calibration_metrics(
        oof, reference, None if hard is None else np.asarray(hard, dtype=int)
    )
    return {
        "oof_probabilities": oof,
        "report": report,
        "n_splits_used": len(fold_sizes),
        "fold_sizes": fold_sizes,
    }


def compare_label_schemes(
    x: np.ndarray,
    soft_by_scheme: dict[str, np.ndarray],
    hard: np.ndarray,
    groups: np.ndarray,
    *,
    n_raters: np.ndarray | None = None,
    estimator: Any = None,
    n_splits: int = 5,
    eval_scheme: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare soft-label schemes against majority-vote training, head to head.

    Every arm sees identical folds **and is scored against the same reference soft
    label**, so differences in the reported metrics are attributable to the label
    definition used for *training* rather than to the split or to a moving target.
    Scoring each arm against its own training target would be circular and would
    flatter whichever scheme happens to be easiest to fit.

    Args:
        x: (n, k) design matrix.
        soft_by_scheme: Mapping of scheme name to (n,) soft labels.
        hard: (n,) majority-vote labels, used for the hard baseline and for AUC.
        groups: (n,) scan identifiers.
        n_raters: (n,) reader counts.
        estimator: Unfitted estimator, shared by all arms.
        n_splits: Folds.
        eval_scheme: Key of ``soft_by_scheme`` to use as the common evaluation
            reference. Defaults to ``"split_3"`` when present, else the first key.
        seed: RNG seed.

    Returns:
        One row per arm with ``auc``, ``brier``, ``brier_vs_hard``, ``ece``,
        ``mce``, ``reliability``, ``resolution`` and ``uncertainty``.

    Raises:
        ValueError: If ``soft_by_scheme`` is empty or ``eval_scheme`` is unknown.
    """
    if not soft_by_scheme:
        raise ValueError("no soft-label schemes supplied")
    if eval_scheme is None:
        eval_scheme = "split_3" if "split_3" in soft_by_scheme else next(iter(soft_by_scheme))
    if eval_scheme not in soft_by_scheme:
        raise ValueError(f"eval_scheme {eval_scheme!r} not in {sorted(soft_by_scheme)}")
    reference = np.asarray(soft_by_scheme[eval_scheme], dtype=float)

    arms: list[tuple[str, np.ndarray, TrainRoute]] = [
        ("majority_vote", np.asarray(hard, dtype=float), "hard")
    ]
    arms += [(f"soft_{name}", np.asarray(v, dtype=float), "sample_weighted")
             for name, v in soft_by_scheme.items()]

    rows: list[dict[str, object]] = []
    for name, target, route in arms:
        res = grouped_cv_soft(
            x, target, groups, estimator=estimator, route=route, n_raters=n_raters,
            hard=np.asarray(hard, dtype=int), n_splits=n_splits,
            eval_soft=reference, seed=seed,
        )
        rep: CalibrationReport = res["report"]
        rows.append(
            {
                "arm": name,
                "route": route,
                "eval_reference": eval_scheme,
                "n": rep.n,
                "auc": rep.auc,
                "brier": rep.brier,
                "brier_vs_hard": rep.brier_vs_hard,
                "ece": rep.ece,
                "mce": rep.mce,
                "reliability": rep.reliability,
                "resolution": rep.resolution,
                "uncertainty": rep.uncertainty,
            }
        )
    return pd.DataFrame(rows)
