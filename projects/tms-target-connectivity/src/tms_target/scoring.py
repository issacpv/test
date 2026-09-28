"""Joint connectivity x E-field target scoring and its evaluation.

The central construct is **network dose**. Classical stimulation network mapping
scores a target by the connectivity between the nominal stimulation site and a
reference region:

    score_conn(s) = FC(s, ref)

which treats the coil as a delta function at ``s`` and ignores dose entirely. The
joint model replaces the delta function with the actual field distribution:

    network_dose(ref) = sum_v  w(E(v)) * FC(v, ref)                        (Eq. 1)

where ``w`` is a dose-response weighting of the local field magnitude. Two
weightings are worth contrasting because they encode different biology:

* ``w(E) = E`` -- linear, i.e. the effect is proportional to field everywhere.
* ``w(E) = relu(E - E_th)`` -- thresholded, i.e. only suprathreshold cortex
  contributes. Given the sigmoidal recruitment curves seen in motor physiology,
  this is the more plausible form, and it is what makes dose normalization matter:
  under a linear weighting a global rescale cancels out of any correlation, while
  under a threshold it does not.

The evaluation asks three questions the literature has largely left open:

1. Does adding dose to a connectivity-only score improve prediction of measured
   stimulation effects?
2. Does an individualized connectome beat a normative one once dose is controlled?
   (Normative connectomes have far higher SNR; individual ones have subject-specific
   anatomy. Which wins is an empirical question, and the answer likely depends on
   how much of the apparent individualization is really just field variation.)
3. Do either survive a spatial null? (:mod:`tms_target.nulls`)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

__all__ = [
    "DoseWeighting",
    "linear_weighting",
    "threshold_weighting",
    "network_dose",
    "TargetScore",
    "score_targets",
    "ScoreModel",
    "fit_score_model",
    "grouped_cv_predictions",
    "compare_normative_vs_individual",
]

LOG = logging.getLogger(__name__)

#: A dose weighting maps per-parcel field magnitude (V/m) to a non-negative weight.
DoseWeighting = Callable[[np.ndarray], np.ndarray]


def linear_weighting(field: np.ndarray) -> np.ndarray:
    """``w(E) = max(E, 0)``: effect proportional to local field magnitude."""
    return np.maximum(np.nan_to_num(np.asarray(field, dtype=float), nan=0.0), 0.0)


def threshold_weighting(threshold_Vpm: float = 60.0, *, steepness: float | None = None) -> DoseWeighting:
    """Build a thresholded dose weighting.

    Args:
        threshold_Vpm: Field magnitude below which a parcel contributes nothing.
            Values in the 50-100 V/m range bracket published estimates of the
            cortical activation threshold for TMS.
        steepness: If given, use a soft (logistic) threshold with this slope in
            V/m instead of a hard ReLU. A soft threshold is differentiable and
            avoids a discontinuous dependence on the exact threshold value, which
            is useful when the threshold itself is fitted.

    Returns:
        A :data:`DoseWeighting`.

    Raises:
        ValueError: If ``threshold_Vpm < 0`` or ``steepness <= 0``.
    """
    if threshold_Vpm < 0:
        raise ValueError("threshold_Vpm must be non-negative")
    if steepness is not None and steepness <= 0:
        raise ValueError("steepness must be positive")

    def weighting(field: np.ndarray) -> np.ndarray:
        e = np.nan_to_num(np.asarray(field, dtype=float), nan=0.0)
        if steepness is None:
            return np.maximum(e - threshold_Vpm, 0.0)
        return e / (1.0 + np.exp(-(e - threshold_Vpm) / steepness))

    return weighting


def network_dose(
    connectivity: np.ndarray,
    field: np.ndarray,
    *,
    weighting: DoseWeighting = linear_weighting,
    normalize: Literal["sum", "max", "none"] = "sum",
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Field-weighted connectivity (Eq. 1), for every reference parcel at once.

    Args:
        connectivity: (p, p) connectome, ``connectivity[v, ref]``.
        field: (p,) per-parcel field magnitude in V/m.
        weighting: Dose-response weighting applied to ``field``.
        normalize: ``"sum"`` divides by the total weight, giving a dose-weighted
            *mean* connectivity that is comparable across montages of different
            total intensity; ``"max"`` divides by the peak weight; ``"none"``
            leaves the integral unnormalised, which keeps absolute dose in the
            score. Report both normalised and unnormalised -- they answer
            "where is the field aimed" versus "how much got delivered".
        exclude: (p,) boolean mask of source parcels to drop, e.g. parcels outside
            the field's coverage or the reference region itself.

    Returns:
        (p,) network dose per reference parcel.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    conn = np.asarray(connectivity, dtype=float)
    f = np.asarray(field, dtype=float).ravel()
    p = f.size
    if conn.shape != (p, p):
        raise ValueError(f"connectivity must be ({p}, {p}), got {conn.shape}")

    w = weighting(f)
    if exclude is not None:
        mask = np.asarray(exclude, dtype=bool).ravel()
        if mask.size != p:
            raise ValueError(f"exclude must have length {p}, got {mask.size}")
        w = np.where(mask, 0.0, w)
    w = np.where(np.isfinite(w), w, 0.0)

    total = w.sum()
    if total <= 0:
        return np.full(p, np.nan)

    contrib = np.nan_to_num(conn, nan=0.0) * w[:, None]
    out = contrib.sum(axis=0)
    if normalize == "sum":
        out = out / total
    elif normalize == "max":
        peak = w.max()
        out = out / peak if peak > 0 else out
    return out


@dataclass
class TargetScore:
    """Per-target scores from the connectivity-only and joint models.

    Attributes:
        labels: (p,) parcel labels.
        connectivity_only: (p,) score using the nominal stimulation site alone.
        network_dose_linear: (p,) Eq. 1 with linear weighting.
        network_dose_threshold: (p,) Eq. 1 with a thresholded weighting.
        local_dose: (p,) field magnitude at each parcel, the dose-only baseline.
        metadata: Provenance.
    """

    labels: np.ndarray
    connectivity_only: np.ndarray
    network_dose_linear: np.ndarray
    network_dose_threshold: np.ndarray
    local_dose: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def as_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Stack the scores into an ``(p, 4)`` design matrix with column names."""
        names = ["connectivity_only", "network_dose_linear", "network_dose_threshold", "local_dose"]
        cols = [getattr(self, n) for n in names]
        return np.column_stack(cols), names


def score_targets(
    connectivity: np.ndarray,
    field: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    site_index: int | None = None,
    threshold_Vpm: float = 60.0,
    exclude: np.ndarray | None = None,
) -> TargetScore:
    """Compute all four target scores for one stimulation session.

    Args:
        connectivity: (p, p) connectome.
        field: (p,) parcellated field magnitude, V/m.
        labels: (p,) parcel labels; generated if omitted.
        site_index: Parcel treated as the nominal stimulation site for the
            connectivity-only baseline. Defaults to the peak-field parcel, which
            is the fairest baseline: it is what a careful connectivity-only study
            would use.
        threshold_Vpm: Threshold for the thresholded weighting.
        exclude: (p,) source parcels to drop from the integrals.

    Returns:
        A :class:`TargetScore`.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    f = np.asarray(field, dtype=float).ravel()
    conn = np.asarray(connectivity, dtype=float)
    p = f.size
    if conn.shape != (p, p):
        raise ValueError(f"connectivity must be ({p}, {p}), got {conn.shape}")
    if labels is None:
        labels = np.array([f"parcel_{i:04d}" for i in range(p)])
    if site_index is None:
        site_index = int(np.nanargmax(f))

    return TargetScore(
        labels=np.asarray(labels),
        connectivity_only=np.nan_to_num(conn[site_index], nan=np.nan),
        network_dose_linear=network_dose(conn, f, weighting=linear_weighting, exclude=exclude),
        network_dose_threshold=network_dose(
            conn, f, weighting=threshold_weighting(threshold_Vpm), exclude=exclude
        ),
        local_dose=f.copy(),
        metadata={"site_index": int(site_index), "threshold_Vpm": float(threshold_Vpm)},
    )


@dataclass
class ScoreModel:
    """A fitted linear target-scoring model.

    Attributes:
        feature_names: Design-matrix column names.
        coefficients: (k,) standardised coefficients.
        intercept: Fitted intercept.
        alpha: Ridge penalty used.
        feature_mean: (k,) training means used for standardisation.
        feature_scale: (k,) training scales used for standardisation.
        n_train: Number of training rows.
    """

    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    n_train: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict outcomes for a design matrix.

        Args:
            features: (n, k) matrix with columns in ``feature_names`` order.

        Returns:
            (n,) predictions.

        Raises:
            ValueError: If the column count does not match the fit.
        """
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        z = (x - self.feature_mean) / self.feature_scale
        return z @ self.coefficients + self.intercept


def fit_score_model(
    features: np.ndarray,
    outcome: np.ndarray,
    feature_names: Sequence[str],
    *,
    alpha: float = 1.0,
) -> ScoreModel:
    """Fit a standardised ridge model of outcome on target scores.

    Ridge rather than OLS because the candidate features are strongly collinear by
    construction -- network dose contains connectivity, and local dose is part of
    network dose. Standardised coefficients are then interpretable as relative
    contributions, which is the quantity the paper needs.

    Args:
        features: (n, k) design matrix.
        outcome: (n,) measured stimulation effect.
        feature_names: Column names.
        alpha: Ridge penalty on standardised features.

    Returns:
        A fitted :class:`ScoreModel`.

    Raises:
        ValueError: If shapes disagree, ``alpha < 0``, or fewer than 3 complete rows.
    """
    x = np.atleast_2d(np.asarray(features, dtype=float))
    y = np.asarray(outcome, dtype=float).ravel()
    if x.shape[0] != y.size:
        raise ValueError(f"features has {x.shape[0]} rows but outcome has {y.size}")
    if x.shape[1] != len(feature_names):
        raise ValueError(f"{x.shape[1]} columns but {len(feature_names)} names")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() < 3:
        raise ValueError(f"need >=3 complete rows, got {int(ok.sum())}")
    x, y = x[ok], y[ok]

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    y_mean = y.mean()

    gram = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T @ (y - y_mean))
    return ScoreModel(
        feature_names=list(feature_names),
        coefficients=coef,
        intercept=float(y_mean),
        alpha=float(alpha),
        feature_mean=mean,
        feature_scale=scale,
        n_train=int(ok.sum()),
    )


def grouped_cv_predictions(
    features: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    feature_names: Sequence[str],
    *,
    alpha: float = 1.0,
) -> dict[str, object]:
    """Leave-one-group-out cross-validated predictions.

    The group must be **subject** (or site, or dataset), never target or session.
    Two sessions from one subject share anatomy, coil calibration and head
    position, so splitting within subject leaks the very thing the model is
    supposed to generalize across -- and would make an individualized connectome
    look good for the wrong reason.

    Args:
        features: (n, k) design matrix.
        outcome: (n,) measured effects.
        groups: (n,) group labels.
        feature_names: Column names.
        alpha: Ridge penalty.

    Returns:
        Dict with ``predictions``, ``outcome``, ``groups``, ``r_pearson``,
        ``r_spearman``, ``mae``, ``n_groups`` and per-fold coefficients.

    Raises:
        ValueError: If shapes disagree or there are fewer than 2 groups.
    """
    x = np.atleast_2d(np.asarray(features, dtype=float))
    y = np.asarray(outcome, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    if not (x.shape[0] == y.size == g.size):
        raise ValueError(f"row mismatch: features {x.shape[0]}, outcome {y.size}, groups {g.size}")
    unique = np.unique(g)
    if unique.size < 2:
        raise ValueError("need at least 2 groups for leave-one-group-out CV")

    pred = np.full(y.size, np.nan)
    fold_coefs: list[dict[str, float]] = []
    for held in unique:
        test = g == held
        train = ~test
        if train.sum() < 3 or not np.isfinite(y[train]).any():
            continue
        try:
            model = fit_score_model(x[train], y[train], feature_names, alpha=alpha)
        except ValueError as exc:
            LOG.warning("fold %s skipped: %s", held, exc)
            continue
        pred[test] = model.predict(x[test])
        fold_coefs.append(dict(zip(feature_names, model.coefficients.tolist())))

    ok = np.isfinite(pred) & np.isfinite(y)
    return {
        "predictions": pred,
        "outcome": y,
        "groups": g,
        "n_groups": int(unique.size),
        "n_scored": int(ok.sum()),
        "r_pearson": _safe_corr(pred[ok], y[ok], rank=False),
        "r_spearman": _safe_corr(pred[ok], y[ok], rank=True),
        "mae": float(np.mean(np.abs(pred[ok] - y[ok]))) if ok.any() else float("nan"),
        "fold_coefficients": fold_coefs,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray, *, rank: bool) -> float:
    """Correlation that returns ``nan`` rather than raising on degenerate input."""
    if a.size < 3:
        return float("nan")
    if rank:
        a = np.argsort(np.argsort(a)).astype(float)
        b = np.argsort(np.argsort(b)).astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_normative_vs_individual(
    normative_features: np.ndarray,
    individual_features: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    feature_names: Sequence[str],
    *,
    alpha: float = 1.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Head-to-head comparison of normative and individualized connectome scoring.

    Both models see the same folds and the same outcome, so the comparison is
    paired. The CI on the difference in predictive correlation comes from a
    **group-level** bootstrap, which is the only resampling unit that respects the
    dependence between sessions of one subject.

    Args:
        normative_features: (n, k) design matrix built from a group-average connectome.
        individual_features: (n, k) design matrix built from subject-specific connectomes.
        outcome: (n,) measured effects.
        groups: (n,) subject labels.
        feature_names: Column names, shared by both matrices.
        alpha: Ridge penalty.
        n_boot: Bootstrap replicates for the difference CI.
        seed: RNG seed.

    Returns:
        Dict with ``normative``, ``individual`` (each a
        :func:`grouped_cv_predictions` result), ``delta_r_spearman`` and its
        bootstrap CI.

    Raises:
        ValueError: If the two design matrices have different shapes.
    """
    a = np.atleast_2d(np.asarray(normative_features, dtype=float))
    b = np.atleast_2d(np.asarray(individual_features, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"design matrices must match: {a.shape} vs {b.shape}")

    res_norm = grouped_cv_predictions(a, outcome, groups, feature_names, alpha=alpha)
    res_indiv = grouped_cv_predictions(b, outcome, groups, feature_names, alpha=alpha)

    y = np.asarray(outcome, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    pn = np.asarray(res_norm["predictions"], dtype=float)
    pi = np.asarray(res_indiv["predictions"], dtype=float)
    delta = float(res_indiv["r_spearman"] - res_norm["r_spearman"])

    rng = np.random.default_rng(seed)
    unique = np.unique(g)
    index = {u: np.flatnonzero(g == u) for u in unique}
    deltas: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index[u] for u in picked])
        ok = np.isfinite(y[rows]) & np.isfinite(pn[rows]) & np.isfinite(pi[rows])
        if ok.sum() < 4:
            continue
        r_n = _safe_corr(pn[rows][ok], y[rows][ok], rank=True)
        r_i = _safe_corr(pi[rows][ok], y[rows][ok], rank=True)
        if np.isfinite(r_n) and np.isfinite(r_i):
            deltas.append(r_i - r_n)

    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {
        "normative": res_norm,
        "individual": res_indiv,
        "delta_r_spearman": delta,
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "n_boot_valid": len(deltas),
    }
