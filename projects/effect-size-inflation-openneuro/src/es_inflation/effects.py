"""Effect-size conversions, funnel asymmetry, and circular vs cross-validated peak effects.

Two inflation mechanisms are handled here:

* across-study: conversions from reported statistics to d / r with standard errors, and an Egger-type
  funnel-asymmetry test over a corpus (``funnel_asymmetry``);
* within-study: the *circular* peak effect (select the best voxel, report its effect on the same
  subjects) versus a *cross-validated* peak effect (select on some subjects, estimate on others),
  implemented on plain arrays so it can be applied to any subject-by-feature effect matrix
  (``circular_peak_effect``, ``cross_validated_peak_effect``, ``split_half_peak_inflation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "ANNOTATION_COLUMNS",
    "d_from_t",
    "r_from_t",
    "fisher_z",
    "d_from_r",
    "annotation_to_effect",
    "funnel_asymmetry",
    "pooled_effect",
    "cohens_d_columns",
    "circular_peak_effect",
    "cross_validated_peak_effect",
    "split_half_peak_inflation",
    "PeakInflation",
]

# Columns of data/papers/effects_annotated.csv (one row per reported primary effect).
ANNOTATION_COLUMNS = [
    "dataset_id",  # OpenNeuro accession
    "doi",  # paper DOI
    "contrast",  # free text, e.g. "faces > shapes"
    "design",  # one_sample | paired | two_sample | correlation
    "stat_type",  # t | z | F | r
    "stat_value",  # reported peak statistic
    "df",  # degrees of freedom if reported
    "n1",  # analysed n (group 1 / total for one-sample)
    "n2",  # group 2 n for two-sample designs, else blank
    "threshold_type",  # voxel_fwe | voxel_fdr | voxel_unc | cluster_fwe | roi_apriori
    "threshold_p",  # e.g. 0.001 (cluster-forming) or 0.05 (voxel FWE)
    "peak_selected_by_search",  # 1 if the peak was found by whole-brain search, 0 if a priori ROI
    "annotator",
]


# ----------------------------------------------------------- conversions
def d_from_t(t: float | np.ndarray, n1: int | np.ndarray, n2: Optional[int | np.ndarray] = None) -> np.ndarray:
    """Cohen's d from a t statistic: ``t / sqrt(n)`` (one-sample/paired) or ``t sqrt(1/n1 + 1/n2)``."""
    t = np.asarray(t, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    if n2 is None:
        return t / np.sqrt(n1)
    n2 = np.asarray(n2, dtype=float)
    return t * np.sqrt(1.0 / n1 + 1.0 / n2)


def r_from_t(t: float | np.ndarray, df: int | np.ndarray) -> np.ndarray:
    """Correlation-scale effect from t and df: ``sqrt(t^2 / (t^2 + df))`` with the sign of t."""
    t = np.asarray(t, dtype=float)
    df = np.asarray(df, dtype=float)
    return np.sign(t) * np.sqrt(t**2 / (t**2 + df))


def fisher_z(r: float | np.ndarray) -> np.ndarray:
    """Fisher z-transform ``atanh(r)``."""
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999))


def d_from_r(r: float | np.ndarray) -> np.ndarray:
    """Cohen's d from a point-biserial-type r: ``2 r / sqrt(1 - r^2)``."""
    r = np.asarray(r, dtype=float)
    return 2.0 * r / np.sqrt(1.0 - r**2)


def annotation_to_effect(row: Dict[str, object]) -> Dict[str, float]:
    """Turn one annotated row (see ``ANNOTATION_COLUMNS``) into ``d, se_d, threshold_z``.

    * ``t`` with one-sample/paired design: d = t/sqrt(n1);  two-sample: d = t sqrt(1/n1+1/n2)
    * ``z`` : treated as a t with large df (d = z/sqrt(n) for one-sample)
    * ``r`` : d = 2r/sqrt(1-r^2), se from Fisher z propagated
    * ``F`` (1 numerator df): t = sqrt(F)
    The selection threshold ``threshold_z`` is the two-sided critical z for ``threshold_p``
    (cluster-level thresholds use the cluster-forming p, a conservative choice).
    """
    from .winners_curse import se_cohens_d, se_fisher_z, z_from_alpha

    stat_type = str(row["stat_type"]).lower()
    val = float(row["stat_value"])
    n1 = int(row["n1"])
    n2 = row.get("n2")
    n2 = int(n2) if n2 not in (None, "", "nan") and not (isinstance(n2, float) and np.isnan(n2)) else None
    design = str(row.get("design", "one_sample")).lower()
    if stat_type == "f":
        val = float(np.sqrt(val))
        stat_type = "t"
    if stat_type in ("t", "z"):
        d = float(d_from_t(val, n1, n2 if design == "two_sample" else None))
        se = float(se_cohens_d(n1, n2 if design == "two_sample" else None, d))
    elif stat_type == "r":
        d = float(d_from_r(val))
        se_z = float(se_fisher_z(n1))
        # delta-method: dd/dr = 2 / (1 - r^2)^{3/2}; dr/dz = 1 - r^2
        se = float(se_z * (1 - val**2) * 2.0 / (1 - val**2) ** 1.5)
    else:
        raise ValueError(f"unknown stat_type {stat_type!r}")
    p = float(row.get("threshold_p", 0.05) or 0.05)
    return {"d": d, "se_d": se, "threshold_z": z_from_alpha(p)}


def cohens_d_columns(betas: np.ndarray) -> np.ndarray:
    """Per-column Cohen's d (mean / sd, ddof=1) for a subjects x features effect matrix."""
    betas = np.asarray(betas, dtype=float)
    sd = betas.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, betas.mean(axis=0) / sd, 0.0)


# --------------------------------------------------------- corpus tools
def funnel_asymmetry(d_hats: Sequence[float], ses: Sequence[float]) -> Dict[str, float]:
    """Egger regression: ``d_hat / se = b0 + b1 (1/se)``; ``b0 != 0`` indicates small-study effects.

    Returns intercept, its SE, t and two-sided p. Descriptive only: heterogeneity in true effects can
    also produce asymmetry.
    """
    y = np.asarray(d_hats, dtype=float) / np.asarray(ses, dtype=float)
    x = 1.0 / np.asarray(ses, dtype=float)
    if y.size < 3:
        raise ValueError("need at least three studies")
    X = np.column_stack([np.ones_like(x), x])
    beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    dfree = y.size - 2
    resid = y - X @ beta
    s2 = float(resid @ resid) / dfree
    cov = s2 * np.linalg.inv(X.T @ X)
    se0 = float(np.sqrt(cov[0, 0]))
    t0 = float(beta[0] / se0) if se0 > 0 else np.inf
    p = float(2 * stats.t.sf(abs(t0), dfree))
    return {"intercept": float(beta[0]), "se": se0, "t": t0, "p": p, "slope": float(beta[1])}


def pooled_effect(d_hats: Sequence[float], ses: Sequence[float]) -> Dict[str, float]:
    """Fixed- and random-effects (DerSimonian-Laird) pooled estimates with SEs and I^2."""
    y = np.asarray(d_hats, dtype=float)
    s2 = np.asarray(ses, dtype=float) ** 2
    w = 1 / s2
    mu_fe = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu_fe) ** 2))
    k = y.size
    cc = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / cc)
    w_re = 1 / (s2 + tau2)
    mu_re = float(np.sum(w_re * y) / np.sum(w_re))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {
        "fe": mu_fe,
        "fe_se": float(np.sqrt(1 / np.sum(w))),
        "re": mu_re,
        "re_se": float(np.sqrt(1 / np.sum(w_re))),
        "tau2": float(tau2),
        "i2": float(i2),
        "k": int(k),
    }


# --------------------------------------------- within-study peak inflation
@dataclass
class PeakInflation:
    """Circular vs cross-validated peak effect for one dataset."""

    d_circular: float
    d_cv: float
    peak_index: int
    n_subjects: int
    n_features: int

    @property
    def ratio(self) -> float:
        """Circular / cross-validated effect.

        Returns ``inf`` when the held-out effect is zero or negative (no replicable effect, so the
        inflation is unbounded); callers should report ``d_cv`` alongside the ratio.
        """
        return float(self.d_circular / self.d_cv) if self.d_cv > 1e-12 else float("inf")


def circular_peak_effect(betas: np.ndarray) -> tuple[float, int]:
    """Select the feature (voxel) with the largest one-sample t and return its d on the same data."""
    d = cohens_d_columns(betas)
    j = int(np.argmax(d))
    return float(d[j]), j


def cross_validated_peak_effect(betas: np.ndarray, n_folds: int = 5, rng: Optional[np.random.Generator] = None) -> float:
    """K-fold cross-validated peak effect.

    For each fold, the peak feature is selected on the training subjects (max d) and the effect is
    estimated on the held-out subjects at that feature; the fold estimates are averaged with weights
    proportional to fold size. Selection and estimation never share subjects.
    """
    betas = np.asarray(betas, dtype=float)
    n = betas.shape[0]
    if n < 2 * n_folds:
        raise ValueError("need at least 2 subjects per fold")
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    total = 0.0
    for test_idx in folds:
        train_idx = np.setdiff1d(perm, test_idx)
        _, j = circular_peak_effect(betas[train_idx])
        held = betas[test_idx, j]
        sd = held.std(ddof=1)
        d_test = held.mean() / sd if sd > 0 else 0.0
        total += d_test * len(test_idx)
    return float(total / n)


def split_half_peak_inflation(betas: np.ndarray, rng: Optional[np.random.Generator] = None, n_splits: int = 50) -> PeakInflation:
    """Average over random split-halves: select the peak on half A, estimate on half B.

    ``d_circular`` is the in-sample peak d on the full sample; ``d_cv`` is the mean held-out effect.
    """
    betas = np.asarray(betas, dtype=float)
    rng = rng or np.random.default_rng(0)
    n, p = betas.shape
    d_circ, j_full = circular_peak_effect(betas)
    held = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2 :]
        _, j = circular_peak_effect(betas[a])
        col = betas[b, j]
        sd = col.std(ddof=1)
        held.append(col.mean() / sd if sd > 0 else 0.0)
    return PeakInflation(d_circular=d_circ, d_cv=float(np.mean(held)), peak_index=j_full, n_subjects=n, n_features=p)
