"""Concept-cell selection criteria, permutation nulls and selection-bias correction.

Every criterion has the signature ``f(rates_or_counts, stim, **kw) -> CriterionResult`` and
operates on one unit. ``rates`` is a 1-D array (one value per trial), ``stim`` a 1-D array of
stimulus labels of the same length. Bin-wise criteria take the ``(n_trials, n_bins)`` matrix
produced by :func:`concept_cells.nwb_loader.binned_counts` plus baseline counts.

Criteria
--------
C1 ``anova_criterion``             one-way ANOVA across stimulus identities.
C2 ``kruskal_criterion``           Kruskal-Wallis across stimulus identities.
C3 ``binwise_ranksum_criterion``   per-stimulus, per-bin rank-sum vs baseline, Holm over bins.
C4 ``response_strength_criterion`` best stimulus z-score and rate ratio vs the others.
C5 ``poisson_glm_criterion``       Poisson GLM LR test of a stimulus factor with a drift spline.

Nulls
-----
``permutation_pvalue`` supports ``"shuffle"`` (i.i.d. label permutation), ``"circular"``
(circular shift of the label sequence, preserving both autocorrelations) and ``"block"``
(permute labels within blocks or permute whole blocks).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

Criterion = Callable[..., "CriterionResult"]


@dataclass
class CriterionResult:
    """Outcome of one criterion for one unit."""

    name: str
    statistic: float
    pvalue: float
    selected: bool
    preferred: Optional[object] = None
    index: float = float("nan")  #: in-sample selectivity index (best vs rest), see :func:`selectivity_index`

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def _groups(values: np.ndarray, stim: np.ndarray, min_trials: int) -> List[np.ndarray]:
    labels = np.unique(stim)
    groups = [values[stim == lab] for lab in labels]
    return [g for g in groups if len(g) >= min_trials]


def selectivity_index(rates: np.ndarray, stim: np.ndarray, preferred: object) -> float:
    """(best - rest) / (best + rest) using mean rates; nan when both are zero."""
    rates = np.asarray(rates, dtype=float)
    stim = np.asarray(stim)
    best = rates[stim == preferred].mean() if np.any(stim == preferred) else np.nan
    rest = rates[stim != preferred].mean() if np.any(stim != preferred) else np.nan
    denom = best + rest
    return float((best - rest) / denom) if denom > 0 else float("nan")


def preferred_stimulus(rates: np.ndarray, stim: np.ndarray) -> object:
    labels, idx = np.unique(stim, return_inverse=True)
    means = np.bincount(idx, weights=rates, minlength=len(labels)) / np.maximum(np.bincount(idx, minlength=len(labels)), 1)
    return labels[int(np.argmax(means))]


# --------------------------------------------------------------------------- C1 / C2
def anova_criterion(rates: np.ndarray, stim: np.ndarray, alpha: float = 0.05, min_trials: int = 5) -> CriterionResult:
    """C1: one-way ANOVA of window firing rates across stimulus identities."""
    rates = np.asarray(rates, dtype=float)
    stim = np.asarray(stim)
    groups = _groups(rates, stim, min_trials)
    if len(groups) < 2 or np.all(rates == rates[0]):
        return CriterionResult("anova", float("nan"), 1.0, False)
    f, p = stats.f_oneway(*groups)
    pref = preferred_stimulus(rates, stim)
    return CriterionResult("anova", float(f), float(p), bool(p < alpha), pref, selectivity_index(rates, stim, pref))


def kruskal_criterion(rates: np.ndarray, stim: np.ndarray, alpha: float = 0.05, min_trials: int = 5) -> CriterionResult:
    """C2: Kruskal-Wallis test across stimulus identities."""
    rates = np.asarray(rates, dtype=float)
    stim = np.asarray(stim)
    groups = _groups(rates, stim, min_trials)
    if len(groups) < 2 or np.all(rates == rates[0]):
        return CriterionResult("kruskal", float("nan"), 1.0, False)
    h, p = stats.kruskal(*groups)
    pref = preferred_stimulus(rates, stim)
    return CriterionResult("kruskal", float(h), float(p), bool(p < alpha), pref, selectivity_index(rates, stim, pref))


# --------------------------------------------------------------------------- C3
def holm_reject(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down rejections for a vector of p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresholds = alpha / (m - np.arange(m))
    passed = p[order] <= thresholds
    # step-down: once a test fails, all subsequent (larger p) fail
    if not passed.all():
        first_fail = int(np.argmin(passed))
        passed[first_fail:] = False
    out = np.zeros(m, dtype=bool)
    out[order] = passed
    return out


def binwise_ranksum_criterion(binned: np.ndarray, baseline: np.ndarray, stim: np.ndarray, alpha: float = 0.05,
                              min_trials: int = 5, min_bins: int = 1) -> CriterionResult:
    """C3: for each stimulus, rank-sum test of each bin's counts vs the baseline counts.

    A stimulus "responds" when at least ``min_bins`` bins survive Holm correction over bins.
    The unit is selected when any stimulus responds. ``baseline`` is a 1-D array of baseline
    counts per trial (all trials pooled as the reference distribution, as in the bin-wise
    approach of Mormann et al., 2008, J Neurosci; exact implementation details differ between labs).
    """
    binned = np.asarray(binned, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    stim = np.asarray(stim)
    if binned.ndim != 2 or len(stim) != binned.shape[0]:
        raise ValueError("binned must be (n_trials, n_bins) and match stim")
    best_p, best_stim, n_sig_bins_best = 1.0, None, 0
    for lab in np.unique(stim):
        rows = binned[stim == lab]
        if len(rows) < min_trials:
            continue
        pvals = np.ones(binned.shape[1])
        for b in range(binned.shape[1]):
            col = rows[:, b]
            if np.all(col == col[0]) and np.all(baseline == col[0]):
                continue
            pvals[b] = stats.mannwhitneyu(col, baseline, alternative="greater").pvalue
        rej = holm_reject(pvals, alpha)
        n_sig = int(rej.sum())
        pmin = float(np.min(pvals) * binned.shape[1])  # Bonferroni-style summary p for reporting
        if n_sig >= min_bins and (n_sig > n_sig_bins_best or (n_sig == n_sig_bins_best and pmin < best_p)):
            best_p, best_stim, n_sig_bins_best = min(pmin, 1.0), lab, n_sig
    selected = best_stim is not None
    rates = binned.sum(axis=1)
    pref = best_stim if selected else preferred_stimulus(rates, stim)
    return CriterionResult("binwise_ranksum", float(n_sig_bins_best), float(best_p if selected else 1.0), selected,
                           pref, selectivity_index(rates, stim, pref))


# --------------------------------------------------------------------------- C4
def response_strength_criterion(rates: np.ndarray, stim: np.ndarray, z_thresh: float = 3.0, ratio_thresh: float = 2.0,
                                min_trials: int = 5) -> CriterionResult:
    """C4: best-stimulus mean rate vs the pooled other trials: z >= ``z_thresh`` and ratio >= ``ratio_thresh``.

    The z-score is (mean_best - mean_rest) / SE with the pooled SE of the two groups. The p-value
    reported is the one-sided normal tail of z (approximate; use :func:`permutation_pvalue` for
    a calibrated value).
    """
    rates = np.asarray(rates, dtype=float)
    stim = np.asarray(stim)
    labels = np.unique(stim)
    best_z, best_ratio, best_lab = -np.inf, 0.0, None
    for lab in labels:
        a = rates[stim == lab]
        b = rates[stim != lab]
        if len(a) < min_trials or len(b) < min_trials:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        z = (a.mean() - b.mean()) / se if se > 0 else (np.inf if a.mean() > b.mean() else 0.0)
        ratio = a.mean() / b.mean() if b.mean() > 0 else (np.inf if a.mean() > 0 else 0.0)
        if z > best_z:
            best_z, best_ratio, best_lab = float(z), float(ratio), lab
    if best_lab is None:
        return CriterionResult("response_strength", float("nan"), 1.0, False)
    selected = bool(best_z >= z_thresh and best_ratio >= ratio_thresh)
    p = float(stats.norm.sf(best_z)) if np.isfinite(best_z) else 0.0
    return CriterionResult("response_strength", best_z, p, selected, best_lab, selectivity_index(rates, stim, best_lab))


# --------------------------------------------------------------------------- C5
def _bspline_basis(x: np.ndarray, n_knots: int = 4, degree: int = 3) -> np.ndarray:
    """Simple B-spline basis on [0, 1] via scipy (columns are basis functions, intercept excluded)."""
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=float)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = np.zeros_like(x)
    inner = np.linspace(0, 1, n_knots + 2)[1:-1]
    t = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    n_basis = len(t) - degree - 1
    basis = np.empty((len(x), n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        basis[:, j] = BSpline(t, coef, degree, extrapolate=False)(x)
    basis = np.nan_to_num(basis)
    return basis[:, 1:]  # drop one column to avoid collinearity with the intercept


def poisson_glm_criterion(counts: np.ndarray, stim: np.ndarray, trial_index: Optional[np.ndarray] = None,
                          alpha: float = 0.05, min_trials: int = 5, drift_knots: int = 4) -> CriterionResult:
    """C5: Poisson GLM likelihood-ratio test for a stimulus factor, adjusting for slow drift.

    Model: ``log E[count] = intercept + spline(trial_index) + stimulus``. The LR test compares
    the model with and without the stimulus dummies; drift is absorbed by the spline so that a
    unit whose rate merely trends over a blocked session is not called selective.
    """
    import statsmodels.api as sm

    counts = np.asarray(counts, dtype=float)
    stim = np.asarray(stim)
    n = len(counts)
    trial_index = np.arange(n) if trial_index is None else np.asarray(trial_index, dtype=float)
    labels, idx = np.unique(stim, return_inverse=True)
    keep_labels = [k for k in range(len(labels)) if np.sum(idx == k) >= min_trials]
    if len(keep_labels) < 2 or counts.sum() == 0:
        return CriterionResult("poisson_glm", float("nan"), 1.0, False)
    mask = np.isin(idx, keep_labels)
    counts, idx, trial_index = counts[mask], idx[mask], trial_index[mask]
    drift = _bspline_basis(trial_index, n_knots=drift_knots)
    x0 = sm.add_constant(drift, has_constant="add")
    dummies = np.stack([(idx == k).astype(float) for k in keep_labels[1:]], axis=1)
    x1 = np.hstack([x0, dummies])
    try:
        m0 = sm.GLM(counts, x0, family=sm.families.Poisson()).fit()
        m1 = sm.GLM(counts, x1, family=sm.families.Poisson()).fit()
    except Exception:  # noqa: BLE001 - singular designs on degenerate synthetic units
        return CriterionResult("poisson_glm", float("nan"), 1.0, False)
    lr = 2.0 * (m1.llf - m0.llf)
    df = dummies.shape[1]
    p = float(stats.chi2.sf(max(lr, 0.0), df))
    pref = preferred_stimulus(counts, stim[mask])
    return CriterionResult("poisson_glm", float(lr), p, bool(p < alpha), pref, selectivity_index(counts, stim[mask], pref))


# --------------------------------------------------------------------------- nulls
def _null_labels(stim: np.ndarray, rng: np.random.Generator, null: str, block: Optional[np.ndarray]) -> np.ndarray:
    if null == "shuffle":
        return rng.permutation(stim)
    if null == "circular":
        shift = int(rng.integers(1, len(stim)))
        return np.roll(stim, shift)
    if null == "block":
        if block is None:
            raise ValueError("block null needs a block vector")
        out = stim.copy()
        blocks = np.unique(block)
        perm = rng.permutation(blocks)  # permute whole blocks (keeps within-block order)
        for src, dst in zip(blocks, perm):
            out[block == dst] = stim[block == src][: np.sum(block == dst)] if np.sum(block == src) >= np.sum(block == dst) \
                else np.resize(stim[block == src], np.sum(block == dst))
        return out
    raise ValueError(f"unknown null {null!r}")


def permutation_pvalue(criterion: Criterion, values: np.ndarray, stim: np.ndarray, n_perm: int = 500,
                       null: str = "circular", block: Optional[np.ndarray] = None, seed: int = 0,
                       stat_larger_is_more_selective: bool = True, **kw) -> Dict[str, float]:
    """Empirical p-value of a criterion's statistic under a label null.

    Returns ``{"observed": stat, "p_perm": p, "null_mean": m, "null_sd": s}``. Uses the
    criterion's own ``statistic`` (F, H, z, LR ...); for criteria whose statistic is not
    monotone in selectivity pass ``stat_larger_is_more_selective=False``.
    """
    rng = np.random.default_rng(seed)
    stim = np.asarray(stim)
    obs = criterion(values, stim, **kw).statistic
    if not np.isfinite(obs):
        return {"observed": float("nan"), "p_perm": 1.0, "null_mean": float("nan"), "null_sd": float("nan")}
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        res = criterion(values, _null_labels(stim, rng, null, block), **kw)
        null_stats[i] = res.statistic if np.isfinite(res.statistic) else -np.inf
    if stat_larger_is_more_selective:
        p = (1.0 + np.sum(null_stats >= obs)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(null_stats <= obs)) / (n_perm + 1.0)
    finite = null_stats[np.isfinite(null_stats)]
    return {"observed": float(obs), "p_perm": float(p),
            "null_mean": float(finite.mean()) if finite.size else float("nan"),
            "null_sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan")}


def drift_index(rates: np.ndarray) -> float:
    """Spearman correlation between trial index and rate: a cheap non-stationarity index."""
    rates = np.asarray(rates, dtype=float)
    if len(rates) < 3 or np.all(rates == rates[0]):
        return 0.0
    return float(stats.spearmanr(np.arange(len(rates)), rates).correlation)


# --------------------------------------------------------------------------- selection bias
def split_half_selectivity(rates: np.ndarray, stim: np.ndarray, n_splits: int = 50, seed: int = 0,
                           min_trials: int = 2) -> Dict[str, float]:
    """Cross-validated selectivity: choose the preferred stimulus on one half, quantify on the other.

    Returns the in-sample index, the mean cross-validated index over random stratified splits
    (both directions), and the shrinkage ``1 - cv / in_sample`` (nan when in-sample <= 0).
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates, dtype=float)
    stim = np.asarray(stim)
    pref_all = preferred_stimulus(rates, stim)
    in_sample = selectivity_index(rates, stim, pref_all)
    labels = np.unique(stim)
    cv_vals: List[float] = []
    for _ in range(n_splits):
        half_a = np.zeros(len(stim), dtype=bool)
        for lab in labels:
            ix = np.flatnonzero(stim == lab)
            if len(ix) < 2 * min_trials:
                continue
            chosen = rng.choice(ix, size=len(ix) // 2, replace=False)
            half_a[chosen] = True
        half_b = ~half_a
        if half_a.sum() < 2 or half_b.sum() < 2:
            continue
        for train, test in ((half_a, half_b), (half_b, half_a)):
            pref = preferred_stimulus(rates[train], stim[train])
            if not np.any(stim[test] == pref):
                continue
            cv_vals.append(selectivity_index(rates[test], stim[test], pref))
    cv = float(np.nanmean(cv_vals)) if cv_vals else float("nan")
    shrink = float(1.0 - cv / in_sample) if in_sample > 0 and np.isfinite(cv) else float("nan")
    return {"in_sample": float(in_sample), "cross_validated": cv, "shrinkage": shrink, "n_splits_used": len(cv_vals) // 2}


# --------------------------------------------------------------------------- multiverse driver
def run_multiverse(rates: np.ndarray, counts: np.ndarray, binned: np.ndarray, baseline: np.ndarray, stim: np.ndarray,
                   alpha: float = 0.05, min_trials: int = 5, criteria: Sequence[str] = ("anova", "kruskal", "binwise_ranksum",
                                                                                          "response_strength", "poisson_glm")) -> Dict[str, CriterionResult]:
    """Apply all criteria to one unit and return ``{criterion_name: CriterionResult}``.

    ``rates`` are window rates (Hz), ``counts`` the integer window counts (for the GLM), ``binned``
    the trials x bins matrix and ``baseline`` the per-trial baseline counts.
    """
    out: Dict[str, CriterionResult] = {}
    if "anova" in criteria:
        out["anova"] = anova_criterion(rates, stim, alpha, min_trials)
    if "kruskal" in criteria:
        out["kruskal"] = kruskal_criterion(rates, stim, alpha, min_trials)
    if "binwise_ranksum" in criteria:
        out["binwise_ranksum"] = binwise_ranksum_criterion(binned, baseline, stim, alpha, min_trials)
    if "response_strength" in criteria:
        out["response_strength"] = response_strength_criterion(rates, stim, min_trials=min_trials)
    if "poisson_glm" in criteria:
        out["poisson_glm"] = poisson_glm_criterion(counts, stim, alpha=alpha, min_trials=min_trials)
    return out


__all__ = [
    "CriterionResult", "selectivity_index", "preferred_stimulus", "anova_criterion", "kruskal_criterion",
    "holm_reject", "binwise_ranksum_criterion", "response_strength_criterion", "poisson_glm_criterion",
    "permutation_pvalue", "drift_index", "split_half_selectivity", "run_multiverse",
]
