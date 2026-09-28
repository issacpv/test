"""Causal-decomposition tools for the subgroup performance gap.

Three candidate mechanisms are separated:

1. **Sample size** - the minority group contributes fewer training subjects
   and the model is tuned to the majority (matched-n retraining).
2. **Motion / scan quality** - head motion differs between groups and
   degrades FC-behaviour associations (Siegel et al., 2017, Cereb Cortex);
   tested by mediation of |error| through mean framewise displacement and by
   motion-matched evaluation.
3. **Label reliability / label bias** - cognitive test scores may be less
   reliable or differently valid in a subgroup, bounding attainable r by
   ``sqrt(rel_x * rel_y)`` (attenuation ceiling).

What remains after these is attributed to distribution shift in the
brain-behaviour mapping itself (Greene et al., 2022 "stereotype" effect).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .subgroup_metrics import basic_metrics


# --------------------------------------------------------------------------- #
# OLS helpers
# --------------------------------------------------------------------------- #
def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta[1:]


def mediation_by_motion(
    group: Sequence,
    mediator: Sequence,
    outcome: Sequence,
    covariates: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Product-of-coefficients mediation with percentile-bootstrap CI.

    Model: ``mediator = a * group + cov``; ``outcome = c' * group + b * mediator + cov``.
    Indirect effect = ``a * b``; total = ``c' + a * b``.

    Parameters
    ----------
    group
        Binary indicator (1 = minority / group of interest).
    mediator
        e.g. mean framewise displacement, tSNR, or a scan-quality composite.
    outcome
        Per-subject prediction error (e.g. ``|y_hat - y|`` or signed residual).
    covariates
        Optional ``(n, k)`` matrix (age, sex, ...).

    Returns
    -------
    dict with ``a, b, c_prime, indirect, total, prop_mediated, ci_low, ci_high``.
    """
    g = np.asarray(group, dtype=float)
    m = np.asarray(mediator, dtype=float)
    o = np.asarray(outcome, dtype=float)
    C = None if covariates is None else np.asarray(covariates, dtype=float)

    def _fit(idx: np.ndarray) -> Tuple[float, float, float]:
        Xa = g[idx, None] if C is None else np.column_stack([g[idx], C[idx]])
        a = _ols(m[idx], Xa)[0]
        Xb = np.column_stack([g[idx], m[idx]]) if C is None else np.column_stack([g[idx], m[idx], C[idx]])
        cb = _ols(o[idx], Xb)
        return float(a), float(cb[1]), float(cb[0])

    idx_all = np.arange(len(g))
    a, b, c_prime = _fit(idx_all)
    indirect = a * b
    total = c_prime + indirect
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(g), size=len(g))
        if len(np.unique(g[idx])) < 2:
            continue
        ab, bb, _ = _fit(idx)
        boots.append(ab * bb)
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if len(boots) else (np.nan, np.nan)
    return {
        "a": a,
        "b": b,
        "c_prime": c_prime,
        "indirect": indirect,
        "total": total,
        "prop_mediated": float(indirect / total) if total != 0 else np.nan,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())) if len(boots) else np.nan,
    }


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def motion_matched_subsample(
    groups: Sequence,
    motion: Sequence,
    reference: object,
    target: object,
    caliper: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour matching (without replacement) of reference-group
    subjects to target-group subjects on a scalar (motion).

    Returns ``(target_idx, matched_reference_idx)`` of equal length.
    """
    groups = np.asarray(groups)
    motion = np.asarray(motion, dtype=float)
    rng = np.random.default_rng(seed)
    tgt = np.where(groups == target)[0]
    ref = np.where(groups == reference)[0]
    rng.shuffle(tgt)
    available = np.ones(len(ref), dtype=bool)
    matched_t, matched_r = [], []
    for i in tgt:
        d = np.abs(motion[ref] - motion[i])
        d[~available] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or (caliper is not None and d[j] > caliper):
            continue
        available[j] = False
        matched_t.append(i)
        matched_r.append(ref[j])
    return np.asarray(matched_t), np.asarray(matched_r)


# --------------------------------------------------------------------------- #
# Attenuation ceiling
# --------------------------------------------------------------------------- #
def attenuation_ceiling(reliability_features: float, reliability_target: float) -> float:
    """Maximum attainable Pearson r given test-retest reliabilities (Spearman, 1904)."""
    return float(np.sqrt(max(reliability_features, 0.0) * max(reliability_target, 0.0)))


def disattenuate(r: float, reliability_features: float, reliability_target: float) -> float:
    """Correct an observed r for attenuation; clipped to [-1, 1]."""
    ceil = attenuation_ceiling(reliability_features, reliability_target)
    if ceil == 0:
        return np.nan
    return float(np.clip(r / ceil, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Sequential gap decomposition
# --------------------------------------------------------------------------- #
def matched_n_reference_performance(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    reference: object,
    target: object,
    test_idx: np.ndarray,
    train_idx: np.ndarray,
    n_draws: int = 20,
    seed: int = 0,
    metric: str = "r",
) -> Dict[str, float]:
    """Train on reference-group subjects only, at (i) full n and (ii) n equal to
    the target group's training size, and evaluate on the reference test subjects.

    The drop from (i) to (ii) is the *sample-size* component of the gap: the
    accuracy a model of the target group could reach if its brain-behaviour
    mapping were identical to the reference group's, but only ``n_target``
    training subjects were available.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    tr_ref = train_idx[groups[train_idx] == reference]
    n_target = int((groups[train_idx] == target).sum())
    te_ref = test_idx[groups[test_idx] == reference]
    if n_target == 0 or len(tr_ref) < n_target or len(te_ref) < 3:
        return {"full_n": np.nan, "matched_n": np.nan, "n_target": float(n_target)}
    est = clone(estimator).fit(X[tr_ref], y[tr_ref])
    full = basic_metrics(y[te_ref], est.predict(X[te_ref]))[metric]
    vals = []
    for _ in range(n_draws):
        sub = rng.choice(tr_ref, size=n_target, replace=False)
        est = clone(estimator).fit(X[sub], y[sub])
        vals.append(basic_metrics(y[te_ref], est.predict(X[te_ref]))[metric])
    return {"full_n": float(full), "matched_n": float(np.nanmean(vals)), "n_target": float(n_target)}


def decompose_gap(
    y: np.ndarray,
    y_hat: np.ndarray,
    groups: Sequence,
    motion: Sequence,
    reference: object,
    target: object,
    matched_n_result: Optional[Dict[str, float]] = None,
    reliability: Optional[Dict[object, Tuple[float, float]]] = None,
    metric: str = "r",
    n_draws: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sequential decomposition of ``metric(reference) - metric(target)``.

    Components (all in metric units):

    * ``sample_size`` - ``full_n - matched_n`` from
      :func:`matched_n_reference_performance` (0 if not supplied).
    * ``motion`` - reference performance on all reference subjects minus
      reference performance on motion-matched reference subjects
      (averaged over ``n_draws`` matchings).
    * ``label_reliability`` - difference in attenuation ceilings scaled by the
      reference performance, if ``reliability = {group: (rel_x, rel_y)}`` given.
    * ``residual`` - remainder (distribution shift in the mapping, label bias
      not captured by reliability, unmeasured quality factors).

    The decomposition is additive and order-dependent (sample size, then
    motion, then reliability); report sensitivity to order in the paper.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    groups = np.asarray(groups)
    motion = np.asarray(motion, dtype=float)
    ref_mask = groups == reference
    tgt_mask = groups == target
    perf_ref = basic_metrics(y[ref_mask], y_hat[ref_mask])[metric]
    perf_tgt = basic_metrics(y[tgt_mask], y_hat[tgt_mask])[metric]
    total_gap = perf_ref - perf_tgt

    size_comp = 0.0
    if matched_n_result is not None and np.isfinite(matched_n_result.get("full_n", np.nan)):
        size_comp = matched_n_result["full_n"] - matched_n_result["matched_n"]

    vals = []
    for d in range(n_draws):
        _, ref_idx = motion_matched_subsample(groups, motion, reference, target, seed=seed + d)
        if len(ref_idx) >= 3:
            vals.append(basic_metrics(y[ref_idx], y_hat[ref_idx])[metric])
    motion_comp = perf_ref - float(np.nanmean(vals)) if vals else np.nan

    rel_comp = 0.0
    if reliability is not None and reference in reliability and target in reliability:
        ceil_ref = attenuation_ceiling(*reliability[reference])
        ceil_tgt = attenuation_ceiling(*reliability[target])
        if ceil_ref > 0:
            rel_comp = perf_ref * (1 - ceil_tgt / ceil_ref)

    residual = total_gap - size_comp - (0.0 if not np.isfinite(motion_comp) else motion_comp) - rel_comp
    rows = {
        "total_gap": total_gap,
        "sample_size": size_comp,
        "motion": motion_comp,
        "label_reliability": rel_comp,
        "residual": residual,
    }
    out = pd.DataFrame({"component": list(rows.keys()), "value": list(rows.values())})
    out["share_of_gap"] = out["value"] / total_gap if total_gap != 0 else np.nan
    out.attrs.update({"perf_reference": perf_ref, "perf_target": perf_tgt, "metric": metric})
    return out
