"""Learning indices, learner classification and specification-curve inference.

Two indices dominate the neurofeedback literature: the *within-session* slope
of the feedback signal over training blocks and the *session-to-session*
slope of session means.  Both, and the "learner" label derived from them, are
computed here for one specification; :func:`specification_curve` and
:func:`specification_curve_test` summarise a whole multiverse of them and give
a joint permutation test in the spirit of Simonsohn, Simmons & Nelson (2020):
under the null of no learning, the sign of each participant's learning effect
is exchangeable, so signs are flipped *jointly across all specifications* to
preserve their dependence.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def block_means(times: np.ndarray, values: np.ndarray, block_edges: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Mean feedback value per training block (NaN-aware)."""
    out = np.full(len(block_edges), np.nan)
    for i, (s0, s1) in enumerate(block_edges):
        m = (times >= s0) & (times < s1) & np.isfinite(values)
        if m.any():
            out[i] = float(np.mean(values[m]))
    return out


def within_session_slope(block_values: np.ndarray) -> Dict[str, float]:
    """OLS slope of block means against block index (per block), with p-value and r."""
    v = np.asarray(block_values, float)
    x = np.arange(len(v), dtype=float)
    m = np.isfinite(v)
    if m.sum() < 3:
        return {"slope": np.nan, "p": np.nan, "r": np.nan, "n_blocks": float(m.sum())}
    res = stats.linregress(x[m], v[m])
    return {"slope": float(res.slope), "p": float(res.pvalue), "r": float(res.rvalue), "n_blocks": float(m.sum())}


def session_to_session_slope(session_means: np.ndarray) -> Dict[str, float]:
    """Same as :func:`within_session_slope` but across sessions."""
    out = within_session_slope(session_means)
    out["n_sessions"] = out.pop("n_blocks")
    return out


def classify_learner(slope: float, p: float, alpha: float = 0.05, direction: str = "up") -> bool:
    """A participant is a 'learner' if the slope is significant in the trained direction."""
    if not np.isfinite(slope) or not np.isfinite(p):
        return False
    ok = slope > 0 if direction == "up" else slope < 0
    return bool(ok and p < alpha)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def learner_agreement(labels: np.ndarray) -> Dict[str, float]:
    """Stability of learner labels across specifications.

    ``labels`` is (n_specs, n_subjects) boolean.  Returns the mean and minimum
    pairwise kappa between specifications, the fraction of participants whose
    label is unanimous, and the fraction whose label flips in >= 25% of specs.
    """
    L = np.asarray(labels, bool)
    n_specs, n_subj = L.shape
    kappas = []
    for i in range(n_specs):
        for j in range(i + 1, n_specs):
            kappas.append(cohen_kappa(L[i], L[j]))
    share = L.mean(0)
    return {
        "mean_pairwise_kappa": float(np.mean(kappas)) if kappas else np.nan,
        "min_pairwise_kappa": float(np.min(kappas)) if kappas else np.nan,
        "fraction_unanimous": float(np.mean((share == 0) | (share == 1))),
        "fraction_unstable": float(np.mean((share >= 0.25) & (share <= 0.75))),
        "learner_rate_range": float(L.mean(1).max() - L.mean(1).min()),
    }


def specification_curve(effects: np.ndarray, spec_names: Optional[Sequence[str]] = None, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per-specification summary of participant-level effects (``effects``: n_specs x n_subjects).

    Columns: mean effect, bootstrap 95% CI over participants, one-sample t p-value,
    fraction of participants with a positive effect, rank by mean.
    """
    E = np.asarray(effects, float)
    rng = np.random.default_rng(seed)
    n_specs, n_subj = E.shape
    rows = []
    for i in range(n_specs):
        e = E[i][np.isfinite(E[i])]
        boot = np.array([np.mean(e[rng.integers(0, len(e), len(e))]) for _ in range(n_boot)]) if len(e) > 1 else np.array([np.nan])
        t = stats.ttest_1samp(e, 0.0) if len(e) > 2 else None
        rows.append(
            {
                "spec_id": i,
                "name": spec_names[i] if spec_names is not None else str(i),
                "mean": float(np.mean(e)) if len(e) else np.nan,
                "median": float(np.median(e)) if len(e) else np.nan,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "p": float(t.pvalue) if t is not None else np.nan,
                "frac_positive": float(np.mean(e > 0)) if len(e) else np.nan,
                "n": int(len(e)),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    return df


def specification_curve_test(effects: np.ndarray, n_perm: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Joint permutation (sign-flip) test over the whole specification curve.

    Three test statistics (Simonsohn et al., 2020): the median across
    specifications of the mean effect, the share of specifications with a
    positive mean, and the share of specifications individually significant.
    Under H0 each participant's effect sign is flipped with probability 1/2,
    *jointly for all specifications*.
    """
    E = np.asarray(effects, float)
    rng = np.random.default_rng(seed)
    n_specs, n_subj = E.shape

    def stats_of(M: np.ndarray) -> Tuple[float, float, float]:
        means = np.nanmean(M, axis=1)
        med = float(np.nanmedian(means))
        share_pos = float(np.mean(means > 0))
        sig = 0.0
        for i in range(n_specs):
            e = M[i][np.isfinite(M[i])]
            if len(e) > 2:
                sig += float(stats.ttest_1samp(e, 0.0).pvalue < alpha and np.mean(e) > 0)
        return med, share_pos, sig / n_specs

    obs = stats_of(E)
    null = np.empty((n_perm, 3))
    for k in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=n_subj)
        null[k] = stats_of(E * flips[None, :])
    p = [(np.sum(null[:, j] >= obs[j]) + 1) / (n_perm + 1) for j in range(3)]
    return {
        "median_effect": obs[0],
        "p_median": float(p[0]),
        "share_positive": obs[1],
        "p_share_positive": float(p[1]),
        "share_significant": obs[2],
        "p_share_significant": float(p[2]),
        "n_specs": float(n_specs),
        "n_subjects": float(n_subj),
    }
