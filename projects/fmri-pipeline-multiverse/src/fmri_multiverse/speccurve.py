"""Specification-curve summaries, pipeline-robustness score, variance
decomposition, multiverse inference and the fragility model.

Definitions
-----------
For a *finding* f (dataset x contrast x ROI) analysed under specifications
``s = 1..S`` with group-level effect ``beta_s`` and p-value ``p_s``:

* **PRS (pipeline-robustness score)** = share of specifications with
  ``p_s < alpha`` *and* ``sign(beta_s) == sign(median_s beta_s)``. It ranges
  from 0 (never reproduced) to 1 (found under every reasonable pipeline).
* **Sign consistency** = share of specifications agreeing with the median sign.
* **Analytic SNR** = ``|median beta| / IQR(beta)`` (a continuous stability index).
* **Vibration of effects** (Patel et al., 2015) = ``max beta - min beta``.

Inference follows Simonsohn, Simmons & Nelson (2020, Nat Hum Behav): the
subject-level effects are sign-flipped (one-sample design) or bootstrapped and
the whole multiverse is recomputed under each resample to obtain nulls for
the median effect and the share of significant specifications.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Curve and score
# --------------------------------------------------------------------------- #
def specification_curve(
    results: pd.DataFrame, effect_col: str = "effect", p_col: str = "p", alpha: float = 0.05
) -> pd.DataFrame:
    """Sort specifications by effect and flag significance / sign agreement."""
    df = results.copy()
    med = df[effect_col].median()
    df["rank"] = df[effect_col].rank(method="first").astype(int)
    df["significant"] = df[p_col] < alpha
    df["same_sign_as_median"] = np.sign(df[effect_col]) == np.sign(med)
    return df.sort_values("rank").reset_index(drop=True)


def robustness_score(results: pd.DataFrame, effect_col: str = "effect", p_col: str = "p", alpha: float = 0.05) -> Dict[str, float]:
    """Pipeline-robustness score and companion statistics for one finding."""
    e = results[effect_col].to_numpy(dtype=float)
    p = results[p_col].to_numpy(dtype=float)
    med = np.median(e)
    same = np.sign(e) == np.sign(med)
    sig = p < alpha
    iqr = np.subtract(*np.percentile(e, [75, 25]))
    return {
        "n_specs": int(len(e)),
        "prs": float(np.mean(sig & same)),
        "share_significant": float(sig.mean()),
        "sign_consistency": float(same.mean()),
        "median_effect": float(med),
        "iqr_effect": float(iqr),
        "analytic_snr": float(abs(med) / iqr) if iqr > 0 else np.inf,
        "vibration": float(e.max() - e.min()),
        "min_effect": float(e.min()),
        "max_effect": float(e.max()),
    }


def variance_decomposition(results: pd.DataFrame, effect_col: str, spec_cols: Sequence[str]) -> pd.Series:
    """Share of between-specification variance in the effect attributable to
    each pipeline dimension (eta-squared from one-way sums of squares).

    Main effects only; the remainder (``residual``) contains interactions.
    Dimensions with a single level get 0.
    """
    y = results[effect_col].to_numpy(dtype=float)
    sst = ((y - y.mean()) ** 2).sum()
    out = {}
    for col in spec_cols:
        levels = results[col].astype(str)
        ssb = 0.0
        for lvl in levels.unique():
            m = (levels == lvl).to_numpy()
            ssb += m.sum() * (y[m].mean() - y.mean()) ** 2
        out[col] = float(ssb / sst) if sst > 0 else 0.0
    out["residual"] = float(max(0.0, 1.0 - sum(out.values())))
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Multiverse inference
# --------------------------------------------------------------------------- #
def _spec_stats(E: np.ndarray) -> tuple:
    """Per-spec mean effect and one-sample p across subjects for ``(S, n)``."""
    n = E.shape[1]
    mean = E.mean(axis=1)
    sd = E.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / np.sqrt(n)), 0.0)
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return mean, p


def multiverse_inference(
    subject_effects: np.ndarray, n_perm: int = 1000, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """Joint inference over a multiverse of one-sample designs.

    Parameters
    ----------
    subject_effects
        ``(n_specs, n_subjects)`` first-level effects for one finding under
        every specification (same subjects across rows).

    Returns
    -------
    dict with observed ``median_effect``, ``share_significant``, ``prs``; the
    sign-flip null p-values ``p_median`` (median effect != 0) and
    ``p_share`` (share of significant specs larger than expected under the
    null); and bootstrap CIs for the median effect and the PRS.
    """
    E = np.asarray(subject_effects, dtype=float)
    S, n = E.shape
    rng = np.random.default_rng(seed)
    mean, p = _spec_stats(E)
    med_obs = float(np.median(mean))
    sig = p < alpha
    same = np.sign(mean) == np.sign(med_obs)
    share_obs = float(sig.mean())
    prs_obs = float((sig & same).mean())
    null_med = np.empty(n_perm)
    null_share = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)[None, :]
        m, pp = _spec_stats(E * signs)
        null_med[i] = np.median(m)
        null_share[i] = (pp < alpha).mean()
    p_median = (np.sum(np.abs(null_med) >= abs(med_obs)) + 1) / (n_perm + 1)
    p_share = (np.sum(null_share >= share_obs) + 1) / (n_perm + 1)
    boot_med = np.empty(n_boot)
    boot_prs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m, pp = _spec_stats(E[:, idx])
        boot_med[i] = np.median(m)
        boot_prs[i] = ((pp < alpha) & (np.sign(m) == np.sign(np.median(m)))).mean()
    return {
        "n_specs": S,
        "n_subjects": n,
        "median_effect": med_obs,
        "share_significant": share_obs,
        "prs": prs_obs,
        "p_median": float(p_median),
        "p_share": float(p_share),
        "median_ci_low": float(np.quantile(boot_med, alpha / 2)),
        "median_ci_high": float(np.quantile(boot_med, 1 - alpha / 2)),
        "prs_ci_low": float(np.quantile(boot_prs, alpha / 2)),
        "prs_ci_high": float(np.quantile(boot_prs, 1 - alpha / 2)),
    }


# --------------------------------------------------------------------------- #
# Fragility model
# --------------------------------------------------------------------------- #
FRAGILITY_FEATURES: List[str] = [
    "n_subjects",
    "tr",
    "n_volumes",
    "mean_fd",
    "frac_fd_above",
    "multiband",
    "design_block",
    "n_events_per_run",
    "median_effect_abs",
    "n_conditions",
]


def fit_fragility_model(features: pd.DataFrame, prs: pd.Series, feature_cols: Optional[Sequence[str]] = None, alpha: float = 1.0) -> Dict[str, object]:
    """Predict a finding's PRS from dataset/design characteristics.

    Leave-one-finding-out ridge regression (standardised features). With a
    dozen datasets and a few findings each the model is descriptive: report
    LOO r, standardised coefficients and their bootstrap CIs, not p-values.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in (feature_cols or FRAGILITY_FEATURES) if c in features.columns]
    X = features[cols].astype(float).fillna(features[cols].astype(float).median()).to_numpy()
    y = prs.to_numpy(dtype=float)
    n = len(y)
    loo = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        loo[i] = m.predict(sc.transform(X[[i]]))[0]
    sc = StandardScaler().fit(X)
    full = Ridge(alpha=alpha).fit(sc.transform(X), y)
    r_loo = float(np.corrcoef(loo, y)[0, 1]) if n > 2 and np.std(loo) > 0 else np.nan
    return {
        "features": cols,
        "coef_standardized": pd.Series(full.coef_, index=cols),
        "loo_predictions": loo,
        "loo_r": r_loo,
        "loo_rmse": float(np.sqrt(np.mean((loo - y) ** 2))),
    }


def plot_specification_curve(curve: pd.DataFrame, spec_cols: Sequence[str], effect_col: str = "effect", ax=None):
    """Two-panel specification curve (effects on top, choice indicators below)."""
    import matplotlib.pyplot as plt  # local import: plotting is optional

    if ax is None:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    else:
        ax0, ax1 = ax
    colors = np.where(curve["significant"], "#1f77b4", "#9e9e9e")
    ax0.scatter(curve["rank"], curve[effect_col], c=colors, s=12)
    ax0.axhline(0, color="k", lw=0.5)
    ax0.set_ylabel("effect")
    y = 0
    labels = []
    for col in spec_cols:
        for lvl in sorted(curve[col].astype(str).unique()):
            m = curve[col].astype(str) == lvl
            ax1.scatter(curve.loc[m, "rank"], np.full(m.sum(), y), s=6, c="k")
            labels.append(f"{col}={lvl}")
            y += 1
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("specification rank")
    return ax0, ax1
