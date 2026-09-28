"""Longitudinal internal-validity metrics for cross-sectional subtype-and-stage models.

All metrics take a long table with one row per session (``subject``, ``time`` in years, model outputs
``stage`` / ``subtype`` / ``subtype_prob``) and come with an explicit null:

* stage monotonicity: within-subject permutation of the session order (keeps each subject's set of
  stages, destroys their temporal order);
* subtype stability: the marginal-frequency null Σ f_k² (agreement expected if subtypes were assigned
  at random with the observed frequencies), plus a within-subject permutation across sessions of
  *different* subjects is not meaningful, so κ against the marginal null is reported.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _pairs(df: pd.DataFrame, subject: str, time: str, value: str) -> pd.DataFrame:
    d = df.sort_values([subject, time])
    nxt = d.groupby(subject).shift(-1)
    out = pd.DataFrame({"subject": d[subject], "v0": d[value], "v1": nxt[value],
                        "dt": nxt[time] - d[time]}).dropna()
    return out


def stage_monotonicity(df: pd.DataFrame, subject: str = "subject", time: str = "years", stage: str = "stage",
                       n_perm: int = 1000, random_state: int = 0) -> dict[str, float]:
    """Fraction of consecutive pairs with non-decreasing / increasing stage, with a permutation null.

    Returns observed fractions, mean Δstage per year, the null mean of the non-decreasing fraction,
    the excess over null and a one-sided permutation p-value.
    """
    p = _pairs(df, subject, time, stage)
    if len(p) == 0:
        raise ValueError("no consecutive pairs")
    d = (p["v1"] - p["v0"]).to_numpy(float)
    obs_nondec = float((d >= 0).mean())
    obs_inc = float((d > 0).mean())
    rate = float((d / p["dt"].to_numpy(float)).mean())
    rng = np.random.default_rng(random_state)
    dd = df[[subject, time, stage]].copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        # permute stage values within each subject (session-order permutation)
        dd[stage] = dd.groupby(subject)[stage].transform(lambda s: s.to_numpy()[rng.permutation(len(s))])
        pp = _pairs(dd, subject, time, stage)
        null[i] = ((pp["v1"] - pp["v0"]) >= 0).mean()
    p_value = float((np.sum(null >= obs_nondec) + 1) / (n_perm + 1))
    return {"n_pairs": int(len(p)), "n_subjects": int(p["subject"].nunique()), "frac_nondecreasing": obs_nondec,
            "frac_increasing": obs_inc, "mean_stage_change_per_year": rate, "null_frac_nondecreasing": float(null.mean()),
            "excess_over_null": obs_nondec - float(null.mean()), "p_value": p_value}


def subtype_stability(df: pd.DataFrame, subject: str = "subject", time: str = "years", subtype: str = "subtype",
                      confidence: Optional[str] = "subtype_prob", min_confidence: float = 0.0,
                      stage: Optional[str] = "stage", stage_bins: Sequence[float] = (0, 3, 6, np.inf)) -> dict:
    """Agreement of the modal subtype between consecutive sessions, overall and by baseline stage bin.

    ``kappa`` uses the marginal-frequency null p_e = Σ f_k² of the baseline subtypes.
    """
    d = df.sort_values([subject, time])
    nxt = d.groupby(subject).shift(-1)
    pairs = pd.DataFrame({"s0": d[subtype], "s1": nxt[subtype]})
    if confidence and confidence in d:
        pairs["conf0"] = d[confidence]
    if stage and stage in d:
        pairs["stage0"] = d[stage]
    pairs = pairs.dropna(subset=["s1"])
    if confidence and "conf0" in pairs:
        pairs = pairs[pairs["conf0"] >= min_confidence]
    if len(pairs) == 0:
        raise ValueError("no pairs after filtering")
    agree = (pairs["s0"] == pairs["s1"]).to_numpy()
    f = pairs["s0"].value_counts(normalize=True).to_numpy()
    p_e = float(np.sum(f ** 2))
    p_o = float(agree.mean())
    out = {"n_pairs": int(len(pairs)), "agreement": p_o, "expected_agreement": p_e,
           "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan}
    if "stage0" in pairs:
        bins = pd.cut(pairs["stage0"], bins=list(stage_bins), right=False)
        out["by_stage_bin"] = pairs.assign(agree=agree).groupby(bins, observed=True)["agree"].agg(["mean", "size"]).to_dict("index")
    return out


def stage_vs_clinical_change(df: pd.DataFrame, clinical: str, subject: str = "subject", time: str = "years",
                             stage: str = "stage") -> dict[str, float]:
    """Within-subject Spearman correlation between Δstage and Δclinical (e.g. CDR-SB) over consecutive pairs."""
    a = _pairs(df, subject, time, stage).rename(columns={"v0": "s0", "v1": "s1"})
    b = _pairs(df, subject, time, clinical).rename(columns={"v0": "c0", "v1": "c1"})
    m = a.join(b[["c0", "c1"]], how="inner")
    ds, dc = m["s1"] - m["s0"], m["c1"] - m["c0"]
    r = stats.spearmanr(ds, dc)
    return {"n_pairs": int(len(m)), "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)}


def mixed_model_comparison(df: pd.DataFrame, outcome: str, predictors: Sequence[str],
                           subject: str = "subject", covariates: Sequence[str] = ("age", "sex_male")) -> pd.DataFrame:
    """Random-intercept mixed models ``outcome ~ predictor + covariates + (1|subject)`` for each predictor
    (e.g. ``stage`` vs ``z_hippocampus``); returns AIC, slope, p-value per predictor.

    Uses statsmodels MixedLM (ML fit so that AICs are comparable across fixed-effect structures).
    """
    import statsmodels.formula.api as smf

    rows = []
    for pred in predictors:
        cols = [outcome, pred, subject, *[c for c in covariates if c in df]]
        d = df[cols].dropna()
        formula = f"{outcome} ~ {pred}" + "".join(f" + {c}" for c in covariates if c in df)
        res = smf.mixedlm(formula, d, groups=d[subject]).fit(reml=False)
        rows.append({"predictor": pred, "n": int(len(d)), "slope": float(res.params[pred]),
                     "p_value": float(res.pvalues[pred]), "aic": float(res.aic)})
    return pd.DataFrame(rows)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
