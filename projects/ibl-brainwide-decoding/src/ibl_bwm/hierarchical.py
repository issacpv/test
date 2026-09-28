"""Hierarchical / meta-analytic summaries of (session, region) decoding scores across labs.

* `dersimonian_laird`  random-effects pooled effect with τ², I², Q.
* `icc_oneway`         method-of-moments lab ICC (no dependency).
* `mixed_model_icc`    statsmodels MixedLM: score ~ covariates + (1|lab) + (1|subject) → variance components.
* `leave_one_lab_out`  fragility of a region-level claim to the removal of any single lab.
* `region_table`       run everything per region and assemble the reproducibility table.

Input convention: a long DataFrame with one row per (eid/pid, region) and columns
`region, lab, subject, corrected, null_sd, n_units, n_trials` (from `decoding.null_corrected` +
`binning.unit_yield_table`).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- meta-analysis
def dersimonian_laird(effects: Sequence[float], variances: Sequence[float]) -> Dict[str, float]:
    """Random-effects meta-analysis (DerSimonian & Laird 1986). Returns pooled effect, SE, z, p, tau2, I2, Q."""
    y = np.asarray(effects, float)
    v = np.asarray(variances, float)
    m = ~np.isnan(y) & ~np.isnan(v) & (v > 0)
    y, v = y[m], v[m]
    k = len(y)
    if k == 0:
        return {k_: np.nan for k_ in ("effect", "se", "z", "p", "tau2", "I2", "Q", "k")}
    w = 1.0 / v
    fixed = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - fixed) ** 2))
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if k > 1 and C > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    eff = float(np.sum(w_re * y) / np.sum(w_re))
    se = float(np.sqrt(1.0 / np.sum(w_re)))
    z = eff / se
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return {"effect": eff, "se": se, "z": float(z), "p": float(stats.norm.sf(z)),  # one-sided: effect > 0
            "tau2": float(tau2), "I2": float(I2), "Q": Q, "k": k}


# ----------------------------------------------------------------------------- ICC
def icc_oneway(values: Sequence[float], groups: Sequence) -> Dict[str, float]:
    """ICC(1) from a one-way random-effects ANOVA (method of moments), e.g. lab as the grouping factor."""
    y = np.asarray(values, float)
    g = pd.Categorical(np.asarray(groups))
    m = ~np.isnan(y)
    y, codes = y[m], g.codes[m]
    k = len(np.unique(codes))
    N = len(y)
    if k < 2 or N <= k:
        return {"icc": np.nan, "var_between": np.nan, "var_within": np.nan, "k": k, "n": N}
    grand = y.mean()
    ns = np.bincount(codes)
    ns = ns[ns > 0]
    means = np.array([y[codes == c].mean() for c in np.unique(codes)])
    ss_b = np.sum(ns * (means - grand) ** 2)
    ss_w = np.sum((y - means[np.searchsorted(np.unique(codes), codes)]) ** 2)
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    n0 = (N - np.sum(ns ** 2) / N) / (k - 1)
    var_b = max(0.0, (ms_b - ms_w) / n0)
    icc = var_b / (var_b + ms_w) if (var_b + ms_w) > 0 else np.nan
    return {"icc": float(icc), "var_between": float(var_b), "var_within": float(ms_w), "k": int(k), "n": int(N)}


def bootstrap_icc(values: Sequence[float], groups: Sequence, n_boot: int = 500, seed: int = 0) -> Dict[str, float]:
    """Cluster bootstrap (resample labs with replacement) CI for ICC(1)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(values, float)
    g = np.asarray(groups)
    labs = np.unique(g)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(labs, len(labs), replace=True)
        yy = np.concatenate([y[g == l] for l in pick])
        gg = np.concatenate([np.full((g == l).sum(), i) for i, l in enumerate(pick)])
        vals.append(icc_oneway(yy, gg)["icc"])
    vals = np.array(vals, float)
    return {"icc_lo": float(np.nanpercentile(vals, 2.5)), "icc_hi": float(np.nanpercentile(vals, 97.5))}


def mixed_model_icc(df: pd.DataFrame, score_col: str = "corrected", lab_col: str = "lab", subject_col: str = "subject",
                    covariates: Sequence[str] = ("log_n_units",)) -> Dict[str, float]:
    """Variance components via statsmodels MixedLM: score ~ covariates + (1|lab) + (1|subject:lab).

    Returns var_lab, var_subject, var_resid, icc_lab, icc_subject and fixed-effect coefficients.
    Falls back to `icc_oneway` when statsmodels is unavailable or the fit fails.
    """
    d = df.dropna(subset=[score_col, lab_col, subject_col]).copy()
    if "log_n_units" in covariates and "log_n_units" not in d and "n_units" in d:
        d["log_n_units"] = np.log(d["n_units"].astype(float).clip(lower=1))
    covs = [c for c in covariates if c in d]
    try:
        import warnings

        import statsmodels.formula.api as smf
        formula = f"{score_col} ~ 1" + "".join(f" + {c}" for c in covs)
        vc = {"subject": f"0 + C({subject_col})"} if d[subject_col].nunique() > d[lab_col].nunique() else None
        model = smf.mixedlm(formula, d, groups=d[lab_col], re_formula="1", vc_formula=vc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # boundary / convergence warnings are expected when var_lab ~ 0
            res = model.fit(reml=True, method=["lbfgs", "powell"], maxiter=500)
        var_lab = float(res.cov_re.iloc[0, 0])
        var_subj = float(res.vcomp[0]) if vc is not None and len(res.vcomp) else 0.0
        var_res = float(res.scale)
        tot = var_lab + var_subj + var_res
        out = {"var_lab": var_lab, "var_subject": var_subj, "var_resid": var_res,
               "icc_lab": var_lab / tot if tot > 0 else np.nan, "icc_subject": var_subj / tot if tot > 0 else np.nan,
               "converged": bool(getattr(res, "converged", True)), "method": "mixedlm"}
        for c in ["Intercept"] + covs:
            out[f"beta_{c}"] = float(res.params.get(c, np.nan))
        return out
    except Exception:  # noqa: BLE001
        o = icc_oneway(d[score_col], d[lab_col])
        return {"var_lab": o["var_between"], "var_subject": np.nan, "var_resid": o["var_within"],
                "icc_lab": o["icc"], "icc_subject": np.nan, "converged": False, "method": "oneway"}


# ----------------------------------------------------------------------------- fragility
def leave_one_lab_out(df: pd.DataFrame, score_col: str = "corrected", var_col: str = "null_var",
                      lab_col: str = "lab", alpha: float = 0.05) -> Dict[str, object]:
    """Re-run the random-effects pooling with each lab removed; flag claims that depend on one lab."""
    full = dersimonian_laird(df[score_col], df[var_col])
    labs = df[lab_col].unique()
    per_lab = {}
    for l in labs:
        sub = df[df[lab_col] != l]
        per_lab[l] = dersimonian_laird(sub[score_col], sub[var_col])["p"] if len(sub) else np.nan
    p_loo = pd.Series(per_lab)
    fragile = bool(full["p"] < alpha and (p_loo.dropna() >= alpha).any())
    single_lab = len(labs) < 2
    return {"p_full": full["p"], "p_loo_max": float(p_loo.max()) if len(p_loo) else np.nan,
            "driving_lab": (p_loo.idxmax() if fragile else None), "fragile": fragile, "single_lab": single_lab,
            "n_labs": int(len(labs))}


def region_table(df: pd.DataFrame, min_sessions: int = 3, min_labs: int = 2, alpha: float = 0.05) -> pd.DataFrame:
    """Per-region summary: pooled corrected score, heterogeneity, lab ICC (+CI), LOLO fragility."""
    rows = []
    d = df.copy()
    if "null_var" not in d and "null_sd" in d:
        d["null_var"] = d["null_sd"].astype(float) ** 2
    for region, sub in d.groupby("region"):
        if len(sub) < min_sessions or sub["lab"].nunique() < min_labs:
            continue
        dl = dersimonian_laird(sub["corrected"], sub["null_var"])
        icc = icc_oneway(sub["corrected"], sub["lab"])
        ci = bootstrap_icc(sub["corrected"], sub["lab"], n_boot=200)
        mm = mixed_model_icc(sub) if len(sub) >= 6 else {"icc_lab": np.nan, "beta_log_n_units": np.nan}
        lolo = leave_one_lab_out(sub)
        rows.append({"region": region, "n_sessions": len(sub), "n_labs": sub["lab"].nunique(),
                     "pooled_effect": dl["effect"], "pooled_se": dl["se"], "p_pooled": dl["p"], "I2": dl["I2"],
                     "tau2": dl["tau2"], "icc_lab": icc["icc"], "icc_lo": ci["icc_lo"], "icc_hi": ci["icc_hi"],
                     "icc_lab_mixed": mm.get("icc_lab", np.nan), "beta_log_n_units": mm.get("beta_log_n_units", np.nan),
                     "fragile_lolo": lolo["fragile"], "driving_lab": lolo["driving_lab"],
                     "mean_n_units": float(sub["n_units"].mean()) if "n_units" in sub else np.nan})
    tab = pd.DataFrame(rows).set_index("region")
    if len(tab):
        tab["fdr_pooled"] = bh_fdr(tab["p_pooled"].to_numpy())
        tab["robust"] = (tab["fdr_pooled"] < alpha) & ~tab["fragile_lolo"] & (tab["icc_lab"].fillna(0) < 0.2)
    return tab.sort_values("p_pooled")


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    q = np.minimum.accumulate((pv[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(q, 0, 1)
    out[ok] = adj
    return out


def yield_matched_subsample(df: pd.DataFrame, n_units_col: str = "n_units", target: Optional[int] = None) -> pd.DataFrame:
    """Keep sessions with >= target units (default: the 25th percentile) so labs are compared at matched yield.

    Real yield matching must be done at the decoding stage (subsample units to `target` before decoding);
    this helper selects the rows and records the target for that re-run.
    """
    t = int(np.percentile(df[n_units_col], 25)) if target is None else target
    out = df[df[n_units_col] >= t].copy()
    out["yield_target"] = t
    return out
