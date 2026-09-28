"""Population statistics on per-cell threshold tables.

Expected long table columns: ``cell_id, cell_type, dataset, threshold_uA`` plus morphometric
columns (from :func:`rgc_prosthesis.swc_morph.morphometrics`) and design factors such as
``ais_start_um, ais_len_um, electrode, pulse_us``.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .swc_morph import AIS, AXON, HILLOCK, SOMA


def type_threshold_summary(df: pd.DataFrame, value: str = "threshold_uA", group: str = "cell_type") -> pd.DataFrame:
    """Median, IQR, CV (of log values -> geometric CV) and n per type; plus between/within variance ratio in attrs."""
    d = df[np.isfinite(df[value]) & (df[value] > 0)]
    g = d.groupby(group)[value]
    out = pd.DataFrame({"n": g.size(), "median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                        "cv": g.apply(lambda s: float(s.std(ddof=1) / s.mean()) if len(s) > 1 else np.nan),
                        "gsd": g.apply(lambda s: float(np.exp(np.log(s).std(ddof=1))) if len(s) > 1 else np.nan)})
    logv = np.log(d[value])
    within = float(d.assign(_l=logv).groupby(group)["_l"].var(ddof=1).mean())
    between = float(d.assign(_l=logv).groupby(group)["_l"].mean().var(ddof=1)) if d[group].nunique() > 1 else np.nan
    out.attrs["between_within_ratio"] = between / within if within > 0 else np.nan
    return out.reset_index()


def morphological_determinants(df: pd.DataFrame, features: Sequence[str], value: str = "threshold_uA",
                               group: Optional[str] = "cell_type", standardize: bool = True) -> Dict[str, object]:
    """Mixed model ``log(threshold) ~ features + (1 | group)`` (statsmodels MixedLM); OLS if no group.

    Returns coefficient table (``feature, coef, se, p``), marginal R^2 (fixed effects) and the fitted
    result object. Features are z-scored so coefficients are comparable (per SD).
    """
    import statsmodels.api as sm

    d = df[np.isfinite(df[value]) & (df[value] > 0)].dropna(subset=list(features)).copy()
    y = np.log(d[value].values)
    X = d[list(features)].astype(float)
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X = sm.add_constant(X, has_constant="add")
    if group is not None and d[group].nunique() > 1:
        model = sm.MixedLM(y, X, groups=d[group].values)
        res = model.fit(reml=True, method="lbfgs")
        params, bse, pvals = res.fe_params, res.bse_fe, res.pvalues[: len(res.fe_params)]
    else:
        res = sm.OLS(y, X).fit()
        params, bse, pvals = res.params, res.bse, res.pvalues
    yhat = X.values @ np.asarray(params)
    r2_marginal = float(1 - np.var(y - yhat) / np.var(y))
    table = pd.DataFrame({"feature": list(X.columns), "coef": np.asarray(params), "se": np.asarray(bse),
                          "p": np.asarray(pvals)})
    return {"table": table, "r2_marginal": r2_marginal, "result": res, "n": int(len(d))}


def variance_fractions(df: pd.DataFrame, value: str, factors: Sequence[str], log: bool = True) -> Dict[str, float]:
    """Main-effect variance fractions Var(E[y|F]) / Var(y) for each factor (first-order Sobol indices
    on a balanced design); ``residual`` = 1 - sum of main effects and pairwise interactions."""
    d = df[np.isfinite(df[value])].copy()
    y = np.log(d[value]) if log else d[value]
    d["_y"] = y
    tot = float(np.var(y))
    out: Dict[str, float] = {}
    for f in factors:
        out[f] = float(np.var(d.groupby(f)["_y"].transform("mean"))) / tot if tot > 0 else np.nan
    for f1, f2 in itertools.combinations(factors, 2):
        joint = float(np.var(d.groupby([f1, f2])["_y"].transform("mean"))) / tot if tot > 0 else np.nan
        out[f"{f1}x{f2}"] = max(joint - out[f1] - out[f2], 0.0)
    out["residual"] = max(1.0 - sum(out.values()), 0.0)
    return out


def selectivity_auc(df: pd.DataFrame, type_a: str, type_b: str, value: str = "threshold_uA",
                    group: str = "cell_type") -> Dict[str, float]:
    """Separability of two types' threshold distributions.

    ``auc``: P(threshold_a < threshold_b) (Mann-Whitney U / (n_a n_b)); 0.5 = inseparable.
    ``frac_a_at_b10``: fraction of type-a cells activated at the amplitude that activates 10% of type b
    (a practical "selective window").
    """
    a = df.loc[(df[group] == type_a) & np.isfinite(df[value]), value].values
    b = df.loc[(df[group] == type_b) & np.isfinite(df[value]), value].values
    if len(a) == 0 or len(b) == 0:
        return {"auc": np.nan, "frac_a_at_b10": np.nan, "p_mannwhitney": np.nan}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = float(u.statistic / (len(a) * len(b)))
    amp = np.percentile(b, 10)
    return {"auc": auc, "frac_a_at_b10": float(np.mean(a <= amp)), "p_mannwhitney": float(u.pvalue),
            "n_a": int(len(a)), "n_b": int(len(b))}


def pairwise_selectivity(df: pd.DataFrame, value: str = "threshold_uA", group: str = "cell_type") -> pd.DataFrame:
    types = sorted(df[group].dropna().unique())
    rows = []
    for a, b in itertools.combinations(types, 2):
        r = selectivity_auc(df, a, b, value, group)
        r.update({"type_a": a, "type_b": b, "auc_max": max(r["auc"], 1 - r["auc"]) if np.isfinite(r["auc"]) else np.nan})
        rows.append(r)
    return pd.DataFrame(rows)


def compare_with_empirical(model_thr: np.ndarray, empirical_thr: np.ndarray, fit_scale: bool = True) -> Dict[str, float]:
    """KS test between model and empirical thresholds after an optional global scale (ratio of medians),
    and the slope of the QQ line on log values (1 = same spread)."""
    m = np.asarray(model_thr, dtype=float)
    e = np.asarray(empirical_thr, dtype=float)
    m, e = m[np.isfinite(m) & (m > 0)], e[np.isfinite(e) & (e > 0)]
    scale = float(np.median(e) / np.median(m)) if fit_scale else 1.0
    ms = m * scale
    ks = stats.ks_2samp(ms, e)
    qs = np.linspace(0.05, 0.95, 19)
    qq_slope = float(np.polyfit(np.log(np.quantile(ms, qs)), np.log(np.quantile(e, qs)), 1)[0])
    return {"scale": scale, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "qq_slope_log": qq_slope,
            "n_model": int(len(ms)), "n_empirical": int(len(e))}


def activation_site(region_code: Optional[int]) -> str:
    """Human-readable activation site from the region code returned by ``first_spiking_region``."""
    return {AIS: "ais", HILLOCK: "hillock", AXON: "distal_axon", SOMA: "soma"}.get(region_code, "dendrite" if region_code else "none")
