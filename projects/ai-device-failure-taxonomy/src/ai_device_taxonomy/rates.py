"""Denominators and rate models: device-years, rate ratios, negative-binomial regression, pre/post update counts."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def device_years(decision_date: pd.Series, data_end: pd.Timestamp, reporting_start: Optional[pd.Timestamp] = None) -> pd.Series:
    """Years at risk per device from its authorisation date (or ``reporting_start`` if later) to ``data_end``."""
    start = pd.to_datetime(decision_date)
    if reporting_start is not None:
        start = start.where(start >= reporting_start, reporting_start)
    yrs = (pd.Timestamp(data_end) - start).dt.days / 365.25
    return yrs.clip(lower=0.0)


def rate_ratio(events_a: int, years_a: float, events_b: int, years_b: float, alpha: float = 0.05) -> Dict[str, float]:
    """Incidence rate ratio A vs B with a log-normal CI and a conditional exact (binomial) p-value.

    Conditional on the total count, ``events_a ~ Binomial(n, years_a / (years_a + years_b))`` under IRR = 1.
    A 0.5 continuity correction is applied to the point estimate when a cell is zero.
    """
    a, b = float(events_a), float(events_b)
    ca, cb = (a + 0.5, b + 0.5) if (a == 0 or b == 0) else (a, b)
    irr = (ca / years_a) / (cb / years_b)
    se = np.sqrt(1.0 / ca + 1.0 / cb)
    z = stats.norm.ppf(1 - alpha / 2)
    n = int(a + b)
    p_exp = years_a / (years_a + years_b)
    p_val = float(stats.binomtest(int(a), n, p_exp).pvalue) if n > 0 else np.nan
    return {"irr": float(irr), "ci_lo": float(np.exp(np.log(irr) - z * se)), "ci_hi": float(np.exp(np.log(irr) + z * se)), "p_exact": p_val, "rate_a": a / years_a, "rate_b": b / years_b}


def negbin_rate_model(df: pd.DataFrame, count_col: str, exposure_col: str, covariate_cols: Sequence[str], cluster_col: Optional[str] = None) -> pd.DataFrame:
    """Negative-binomial GLM of counts with ``offset = log(exposure)``; returns IRRs with 95% CIs.

    ``alpha`` (dispersion) is estimated by a Poisson fit followed by the auxiliary OLS regression of
    Cameron & Trivedi; categorical covariates should be passed pre-dummified (``pd.get_dummies``).
    """
    d = df[(df[exposure_col] > 0)].copy()
    X = sm.add_constant(d[list(covariate_cols)].astype(float))
    off = np.log(d[exposure_col].astype(float))
    y = d[count_col].astype(float)
    pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
    mu = pois.mu
    aux = ((y - mu) ** 2 - y) / mu
    alpha = float(max(sm.OLS(aux, mu).fit().params.iloc[0], 1e-6))
    fam = sm.families.NegativeBinomial(alpha=alpha)
    if cluster_col:
        fit = sm.GLM(y, X, family=fam, offset=off).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col].astype("category").cat.codes.to_numpy()})
    else:
        fit = sm.GLM(y, X, family=fam, offset=off).fit()
    ci = fit.conf_int()
    out = pd.DataFrame({"irr": np.exp(fit.params), "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1]), "p": fit.pvalues})
    out.attrs["alpha"] = alpha
    out.attrs["n"] = int(len(d))
    return out


def pre_post_update_counts(event_dates: pd.Series, update_dates: Sequence[pd.Timestamp], window_days: int = 180, exclude_days: int = 0) -> pd.DataFrame:
    """Report counts in ``[update - window, update - exclude)`` vs ``(update + exclude, update + window]`` per update.

    A self-controlled design: the device is its own comparator. ``exclude_days`` drops a washout around the
    update date. Returns one row per update with pre/post counts and the rate ratio from :func:`rate_ratio`.
    """
    ev = pd.to_datetime(pd.Series(event_dates)).dropna().sort_values().to_numpy()
    rows = []
    for u in pd.to_datetime(pd.Series(list(update_dates))).dropna():
        pre_lo, pre_hi = u - pd.Timedelta(days=window_days), u - pd.Timedelta(days=exclude_days)
        post_lo, post_hi = u + pd.Timedelta(days=exclude_days), u + pd.Timedelta(days=window_days)
        pre = int(((ev >= np.datetime64(pre_lo)) & (ev < np.datetime64(pre_hi))).sum())
        post = int(((ev > np.datetime64(post_lo)) & (ev <= np.datetime64(post_hi))).sum())
        yrs = (window_days - exclude_days) / 365.25
        rr = rate_ratio(post, yrs, pre, yrs)
        rows.append({"update_date": u, "pre": pre, "post": post, "irr_post_vs_pre": rr["irr"], "ci_lo": rr["ci_lo"], "ci_hi": rr["ci_hi"], "p_exact": rr["p_exact"]})
    return pd.DataFrame(rows)


def monthly_counts(dates: pd.Series, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.Series:
    """Monthly report counts (complete month index, zeros filled)."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if d.empty:
        return pd.Series(dtype=int)
    start = pd.Timestamp(start or d.min()).to_period("M")
    end = pd.Timestamp(end or d.max()).to_period("M")
    counts = d.dt.to_period("M").value_counts().sort_index()
    idx = pd.period_range(start, end, freq="M")
    return counts.reindex(idx, fill_value=0).astype(int)


def category_share_table(labels: pd.DataFrame, categories: Sequence[str], group: Optional[pd.Series] = None) -> pd.DataFrame:
    """Share of reports with each category (overall or by ``group``), with Wilson 95% CIs."""
    cols = [f"tax_{c}" for c in categories]
    rows = []
    groups = [("all", labels)] if group is None else [(g, labels[group.to_numpy() == g]) for g in pd.unique(group)]
    for g, sub in groups:
        n = len(sub)
        for c, col in zip(categories, cols):
            k = int(sub[col].sum()) if n else 0
            p = k / n if n else np.nan
            if n:
                z = 1.96
                denom = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denom
                half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
                lo, hi = centre - half, centre + half
            else:
                lo = hi = np.nan
            rows.append({"group": g, "category": c, "n": n, "k": k, "share": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
