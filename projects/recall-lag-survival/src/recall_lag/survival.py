"""Survival estimators: Kaplan-Meier with Greenwood CIs and left truncation, log-rank, Cox PH, RMST, ITS."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


def kaplan_meier(time: Sequence[float], event: Sequence[int], entry: Optional[Sequence[float]] = None, alpha: float = 0.05) -> pd.DataFrame:
    """Kaplan-Meier survival curve at each distinct event time.

    With ``entry`` (left truncation), a subject is at risk at ``t`` only if ``entry < t <= time``.
    Columns: ``time, at_risk, events, survival, se, ci_lo, ci_hi`` (log-log CI).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    en = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=float)
    times = np.unique(t[e == 1])
    surv, var_sum, rows = 1.0, 0.0, []
    z = stats.norm.ppf(1 - alpha / 2)
    for tt in times:
        at_risk = int(np.sum((en < tt) & (t >= tt)))
        d = int(np.sum((t == tt) & (e == 1)))
        if at_risk == 0:
            continue
        surv *= 1.0 - d / at_risk
        if at_risk > d:
            var_sum += d / (at_risk * (at_risk - d))
        se = surv * np.sqrt(var_sum)
        if 0 < surv < 1:
            ll = np.log(-np.log(surv))
            se_ll = np.sqrt(var_sum) / abs(np.log(surv))
            lo, hi = np.exp(-np.exp(ll + z * se_ll)), np.exp(-np.exp(ll - z * se_ll))
        else:
            lo = hi = surv
        rows.append({"time": tt, "at_risk": at_risk, "events": d, "survival": surv, "se": se, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows, columns=["time", "at_risk", "events", "survival", "se", "ci_lo", "ci_hi"])


def median_survival(km: pd.DataFrame) -> float:
    """First time at which the KM curve drops to <= 0.5 (NaN if never)."""
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("nan")


def restricted_mean(km: pd.DataFrame, tau: float) -> float:
    """Restricted mean survival time up to ``tau`` (area under the step function)."""
    times = np.concatenate([[0.0], km["time"].to_numpy(), [tau]])
    surv = np.concatenate([[1.0], km["survival"].to_numpy(), [km["survival"].iloc[-1] if len(km) else 1.0]])
    times = np.clip(times, 0, tau)
    area = 0.0
    for i in range(1, len(times)):
        area += surv[i - 1] * (times[i] - times[i - 1])
    return float(area)


def logrank_test(time: Sequence[float], event: Sequence[int], group: Sequence, entry: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """k-sample log-rank test (with optional left truncation). Returns chi2, df and p."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    g = np.asarray(group)
    en = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=float)
    groups = np.unique(g)
    k = len(groups)
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for tt in np.unique(t[e == 1]):
        at = (en < tt) & (t >= tt)
        n = at.sum()
        d = int(np.sum((t == tt) & (e == 1)))
        if n == 0:
            continue
        nj = np.array([np.sum(at & (g == gg)) for gg in groups], dtype=float)
        dj = np.array([np.sum((t == tt) & (e == 1) & (g == gg)) for gg in groups], dtype=float)
        O += dj
        E += d * nj / n
        if n > 1:
            for a in range(k):
                for b in range(k):
                    V[a, b] += d * (nj[a] / n) * ((1.0 if a == b else 0.0) - nj[b] / n) * (n - d) / (n - 1)
    diff = (O - E)[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2 = float(diff @ np.linalg.solve(Vr, diff))
    except np.linalg.LinAlgError:
        chi2 = float("nan")
    df = k - 1
    return {"chi2": chi2, "df": df, "p": float(1 - stats.chi2.cdf(chi2, df)) if np.isfinite(chi2) else float("nan"), "observed": O.tolist(), "expected": E.tolist()}


def cox_ph(df: pd.DataFrame, time_col: str, event_col: str, covariate_cols: Sequence[str], entry_col: Optional[str] = None, cluster_col: Optional[str] = None) -> pd.DataFrame:
    """Cox proportional-hazards model via ``statsmodels`` PHReg (Efron ties), optional left truncation and
    cluster-robust standard errors. Returns hazard ratios with 95% CIs."""
    d = df.dropna(subset=[time_col, event_col, *covariate_cols]).copy()
    X = d[list(covariate_cols)].astype(float)
    entry = None if entry_col is None else d[entry_col].astype(float).to_numpy()
    # PHReg computes a grouped (sandwich) robust covariance itself when ``groups`` is given at construction
    groups = None if cluster_col is None else d[cluster_col].astype("category").cat.codes.to_numpy()
    model = PHReg(d[time_col].astype(float), X, status=d[event_col].astype(int), entry=entry, ties="efron", groups=groups)
    fit = model.fit()
    ci = fit.conf_int()
    out = pd.DataFrame({"hr": np.exp(fit.params), "ci_lo": np.exp(ci[:, 0]), "ci_hi": np.exp(ci[:, 1]), "p": fit.pvalues}, index=list(covariate_cols))
    out.attrs["n"] = int(len(d))
    out.attrs["events"] = int(d[event_col].sum())
    return out


def interrupted_time_series(y: Sequence[float], t: Sequence[float], break_at: float, maxlags: int = 4) -> pd.DataFrame:
    """Segmented regression ``y ~ t + post + post*(t - break)`` with Newey-West (HAC) standard errors."""
    yy = np.asarray(y, dtype=float)
    tt = np.asarray(t, dtype=float)
    post = (tt >= break_at).astype(float)
    X = sm.add_constant(np.column_stack([tt, post, post * (tt - break_at)]))
    fit = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    names = ["const", "trend", "level_change", "slope_change"]
    ci = fit.conf_int()
    return pd.DataFrame({"coef": fit.params, "ci_lo": ci[:, 0], "ci_hi": ci[:, 1], "p": fit.pvalues}, index=names)
