"""Interrupted time series (ITS) for stimulated reporting in FAERS.

We model monthly report counts ``y_t`` around an FDA Drug Safety Communication
(DSC) or media event with segmented regression:

    log E[y_t] = b0 + b1 * t + b2 * post_t + b3 * (t - T0) * post_t
                 + seasonal terms + log(offset_t)

* ``exp(b2)`` is the immediate level change (rate ratio) at the event,
* ``exp(b3)`` is the change in monthly slope,
* ``offset_t`` (optional) is the *total* number of FAERS reports that month (or
  an exposure denominator), which turns the model into one for the *share* of
  reporting rather than the absolute count and removes secular growth of FAERS.

Counts are over-dispersed, so the default family is negative binomial
(``statsmodels`` ``NegativeBinomial`` with alpha estimated by a Poisson-based
auxiliary regression); Poisson with HAC (Newey-West) covariance is available.

A placebo (permutation) null re-fits the model at random pseudo-event dates to
calibrate the level-change statistic against the natural volatility of the
series (see :func:`placebo_its`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class ITSResult:
    """Container for a fitted segmented regression."""

    params: pd.Series
    conf_int: pd.DataFrame
    pvalues: pd.Series
    level_rr: float
    level_rr_ci: tuple
    slope_rr: float
    slope_rr_ci: tuple
    fitted: pd.Series
    counterfactual: pd.Series
    excess_reports: float
    stimulation_ratio: float
    family: str
    n_pre: int
    n_post: int

    def summary(self) -> pd.DataFrame:
        """One-row DataFrame with the headline quantities."""
        return pd.DataFrame(
            {
                "level_rr": [self.level_rr],
                "level_rr_lo": [self.level_rr_ci[0]],
                "level_rr_hi": [self.level_rr_ci[1]],
                "level_p": [self.pvalues.get("post", np.nan)],
                "slope_rr": [self.slope_rr],
                "slope_rr_lo": [self.slope_rr_ci[0]],
                "slope_rr_hi": [self.slope_rr_ci[1]],
                "slope_p": [self.pvalues.get("post_trend", np.nan)],
                "excess_reports": [self.excess_reports],
                "stimulation_ratio": [self.stimulation_ratio],
                "family": [self.family],
                "n_pre": [self.n_pre],
                "n_post": [self.n_post],
            }
        )


def build_monthly_series(
    monthly_counts: pd.DataFrame,
    event_date: str,
    pre_months: int = 36,
    post_months: int = 24,
    offset_counts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Window a monthly count table around an event and add design columns.

    Parameters
    ----------
    monthly_counts:
        DataFrame with ``month`` (month-start timestamp) and ``count``.
    event_date:
        ISO date of the DSC / media event. The event month is the first
        post month.
    pre_months, post_months:
        Window sizes. Months with no reports are filled with 0.
    offset_counts:
        Optional DataFrame (``month``, ``count``) of *all* FAERS reports per
        month, merged as ``offset`` for share-of-reporting models.
    """
    ev = pd.Timestamp(event_date).to_period("M").to_timestamp()
    start = ev - pd.DateOffset(months=pre_months)
    end = ev + pd.DateOffset(months=post_months - 1)
    idx = pd.date_range(start, end, freq="MS")
    s = monthly_counts.set_index("month")["count"].reindex(idx, fill_value=0)
    df = pd.DataFrame({"month": idx, "count": s.values.astype(float)})
    df["t"] = np.arange(len(df))
    df["post"] = (df["month"] >= ev).astype(int)
    t0 = int(df.loc[df["post"] == 1, "t"].min()) if df["post"].any() else len(df)
    df["post_trend"] = np.where(df["post"] == 1, df["t"] - t0, 0)
    df["month_of_year"] = df["month"].dt.month
    if offset_counts is not None:
        off = offset_counts.set_index("month")["count"].reindex(idx)
        df["offset"] = off.values.astype(float)
    return df


def _design(df: pd.DataFrame, seasonal: bool) -> pd.DataFrame:
    X = df[["t", "post", "post_trend"]].astype(float).copy()
    if seasonal:
        # Fourier pair for annual seasonality (2 df) — cheaper than 11 dummies
        w = 2 * np.pi * df["t"].values / 12.0
        X["sin12"] = np.sin(w)
        X["cos12"] = np.cos(w)
    return sm.add_constant(X, has_constant="add")


def fit_its(
    df: pd.DataFrame,
    family: str = "nb",
    seasonal: bool = True,
    use_offset: bool = False,
    hac_maxlags: Optional[int] = 3,
    post_window_months: Optional[int] = 6,
) -> ITSResult:
    """Fit segmented regression to the output of :func:`build_monthly_series`.

    Parameters
    ----------
    family:
        ``"nb"`` (negative binomial, default), ``"poisson"`` or ``"ols"`` (on
        ``log1p(count)``; useful as a robustness check).
    seasonal:
        Add an annual Fourier pair.
    use_offset:
        Include ``log(offset)`` (requires an ``offset`` column). Rows with a
        zero/NaN offset are dropped.
    hac_maxlags:
        If not ``None``, use Newey-West (HAC) robust covariance with this many
        lags to guard against residual autocorrelation.
    post_window_months:
        Horizon over which excess reports and the stimulation ratio
        (observed / counterfactual) are accumulated. ``None`` = whole post period.
    """
    work = df.copy()
    offset = None
    if use_offset:
        if "offset" not in work.columns:
            raise ValueError("use_offset=True requires an 'offset' column")
        work = work[(work["offset"] > 0) & work["offset"].notna()].reset_index(drop=True)
        offset = np.log(work["offset"].values.astype(float))
    y = work["count"].values.astype(float)
    X = _design(work, seasonal)
    cov_kw: Dict[str, object] = {}
    if hac_maxlags is not None:
        cov_kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": int(hac_maxlags)}}

    if family == "ols":
        model = sm.OLS(np.log1p(y), X)
        res = model.fit(**cov_kw)
        link_exp = True
    elif family == "poisson":
        model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        res = model.fit(**cov_kw)
        link_exp = True
    elif family == "nb":
        # estimate alpha from Poisson fit (Cameron-Trivedi auxiliary regression)
        pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
        mu = pois.fittedvalues
        aux = ((y - mu) ** 2 - y) / np.maximum(mu, 1e-8)
        alpha = float(np.clip(np.nanmean(aux / np.maximum(mu, 1e-8)) if len(mu) else 0.0, 1e-4, 10.0))
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset)
        res = model.fit(**cov_kw)
        link_exp = True
    else:
        raise ValueError(f"unknown family {family!r}")

    params = res.params
    ci = res.conf_int()
    ci.columns = ["lo", "hi"]
    level_rr = float(np.exp(params["post"])) if link_exp else float(params["post"])
    level_ci = (float(np.exp(ci.loc["post", "lo"])), float(np.exp(ci.loc["post", "hi"])))
    slope_rr = float(np.exp(params["post_trend"]))
    slope_ci = (float(np.exp(ci.loc["post_trend", "lo"])), float(np.exp(ci.loc["post_trend", "hi"])))

    # counterfactual: same design with post terms zeroed
    Xcf = X.copy()
    Xcf["post"] = 0.0
    Xcf["post_trend"] = 0.0
    if family == "ols":
        fitted = pd.Series(np.expm1(res.predict(X)), index=work.index)
        cf = pd.Series(np.expm1(res.predict(Xcf)), index=work.index)
    else:
        fitted = pd.Series(res.predict(X, offset=offset), index=work.index)
        cf = pd.Series(res.predict(Xcf, offset=offset), index=work.index)

    post_mask = work["post"].values == 1
    if post_window_months is not None:
        first_post = int(np.argmax(post_mask)) if post_mask.any() else len(work)
        window = np.zeros(len(work), dtype=bool)
        window[first_post : first_post + int(post_window_months)] = True
        post_mask = post_mask & window
    excess = float(np.sum(y[post_mask] - cf.values[post_mask]))
    denom = float(np.sum(cf.values[post_mask]))
    stim_ratio = float(np.sum(y[post_mask]) / denom) if denom > 0 else np.nan

    return ITSResult(
        params=params,
        conf_int=ci,
        pvalues=res.pvalues,
        level_rr=level_rr,
        level_rr_ci=level_ci,
        slope_rr=slope_rr,
        slope_rr_ci=slope_ci,
        fitted=fitted,
        counterfactual=cf,
        excess_reports=excess,
        stimulation_ratio=stim_ratio,
        family=family,
        n_pre=int((work["post"] == 0).sum()),
        n_post=int((work["post"] == 1).sum()),
    )


def placebo_its(
    monthly_counts: pd.DataFrame,
    event_date: str,
    n_placebo: int = 200,
    pre_months: int = 36,
    post_months: int = 24,
    min_gap_months: int = 12,
    seed: int = 0,
    **fit_kw,
) -> pd.DataFrame:
    """Permutation null for the level-change coefficient.

    Re-fits the ITS at ``n_placebo`` pseudo-event months drawn uniformly from
    the available range (excluding ``min_gap_months`` either side of the true
    event) and returns their ``log level_rr`` so the observed value can be
    compared to the empirical distribution (``p_perm`` = share of placebos with
    |log RR| >= observed).
    """
    rng = np.random.default_rng(seed)
    months = pd.to_datetime(monthly_counts["month"]).sort_values().reset_index(drop=True)
    ev = pd.Timestamp(event_date).to_period("M").to_timestamp()
    candidates = [
        m
        for m in months
        if (m - months.iloc[0]).days / 30.44 >= pre_months
        and (months.iloc[-1] - m).days / 30.44 >= post_months
        and abs((m - ev).days) / 30.44 >= min_gap_months
    ]
    if not candidates:
        return pd.DataFrame(columns=["placebo_month", "log_level_rr", "level_p"])
    rows = []
    for _ in range(int(n_placebo)):
        m = candidates[int(rng.integers(len(candidates)))]
        d = build_monthly_series(monthly_counts, str(m.date()), pre_months, post_months)
        try:
            r = fit_its(d, **fit_kw)
            rows.append({"placebo_month": m, "log_level_rr": np.log(r.level_rr), "level_p": r.pvalues.get("post", np.nan)})
        except Exception:  # singular fits on sparse series
            continue
    return pd.DataFrame(rows)


def permutation_pvalue(observed_log_rr: float, placebo: pd.DataFrame) -> float:
    """Two-sided empirical p-value of ``observed_log_rr`` against placebo fits."""
    if placebo.empty:
        return float("nan")
    vals = placebo["log_level_rr"].abs().values
    return float((np.sum(vals >= abs(observed_log_rr)) + 1) / (len(vals) + 1))


def batch_its(
    series: Dict[str, pd.DataFrame],
    events: Iterable[Dict[str, str]],
    **kw,
) -> pd.DataFrame:
    """Fit one ITS per (drug, event) pair.

    ``series`` maps a drug key to its monthly count table; each element of
    ``events`` is ``{"drug": key, "event_date": "YYYY-MM-DD", "label": ...}``.
    """
    rows = []
    for ev in events:
        s = series.get(ev["drug"])
        if s is None or s.empty:
            continue
        d = build_monthly_series(s, ev["event_date"], kw.pop("pre_months", 36), kw.pop("post_months", 24))
        try:
            res = fit_its(d, **kw)
        except Exception as exc:
            rows.append({"drug": ev["drug"], "event_date": ev["event_date"], "error": str(exc)})
            continue
        out = res.summary().iloc[0].to_dict()
        out.update({"drug": ev["drug"], "event_date": ev["event_date"], "label": ev.get("label", "")})
        rows.append(out)
    return pd.DataFrame(rows)
