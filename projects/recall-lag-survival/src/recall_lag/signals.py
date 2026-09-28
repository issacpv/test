"""Monthly report counts, first serious events and Poisson-CUSUM signal dates.

The Poisson CUSUM (Lucas, 1985) accumulates ``S_t = max(0, S_{t-1} + x_t - k)`` with reference value
``k = (lambda1 - lambda0) / ln(lambda1 / lambda0)`` for an in-control rate ``lambda0`` and an out-of-control
rate ``lambda1 = multiplier * lambda0``; a signal is raised when ``S_t >= h``.  The threshold ``h`` is chosen
for a target false-alarm rate on never-recalled device keys (:func:`false_alarm_rate`).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def monthly_counts(dates: pd.Series, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.Series:
    """Counts per calendar month (complete monthly PeriodIndex, zeros filled)."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if d.empty:
        return pd.Series(dtype=int)
    p0 = pd.Timestamp(start or d.min()).to_period("M")
    p1 = pd.Timestamp(end or d.max()).to_period("M")
    counts = d.dt.to_period("M").value_counts().sort_index()
    return counts.reindex(pd.period_range(p0, p1, freq="M"), fill_value=0).astype(int)


def counts_by_key(reports: pd.DataFrame, key_col: str = "key", date_col: str = "date_received", serious_col: str = "serious", end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Long table ``(key, month, n_all, n_serious)`` with complete monthly ranges per key from its first report."""
    rows = []
    for k, g in reports.groupby(key_col):
        all_c = monthly_counts(g[date_col], end=end)
        ser_c = monthly_counts(g.loc[g[serious_col].astype(bool), date_col], start=all_c.index[0].to_timestamp() if len(all_c) else None, end=end) if g[serious_col].any() else pd.Series(0, index=all_c.index, dtype=int)
        ser_c = ser_c.reindex(all_c.index, fill_value=0)
        for m in all_c.index:
            rows.append({"key": k, "month": m, "n_all": int(all_c.loc[m]), "n_serious": int(ser_c.loc[m])})
    return pd.DataFrame(rows)


def first_event_dates(reports: pd.DataFrame, key_col: str = "key", date_col: str = "date_received", serious_col: str = "serious") -> pd.DataFrame:
    """Per key: first report date, first serious report date, counts."""
    rows = []
    for k, g in reports.groupby(key_col):
        d = pd.to_datetime(g[date_col], errors="coerce")
        s = d[g[serious_col].astype(bool)]
        rows.append({"key": k, "first_any": d.min(), "first_serious": s.min() if len(s) else pd.NaT, "n_reports": int(d.notna().sum()), "n_serious": int(len(s))})
    return pd.DataFrame(rows)


def poisson_cusum(counts: Sequence[float], lambda0: float, multiplier: float = 2.0, h: float = 4.0) -> Dict[str, object]:
    """Upper Poisson CUSUM on a count series. Returns the statistic path and the first signal index (or None)."""
    x = np.asarray(counts, dtype=float)
    lambda0 = max(float(lambda0), 1e-6)
    lambda1 = multiplier * lambda0
    k = (lambda1 - lambda0) / np.log(lambda1 / lambda0)
    s = np.zeros(x.size)
    first = None
    acc = 0.0
    for i, xi in enumerate(x):
        acc = max(0.0, acc + xi - k)
        s[i] = acc
        if first is None and acc >= h:
            first = i
    return {"statistic": s, "first_signal_index": first, "k": float(k), "h": float(h)}


def signal_date(counts: pd.Series, baseline_months: int = 12, multiplier: float = 2.0, h: float = 4.0, min_baseline_rate: float = 0.1, fallback_rate: Optional[float] = None) -> Optional[pd.Period]:
    """First month at which the CUSUM (baseline from the first ``baseline_months``) signals; None if never.

    Sparse keys (baseline rate below ``min_baseline_rate``) use ``fallback_rate`` (e.g. the product-code
    average) when given, else ``min_baseline_rate``. The baseline months themselves are not eligible to signal.
    """
    if len(counts) <= baseline_months:
        return None
    base = float(np.mean(counts.iloc[:baseline_months]))
    if base < min_baseline_rate:
        base = fallback_rate if fallback_rate is not None else min_baseline_rate
    res = poisson_cusum(counts.iloc[baseline_months:].to_numpy(), base, multiplier=multiplier, h=h)
    idx = res["first_signal_index"]
    return None if idx is None else counts.index[baseline_months + int(idx)]


def false_alarm_rate(count_series: Dict[str, pd.Series], h: float, baseline_months: int = 12, multiplier: float = 2.0) -> Dict[str, float]:
    """Signals per 100 key-years on a set of (never-recalled) monthly series, for threshold ``h``."""
    n_sig, years = 0, 0.0
    for s in count_series.values():
        if len(s) <= baseline_months:
            continue
        years += (len(s) - baseline_months) / 12.0
        if signal_date(s, baseline_months, multiplier, h) is not None:
            n_sig += 1
    return {"signals": n_sig, "key_years": years, "per_100_key_years": 100.0 * n_sig / years if years > 0 else np.nan}


def lead_times(count_series: Dict[str, pd.Series], recall_month: Dict[str, pd.Period], h: float, baseline_months: int = 12, multiplier: float = 2.0) -> pd.DataFrame:
    """For recalled keys: months between the CUSUM signal and recall initiation (positive = signal first).

    Only months strictly before the recall month are used for the CUSUM (no post-recall reporting).
    """
    rows = []
    for k, s in count_series.items():
        if k not in recall_month:
            continue
        rm = recall_month[k]
        pre = s[s.index < rm]
        sd = signal_date(pre, baseline_months, multiplier, h)
        rows.append({"key": k, "recall_month": rm, "signal_month": sd, "lead_months": (rm - sd).n if sd is not None else np.nan, "flagged": sd is not None})
    return pd.DataFrame(rows)
