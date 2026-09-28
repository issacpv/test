"""FAERS disproportionality with a quarterly time scan and unlabeled-signal flags.

Input is a long report table with one row per (report, drug, event):

    report_id | quarter (pandas Period[Q] or 'YYYYQn') | drug (normalised) | pt

From it we build per-quarter *cumulative* 2x2 tables for every drug-event pair
and compute ROR, PRR and the shrunk information component (IC). The **first
signal date** of a pair is the first quarter at which the chosen criterion
(default: IC025 > 0 and n >= 3) holds and keeps holding for ``k_sustain``
consecutive quarters. Signals whose PT is not on the drug's label (after
synonym mapping) are flagged *unlabeled* and ranked by strength and recency.

Count-based inputs are also supported (``cumulative_from_counts``) so the scan
can be run from openFDA ``count`` queries without bulk data (top-1000 PTs per
drug-quarter).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------- measures
def disproportionality(a, b, c, d) -> pd.DataFrame:
    """Vectorised ROR / PRR / IC with 95% intervals for arrays of 2x2 cells."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    zero = (np.minimum.reduce([a, b, c, d]) == 0)
    a_, b_, c_, d_ = (x + 0.5 * zero for x in (a, b, c, d))
    with np.errstate(divide="ignore", invalid="ignore"):
        ror = (a_ * d_) / (b_ * c_)
        se_ror = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
        prr = (a_ / (a_ + b_)) / (c_ / (c_ + d_))
        se_prr = np.sqrt(1 / a_ - 1 / (a_ + b_) + 1 / c_ - 1 / (c_ + d_))
        n = a + b + c + d
        expected = (a + b) * (a + c) / np.where(n > 0, n, np.nan)
        ic = np.log2((a + 0.5) / (expected + 0.5))
        se_ic = 1 / np.sqrt(a + 0.5)
        chi2 = n * (np.abs(a * d - b * c) - n / 2) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "ror": ror,
            "ror025": np.exp(np.log(ror) - 1.96 * se_ror),
            "ror975": np.exp(np.log(ror) + 1.96 * se_ror),
            "prr": prr,
            "prr025": np.exp(np.log(prr) - 1.96 * se_prr),
            "chi2": chi2,
            "ic": ic,
            "ic025": ic - 1.96 * se_ic,
            "ic975": ic + 1.96 * se_ic,
            "expected": expected,
        }
    )


# -------------------------------------------------------- cumulative tables
def _to_quarter(q) -> pd.PeriodIndex:
    if isinstance(q, pd.PeriodIndex):
        return q.asfreq("Q")
    s = pd.Series(q)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.PeriodIndex(s.dt.to_period("Q"))
    if isinstance(s.dtype, pd.PeriodDtype):
        return pd.PeriodIndex(s).asfreq("Q")
    return pd.PeriodIndex(s.astype(str), freq="Q")


def cumulative_pair_tables(
    reports: pd.DataFrame,
    pairs: Optional[Iterable[Tuple[str, str]]] = None,
    min_total_n: int = 3,
    report_col: str = "report_id",
    quarter_col: str = "quarter",
    drug_col: str = "drug",
    pt_col: str = "pt",
) -> pd.DataFrame:
    """Per-quarter cumulative 2x2 cells for drug-event pairs.

    Margins are computed on *distinct reports* per quarter, so a report listing
    several drugs or events counts once in each margin. Returns a long frame
    with ``drug, pt, quarter, a, b, c, d`` for every quarter from the first to
    the last in ``reports`` (pairs are filtered to cumulative n >= ``min_total_n``).
    """
    r = reports[[report_col, quarter_col, drug_col, pt_col]].dropna().copy()
    r["q"] = _to_quarter(r[quarter_col].values)
    quarters = pd.period_range(r["q"].min(), r["q"].max(), freq="Q")
    # per-quarter margins
    n_total = r.drop_duplicates([report_col, "q"]).groupby("q")[report_col].nunique().reindex(quarters, fill_value=0).cumsum()
    n_drug = r.drop_duplicates([report_col, drug_col, "q"]).groupby([drug_col, "q"])[report_col].nunique()
    n_pt = r.drop_duplicates([report_col, pt_col, "q"]).groupby([pt_col, "q"])[report_col].nunique()
    n_pair = r.drop_duplicates([report_col, drug_col, pt_col, "q"]).groupby([drug_col, pt_col, "q"])[report_col].nunique()
    if pairs is None:
        totals = n_pair.groupby(level=[0, 1]).sum()
        pairs = [p for p, v in totals.items() if v >= min_total_n]
    rows = []
    for drug, pt in pairs:
        pa = n_pair.get((drug, pt), pd.Series(dtype=float))
        pa = pa.droplevel([0, 1]) if isinstance(pa.index, pd.MultiIndex) else pa
        a = pa.reindex(quarters, fill_value=0).cumsum()
        nd = n_drug.xs(drug, level=0).reindex(quarters, fill_value=0).cumsum() if drug in n_drug.index.get_level_values(0) else pd.Series(0, index=quarters)
        ne = n_pt.xs(pt, level=0).reindex(quarters, fill_value=0).cumsum() if pt in n_pt.index.get_level_values(0) else pd.Series(0, index=quarters)
        df = pd.DataFrame({"quarter": quarters, "a": a.values, "b": (nd - a).values, "c": (ne - a).values})
        df["d"] = n_total.values - df["a"] - df["b"] - df["c"]
        df["drug"] = drug
        df["pt"] = pt
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["drug", "pt", "quarter", "a", "b", "c", "d"])
    return pd.concat(rows, ignore_index=True)[["drug", "pt", "quarter", "a", "b", "c", "d"]]


def cumulative_from_counts(
    pair_counts: pd.DataFrame,
    drug_totals: pd.DataFrame,
    pt_totals: pd.DataFrame,
    all_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build cumulative tables from openFDA ``count`` outputs per quarter.

    ``pair_counts``: drug, pt, quarter, n ; ``drug_totals``: drug, quarter, n ;
    ``pt_totals``: pt, quarter, n ; ``all_totals``: quarter, n.
    """
    quarters = pd.period_range(min(_to_quarter(all_totals["quarter"])), max(_to_quarter(all_totals["quarter"])), freq="Q")

    def cum(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        d = df.copy()
        d["quarter"] = _to_quarter(d["quarter"].values)
        if keys:
            return d.groupby(keys).apply(lambda g: g.set_index("quarter")["n"].reindex(quarters, fill_value=0).cumsum()).stack().rename("n").reset_index()
        return d.set_index("quarter")["n"].reindex(quarters, fill_value=0).cumsum().rename("n").reset_index()

    pc = cum(pair_counts, ["drug", "pt"]).rename(columns={"n": "a"})
    dt = cum(drug_totals, ["drug"]).rename(columns={"n": "n_drug"})
    et = cum(pt_totals, ["pt"]).rename(columns={"n": "n_pt"})
    at = cum(all_totals, []).rename(columns={"n": "n_all"})
    pc.columns = ["drug", "pt", "quarter", "a"]
    dt.columns = ["drug", "quarter", "n_drug"]
    et.columns = ["pt", "quarter", "n_pt"]
    at.columns = ["quarter", "n_all"]
    out = pc.merge(dt, on=["drug", "quarter"]).merge(et, on=["pt", "quarter"]).merge(at, on="quarter")
    out["b"] = out["n_drug"] - out["a"]
    out["c"] = out["n_pt"] - out["a"]
    out["d"] = out["n_all"] - out["a"] - out["b"] - out["c"]
    return out[["drug", "pt", "quarter", "a", "b", "c", "d"]]


# ---------------------------------------------------------------- scan
def time_scan(
    cum_tables: pd.DataFrame,
    criterion: str = "ic",
    min_n: int = 3,
    k_sustain: int = 2,
) -> pd.DataFrame:
    """Date the first sustained signal for each drug-event pair.

    ``criterion``: ``"ic"`` (IC025 > 0), ``"ror"`` (ROR025 > 1) or ``"prr"``
    (PRR >= 2 & chi2 >= 4, Evans). A signal must hold for ``k_sustain``
    consecutive quarters (counting the first) to be dated.
    Returns one row per pair with ``first_signal_quarter`` (NaT if never),
    the statistic at signal time and the latest quarter's values.
    """
    t = cum_tables.copy()
    t["quarter"] = _to_quarter(t["quarter"].values)
    est = disproportionality(t["a"], t["b"], t["c"], t["d"])
    t = pd.concat([t.reset_index(drop=True), est.drop(columns=["a", "b", "c", "d"])], axis=1)
    if criterion == "ic":
        t["flag"] = (t["ic025"] > 0) & (t["a"] >= min_n)
    elif criterion == "ror":
        t["flag"] = (t["ror025"] > 1) & (t["a"] >= min_n)
    elif criterion == "prr":
        t["flag"] = (t["prr"] >= 2) & (t["chi2"] >= 4) & (t["a"] >= min_n)
    else:
        raise ValueError(criterion)
    rows = []
    for (drug, pt), g in t.sort_values("quarter").groupby(["drug", "pt"], sort=False):
        flags = g["flag"].values
        first_idx = None
        run = 0
        for i, f in enumerate(flags):
            run = run + 1 if f else 0
            if run >= k_sustain:
                first_idx = i - k_sustain + 1
                break
        last = g.iloc[-1]
        row = {
            "drug": drug,
            "pt": pt,
            "first_signal_quarter": g.iloc[first_idx]["quarter"] if first_idx is not None else pd.NaT,
            "n_at_signal": g.iloc[first_idx]["a"] if first_idx is not None else np.nan,
            "ic025_at_signal": g.iloc[first_idx]["ic025"] if first_idx is not None else np.nan,
            "n_current": last["a"],
            "ic_current": last["ic"],
            "ic025_current": last["ic025"],
            "ror_current": last["ror"],
            "ror025_current": last["ror025"],
            "signal_current": bool(last["flag"]),
            "last_quarter": last["quarter"],
        }
        # recency: new reports in the last 4 quarters
        if len(g) >= 5:
            row["n_last_4q"] = float(last["a"] - g.iloc[-5]["a"])
        else:
            row["n_last_4q"] = float(last["a"])
        rows.append(row)
    return pd.DataFrame(rows)


def scan_all_pairs(reports: pd.DataFrame, min_total_n: int = 3, **scan_kw) -> pd.DataFrame:
    """Convenience: cumulative tables for all pairs, then :func:`time_scan`."""
    tables = cumulative_pair_tables(reports, min_total_n=min_total_n)
    return time_scan(tables, **scan_kw)


# ----------------------------------------------------------- unlabeled
def flag_unlabeled(
    signals: pd.DataFrame,
    labeled: Dict[str, Set[str]],
    dictionary=None,
    pt_col: str = "pt",
    drug_col: str = "drug",
) -> pd.DataFrame:
    """Add ``on_label`` and ``label_match`` ('exact' | 'synonym' | 'none' | 'no_label').

    ``labeled`` maps drug -> set of PTs extracted from its label(s). If a
    ``MedDRADictionary`` is given, FAERS PTs are also mapped through
    ``match_string`` (handles UK/US spelling and LLT-level FAERS terms).
    """
    out = signals.copy()
    matches: List[str] = []
    for drug, pt in zip(out[drug_col], out[pt_col]):
        lab = labeled.get(drug)
        if lab is None:
            matches.append("no_label")
            continue
        if pt in lab:
            matches.append("exact")
            continue
        if dictionary is not None:
            mapped = dictionary.match_string(pt)
            if mapped is not None and mapped in lab:
                matches.append("synonym")
                continue
        matches.append("none")
    out["label_match"] = matches
    out["on_label"] = out["label_match"].isin(["exact", "synonym"])
    return out


def rank_unlabeled(flagged: pd.DataFrame, recency_weight: float = 0.5, serious_col: Optional[str] = None) -> pd.DataFrame:
    """Score = IC025 (strength) + w * log1p(recent reports) [+ 0.5 if serious]."""
    f = flagged[(~flagged["on_label"]) & (flagged["label_match"] != "no_label") & flagged["signal_current"]].copy()
    f["score"] = f["ic025_current"].clip(lower=0) + recency_weight * np.log1p(f["n_last_4q"].fillna(0))
    if serious_col and serious_col in f:
        f["score"] += 0.5 * f[serious_col].fillna(0)
    return f.sort_values("score", ascending=False).reset_index(drop=True)


def benjamini_hochberg(p: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out
