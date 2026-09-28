"""Disproportionality trajectories and landmark features.

Time is indexed by integer quarter (:func:`quarter_index`; 0 = 2004Q1). For a (drug, PT) pair the
inputs are per-quarter counts ``a`` (drug & PT), ``drug_total``, ``pt_total`` and ``N``; from their
cumulative sums we compute a cumulative 2x2 table per quarter and the BCPNN information
component (IC) with the Norén et al. (2013) shrinkage and credibility bounds:

    E   = (a+b)(a+c)/N
    IC  = log2((a + 0.5) / (E + 0.5))
    IC025 ~= IC - 3.3 (a+0.5)^(-1/2) - 2 (a+0.5)^(-3/2)      (Norén 2006 approximation)

Features at landmark L use only quarters <= L. That restriction is the whole point of the
project: it is what makes the resulting model a forecast rather than a description.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE_YEAR = 2004


def quarter_index(q: str | pd.Period) -> int:
    """``'2019Q3'`` -> integer index (0 = 2004Q1)."""
    p = pd.Period(q, freq="Q") if isinstance(q, str) else q
    return (p.year - BASE_YEAR) * 4 + (p.quarter - 1)


def index_to_quarter(i: int) -> str:
    return f"{BASE_YEAR + i // 4}Q{i % 4 + 1}"


# --------------------------------------------------------------------------- disproportionality
def bcpnn_ic(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised IC, IC025, IC975 (Norén et al. 2013 shrinkage; Norén 2006 interval approximation)."""
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    n = a + b + c + d
    e = (a + b) * (a + c) / np.maximum(n, 1.0)
    ic = np.log2((a + 0.5) / (e + 0.5))
    s = np.maximum(a + 0.5, 0.5)
    ic025 = ic - 3.3 * s ** -0.5 - 2.0 * s ** -1.5
    ic975 = ic + 2.4 * s ** -0.5 - 0.5 * s ** -1.5
    return {"ic": ic, "ic025": ic025, "ic975": ic975, "expected": e}


def ror(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, cc: float = 0.5) -> Dict[str, np.ndarray]:
    """Reporting odds ratio with 95% CI (continuity constant ``cc`` added to every cell)."""
    a, b, c, d = (np.asarray(x, dtype=float) + cc for x in (a, b, c, d))
    lr = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"ror": np.exp(lr), "ror_lo": np.exp(lr - 1.96 * se), "ror_hi": np.exp(lr + 1.96 * se), "log_ror": lr, "se": se}


def cumulative_2x2(pair_counts: pd.Series, drug_total: pd.Series, pt_total: pd.Series, n_total: pd.Series,
                   quarters: Iterable[int]) -> pd.DataFrame:
    """Cumulative 2x2 tables per quarter.

    Inputs are per-quarter counts indexed by integer quarter (missing quarters = 0). Returns a
    DataFrame indexed by quarter with ``a, b, c, d, a_q`` (``a_q`` = non-cumulative count that quarter)
    and the IC/ROR columns.
    """
    qs = np.array(sorted(set(quarters)), dtype=int)
    idx = pd.Index(qs)
    a_q = pd.Series(pair_counts).reindex(idx, fill_value=0).astype(float)
    dt = pd.Series(drug_total).reindex(idx, fill_value=0).astype(float)
    pt = pd.Series(pt_total).reindex(idx, fill_value=0).astype(float)
    nn = pd.Series(n_total).reindex(idx, fill_value=0).astype(float)
    a = a_q.cumsum()
    b = dt.cumsum() - a
    c = pt.cumsum() - a
    d = nn.cumsum() - a - b - c
    b, c, d = b.clip(lower=0), c.clip(lower=0), d.clip(lower=0)
    out = pd.DataFrame({"a_q": a_q, "a": a, "b": b, "c": c, "d": d}, index=idx)
    ic = bcpnn_ic(a.values, b.values, c.values, d.values)
    rr = ror(a.values, b.values, c.values, d.values)
    for k in ("ic", "ic025", "ic975"):
        out[k] = ic[k]
    out["log_ror"] = rr["log_ror"]
    out["ror_lo"] = rr["ror_lo"]
    return out


# --------------------------------------------------------------------------- landmark features
def _consecutive_positive(x: np.ndarray) -> int:
    n = 0
    for v in x[::-1]:
        if v > 0:
            n += 1
        else:
            break
    return n


def _slope(y: np.ndarray) -> float:
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def landmark_features(table: pd.DataFrame, landmark: int, approval_quarter: Optional[int] = None,
                      serious_q: Optional[pd.Series] = None, hcp_q: Optional[pd.Series] = None,
                      class_warning: int = 0) -> Dict[str, float]:
    """Features for one pair at landmark ``landmark`` using rows with quarter <= landmark only.

    ``table`` is the output of :func:`cumulative_2x2`. Returns a flat dict; ``n_quarters_obs`` is
    the number of quarters with data up to the landmark (0 -> all features NaN-safe zeros).
    """
    t = table.loc[table.index <= landmark]
    if t.empty:
        return {"ic025": -5.0, "ic": 0.0, "ic_slope4": 0.0, "ic_slope8": 0.0, "consec_pos": 0, "log_a": 0.0,
                "recent_share4": 0.0, "log_ror_lo": 0.0, "drug_age_q": float(landmark - approval_quarter) if approval_quarter is not None else np.nan,
                "serious_share": 0.0, "hcp_share": 0.0, "class_warning": float(class_warning), "n_quarters_obs": 0}
    ic025 = t["ic025"].values
    ic = t["ic"].values
    a_q = t["a_q"].values
    a_cum = float(t["a"].iloc[-1])
    recent4 = float(a_q[-4:].sum())
    feats = {
        "ic025": float(ic025[-1]),
        "ic": float(ic[-1]),
        "ic_slope4": _slope(ic[-4:]),
        "ic_slope8": _slope(ic[-8:]),
        "consec_pos": float(_consecutive_positive(ic025)),
        "log_a": float(np.log1p(a_cum)),
        "recent_share4": recent4 / a_cum if a_cum > 0 else 0.0,
        "log_ror_lo": float(np.log(max(t["ror_lo"].iloc[-1], 1e-6))),
        "drug_age_q": float(landmark - approval_quarter) if approval_quarter is not None else np.nan,
        "class_warning": float(class_warning),
        "n_quarters_obs": float(len(t)),
    }
    for name, ser in (("serious_share", serious_q), ("hcp_share", hcp_q)):
        if ser is None:
            feats[name] = 0.0
        else:
            s = pd.Series(ser).reindex(t.index, fill_value=0).astype(float).sum()
            feats[name] = float(s / a_cum) if a_cum > 0 else 0.0
    return feats


def features_over_landmarks(table: pd.DataFrame, landmarks: Iterable[int], **kw) -> pd.DataFrame:
    rows = []
    for L in landmarks:
        f = landmark_features(table, L, **kw)
        f["landmark"] = L
        rows.append(f)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- notoriety
def notoriety_ratio(pair_q: pd.Series, drug_total_q: pd.Series, event_quarter: int, window: int = 4,
                    eps: float = 0.5) -> Dict[str, float]:
    """Ratio of the pair's reporting share (pair / drug total) in the ``window`` quarters after the
    label event to the ``window`` quarters before it (event quarter excluded).

    A ratio well above 1 indicates stimulated reporting; a model whose features straddle the
    event would learn from it (reverse causation).
    """
    pre_idx = range(event_quarter - window, event_quarter)
    post_idx = range(event_quarter + 1, event_quarter + 1 + window)
    p = pd.Series(pair_q).astype(float)
    dtot = pd.Series(drug_total_q).astype(float)
    pre_p, pre_d = p.reindex(pre_idx, fill_value=0).sum(), dtot.reindex(pre_idx, fill_value=0).sum()
    post_p, post_d = p.reindex(post_idx, fill_value=0).sum(), dtot.reindex(post_idx, fill_value=0).sum()
    pre_share = (pre_p + eps) / (pre_d + eps)
    post_share = (post_p + eps) / (post_d + eps)
    return {"pre_share": float(pre_share), "post_share": float(post_share), "ratio": float(post_share / pre_share),
            "pre_reports": float(pre_p), "post_reports": float(post_p)}
