"""Concordance of real-world dose slopes with external knowledge.

Two comparators:
- CredibleMeds ordinal torsades-risk categories (known > possible > conditional
  > no known risk), compared to the estimated dose slopes by rank correlation.
- FAERS torsades-de-pointes reporting, summarised as a reporting odds ratio
  (ROR) with a 2x2 disproportionality table and 0.5 continuity correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CREDIBLEMEDS_ORDER = {
    "known": 3,
    "possible": 2,
    "conditional": 1,
    "no known risk": 0,
    "none": 0,
}


def crediblemeds_rank(category: str) -> float:
    """Map a CredibleMeds category string to an ordinal risk score (0-3)."""
    return CREDIBLEMEDS_ORDER.get(str(category).strip().lower(), np.nan)


def slope_vs_crediblemeds(slopes: pd.DataFrame, cm: pd.DataFrame) -> dict:
    """Spearman correlation between dose slope and CredibleMeds ordinal risk.

    ``slopes``: columns ingredient, slope. ``cm``: columns drug, risk_category.
    """
    cm = cm.copy()
    cm["rank"] = cm["risk_category"].map(crediblemeds_rank)
    merged = slopes.merge(cm, left_on="ingredient", right_on="drug", how="inner")
    merged = merged.dropna(subset=["slope", "rank"])
    if len(merged) < 3:
        return {"rho": np.nan, "p": np.nan, "n": len(merged)}
    rho, p = stats.spearmanr(merged["slope"], merged["rank"])
    return {"rho": float(rho), "p": float(p), "n": int(len(merged))}


def reporting_odds_ratio(a: int, b: int, c: int, d: int, cc: float = 0.5) -> dict:
    """ROR for a 2x2 disproportionality table with continuity correction.

    a = reports of drug with event (TdP); b = reports of drug without event;
    c = reports of other drugs with event; d = reports of other drugs without.
    """
    a2, b2, c2, d2 = a + cc, b + cc, c + cc, d + cc
    ror = (a2 / b2) / (c2 / d2)
    se_ln = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    lo = float(np.exp(np.log(ror) - 1.96 * se_ln))
    hi = float(np.exp(np.log(ror) + 1.96 * se_ln))
    return {"ror": float(ror), "ci_low": lo, "ci_high": hi}


def faers_ror_table(counts: dict[str, dict]) -> pd.DataFrame:
    """Compute per-drug TdP ROR from openFDA counts.

    ``counts`` maps drug -> {"tdp": int, "total": int}. The "rest of database"
    background is the sum over the other drugs supplied (a conservative,
    self-contained approximation; a full study uses the whole FAERS denominator).
    """
    drugs = [k for k, v in counts.items() if v.get("total")]
    total_all = sum(counts[k]["total"] for k in drugs)
    tdp_all = sum(counts[k]["tdp"] or 0 for k in drugs)
    rows = []
    for drug in drugs:
        a = counts[drug]["tdp"] or 0
        n = counts[drug]["total"] or 0
        b = n - a
        c = tdp_all - a
        d = (total_all - n) - c
        rows.append({"ingredient": drug, **reporting_odds_ratio(a, b, c, d)})
    return pd.DataFrame(rows)


def discordance(slopes: pd.DataFrame, faers: pd.DataFrame) -> pd.DataFrame:
    """Flag drugs where the real-world slope and FAERS ROR disagree (rank z-score gap)."""
    m = slopes.merge(faers, on="ingredient", how="inner").dropna(subset=["slope", "ror"])
    if m.empty:
        return m.assign(discordance=np.nan)

    def _z(v: pd.Series) -> pd.Series:
        s = v.std(ddof=0)
        return (v - v.mean()) / s if s > 0 else v * 0.0

    m = m.copy()
    m["z_slope"] = _z(m["slope"])
    m["z_ror"] = _z(np.log(m["ror"]))
    m["discordance"] = (m["z_slope"] - m["z_ror"]).abs()
    return m.sort_values("discordance", ascending=False)
