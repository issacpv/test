"""Specification grids, variance decomposition and specification curves for the sorter multiverse."""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def specification_grid(**factors: Sequence[object]) -> pd.DataFrame:
    """Cartesian product of factor levels, one row per specification."""
    names = list(factors)
    rows = [dict(zip(names, combo)) for combo in itertools.product(*[list(v) for v in factors.values()])]
    df = pd.DataFrame(rows)
    df.insert(0, "spec_id", np.arange(len(df)))
    return df


def variance_decomposition(df: pd.DataFrame, outcome: str, factors: Sequence[str],
                           interactions: bool = False) -> pd.DataFrame:
    """Type-II ANOVA eta-squared (SS_factor / SS_total) of ``outcome`` across categorical ``factors``.

    Use e.g. factors=('sorter', 'qc_rule', 'mouse') to compare sorter-attributable variance with
    between-animal variance. Returns a table with ss, df, eta2 and p per term plus the residual."""
    terms = [f"C({f})" for f in factors]
    if interactions and len(factors) > 1:
        terms += [f"C({a}):C({b})" for a, b in itertools.combinations(factors, 2)]
    formula = f"{outcome} ~ " + " + ".join(terms)
    d = df[[outcome, *factors]].dropna()
    model = smf.ols(formula, data=d).fit()
    tab = anova_lm(model, typ=2)
    ss_total = float(tab["sum_sq"].sum())
    tab["eta2"] = tab["sum_sq"] / ss_total if ss_total > 0 else np.nan
    tab.index = [i.replace("C(", "").replace(")", "") for i in tab.index]
    return tab


def specification_curve(df: pd.DataFrame, outcome: str, lo: Optional[str] = None, hi: Optional[str] = None
                        ) -> pd.DataFrame:
    """Sort specifications by the outcome and add a rank column (Simonsohn, Simmons & Nelson, 2020)."""
    d = df.sort_values(outcome).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    if lo and hi:
        d["excludes_zero"] = (d[lo] > 0) | (d[hi] < 0)
    return d


def claim_stability(df: pd.DataFrame, claim: Callable[[pd.Series], bool]) -> Dict[str, float]:
    """Fraction of specifications in which a boolean claim holds (e.g., lambda r: r['drift_index'] > 0.1)."""
    held = np.array([bool(claim(r)) for _, r in df.iterrows()])
    return {"fraction_holds": float(held.mean()) if len(held) else np.nan, "n_specs": int(len(held))}


def between_vs_within(df: pd.DataFrame, outcome: str, unit_col: str = "mouse", spec_col: str = "sorter") -> Dict[str, float]:
    """Compare the SD of the outcome across specifications (within unit) with the SD across units (within spec).

    Returns the mean within-unit SD across specifications, the mean between-unit SD within specifications, and
    their ratio (H3 in the README: ratio >= 0.5 means sorter choice matters as much as half the animal spread)."""
    within = df.groupby(unit_col)[outcome].std(ddof=1)
    between = df.groupby(spec_col)[outcome].std(ddof=1)
    w = float(within.mean())
    b = float(between.mean())
    return {"sd_across_specs_within_unit": w, "sd_across_units_within_spec": b,
            "ratio": w / b if b > 0 else np.nan}


def sorter_swap_null(df: pd.DataFrame, outcome: str, spec_col: str = "sorter", unit_col: str = "mouse",
                     n_perm: int = 500, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Null distribution of the between-specification SD obtained by shuffling specification labels within
    each unit (mouse). If the observed SD across sorters exceeds this null, sorter effects are not sampling noise."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = np.empty(n_perm)
    d = df[[outcome, spec_col, unit_col]].copy()
    for i in range(n_perm):
        d["_perm"] = d.groupby(unit_col)[spec_col].transform(lambda s: rng.permutation(s.to_numpy()))
        out[i] = d.groupby("_perm")[outcome].mean().std(ddof=1)
    return out


def summarize_grid(results: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Stack per-specification result dicts into a tidy DataFrame."""
    return pd.DataFrame(list(results))
