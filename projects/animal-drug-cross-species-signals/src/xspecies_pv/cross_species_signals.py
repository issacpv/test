"""Per-species disproportionality tables, cross-species concordance, lead-lag and dose scaling.

Each species' table is computed against its *own* database totals (a dog signal is
"more than expected among dog reports"), which is the only defensible denominator for
spontaneous reports. Concordance is then a comparison of two standardised signal surfaces on
the harmonised ingredient x organ-system grid.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- disproportionality
def bcpnn_ic(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Dict[str, np.ndarray]:
    """BCPNN information component with shrinkage (Norén et al. 2013) and 95% credibility bounds."""
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    n = a + b + c + d
    e = (a + b) * (a + c) / np.maximum(n, 1.0)
    ic = np.log2((a + 0.5) / (e + 0.5))
    s = np.maximum(a + 0.5, 0.5)
    return {"ic": ic, "ic025": ic - 3.3 * s ** -0.5 - 2.0 * s ** -1.5, "ic975": ic + 2.4 * s ** -0.5 - 0.5 * s ** -1.5,
            "expected": e}


def signal_table(events: pd.DataFrame, ingredient_col: str = "ingredient", bucket_col: str = "bucket",
                 report_col: str = "report_id", min_reports: int = 3) -> pd.DataFrame:
    """Ingredient x bucket 2x2 tables and IC from a long (report, ingredient, bucket) table.

    Each report contributes at most once per (ingredient, bucket) cell; totals are numbers of
    distinct reports. Returns columns ``ingredient, bucket, a, b, c, d, ic, ic025, ic975, signal``.
    """
    ev = events[[report_col, ingredient_col, bucket_col]].dropna().drop_duplicates()
    n_total = ev[report_col].nunique()
    a = ev.groupby([ingredient_col, bucket_col])[report_col].nunique().rename("a").reset_index()
    ing_tot = ev.groupby(ingredient_col)[report_col].nunique().rename("ing_total")
    buck_tot = ev.groupby(bucket_col)[report_col].nunique().rename("bucket_total")
    t = a.merge(ing_tot, left_on=ingredient_col, right_index=True).merge(buck_tot, left_on=bucket_col, right_index=True)
    t["b"] = (t["ing_total"] - t["a"]).clip(lower=0)
    t["c"] = (t["bucket_total"] - t["a"]).clip(lower=0)
    t["d"] = (n_total - t["a"] - t["b"] - t["c"]).clip(lower=0)
    ic = bcpnn_ic(t["a"].values, t["b"].values, t["c"].values, t["d"].values)
    for k in ("ic", "ic025", "ic975"):
        t[k] = ic[k]
    t["signal"] = (t["ic025"] > 0) & (t["a"] >= min_reports)
    return t.rename(columns={ingredient_col: "ingredient", bucket_col: "bucket"})[
        ["ingredient", "bucket", "a", "b", "c", "d", "ic", "ic025", "ic975", "signal"]]


# --------------------------------------------------------------------------- concordance
def _kappa(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(bool), y.astype(bool)
    n = len(x)
    if n == 0:
        return np.nan
    po = np.mean(x == y)
    pe = np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def align_tables(t1: pd.DataFrame, t2: pd.DataFrame, min_a: int = 1, suffixes: Tuple[str, str] = ("_1", "_2")) -> pd.DataFrame:
    """Inner-join two signal tables on (ingredient, bucket), keeping cells with a >= min_a in both."""
    m = t1.merge(t2, on=["ingredient", "bucket"], suffixes=suffixes)
    return m[(m[f"a{suffixes[0]}"] >= min_a) & (m[f"a{suffixes[1]}"] >= min_a)].reset_index(drop=True)


def concordance(t1: pd.DataFrame, t2: pd.DataFrame, min_a: int = 1, n_boot: int = 500, seed: int = 0,
                by_bucket: bool = True) -> Dict[str, object]:
    """Spearman rho of IC and kappa of signal flags between two species, with an ingredient-cluster bootstrap."""
    m = align_tables(t1, t2, min_a)
    rho = stats.spearmanr(m["ic_1"], m["ic_2"]).statistic if len(m) > 2 else np.nan
    kap = _kappa(m["signal_1"].values, m["signal_2"].values)
    rng = np.random.default_rng(seed)
    ings = m["ingredient"].unique()
    groups = {i: g for i, g in m.groupby("ingredient")}
    rhos, kaps = [], []
    for _ in range(n_boot):
        sel = rng.choice(ings, size=len(ings), replace=True)
        bs = pd.concat([groups[i] for i in sel], ignore_index=True)
        if len(bs) > 2 and bs["ic_1"].nunique() > 1 and bs["ic_2"].nunique() > 1:
            rhos.append(stats.spearmanr(bs["ic_1"], bs["ic_2"]).statistic)
            kaps.append(_kappa(bs["signal_1"].values, bs["signal_2"].values))
    out: Dict[str, object] = {
        "n_cells": int(len(m)), "n_ingredients": int(len(ings)), "rho": float(rho), "kappa": float(kap),
        "rho_ci": tuple(np.nanpercentile(rhos, [2.5, 97.5])) if rhos else (np.nan, np.nan),
        "kappa_ci": tuple(np.nanpercentile(kaps, [2.5, 97.5])) if kaps else (np.nan, np.nan),
        "contingency": pd.crosstab(m["signal_1"], m["signal_2"]),
    }
    if by_bucket:
        rows = []
        for b, g in m.groupby("bucket"):
            r = stats.spearmanr(g["ic_1"], g["ic_2"]).statistic if len(g) > 4 and g["ic_1"].nunique() > 1 else np.nan
            rows.append({"bucket": b, "n": len(g), "rho": r, "kappa": _kappa(g["signal_1"].values, g["signal_2"].values)})
        out["by_bucket"] = pd.DataFrame(rows)
    return out


def permutation_null_rho(t1: pd.DataFrame, t2: pd.DataFrame, n_perm: int = 500, seed: int = 0, min_a: int = 1) -> Dict[str, float]:
    """Null distribution of rho by permuting ingredient labels in ``t2`` (keeps each species' marginal structure)."""
    rng = np.random.default_rng(seed)
    m = align_tables(t1, t2, min_a)
    obs = stats.spearmanr(m["ic_1"], m["ic_2"]).statistic
    ings = list(t2["ingredient"].unique())
    null = []
    for _ in range(n_perm):
        perm = dict(zip(ings, rng.permutation(ings)))
        t2p = t2.assign(ingredient=t2["ingredient"].map(perm))
        mp = align_tables(t1, t2p, min_a)
        if len(mp) > 2 and mp["ic_1"].nunique() > 1 and mp["ic_2"].nunique() > 1:
            null.append(stats.spearmanr(mp["ic_1"], mp["ic_2"]).statistic)
    null = np.array(null)
    p = (1 + np.sum(null >= obs)) / (1 + len(null)) if len(null) else np.nan
    return {"observed": float(obs), "null_mean": float(null.mean()) if len(null) else np.nan, "p_value": float(p)}


# --------------------------------------------------------------------------- lead-lag
def cumulative_ic_by_quarter(events: pd.DataFrame, ingredient: str, bucket: str, quarter_col: str = "quarter",
                             ingredient_col: str = "ingredient", bucket_col: str = "bucket",
                             report_col: str = "report_id") -> pd.DataFrame:
    """Cumulative 2x2 and IC per quarter for one (ingredient, bucket) pair within one species."""
    ev = events[[report_col, ingredient_col, bucket_col, quarter_col]].dropna().drop_duplicates()
    qs = np.sort(ev[quarter_col].unique())
    rows = []
    for q in qs:
        sub = ev[ev[quarter_col] <= q]
        n = sub[report_col].nunique()
        a = sub[(sub[ingredient_col] == ingredient) & (sub[bucket_col] == bucket)][report_col].nunique()
        ing = sub[sub[ingredient_col] == ingredient][report_col].nunique()
        bk = sub[sub[bucket_col] == bucket][report_col].nunique()
        rows.append({"quarter": q, "a": a, "b": ing - a, "c": bk - a, "d": max(n - ing - bk + a, 0)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    ic = bcpnn_ic(t["a"].values, t["b"].values, t["c"].values, t["d"].values)
    t["ic"], t["ic025"] = ic["ic"], ic["ic025"]
    return t


def first_sustained_signal(traj: pd.DataFrame, consecutive: int = 2, min_a: int = 3) -> Optional[int]:
    """First quarter index at which IC025 > 0 (and a >= min_a) holds for ``consecutive`` quarters."""
    if traj is None or traj.empty:
        return None
    ok = ((traj["ic025"] > 0) & (traj["a"] >= min_a)).values
    run = 0
    for q, v in zip(traj["quarter"].values, ok):
        run = run + 1 if v else 0
        if run >= consecutive:
            return int(q) - consecutive + 1
    return None


def lead_lag(traj_animal: pd.DataFrame, traj_human: pd.DataFrame, **kw) -> Dict[str, Optional[int]]:
    """Quarters by which the animal signal precedes the human one (positive = animal first)."""
    fa, fh = first_sustained_signal(traj_animal, **kw), first_sustained_signal(traj_human, **kw)
    lead = (fh - fa) if (fa is not None and fh is not None) else None
    return {"first_animal": fa, "first_human": fh, "animal_lead_quarters": lead}


# --------------------------------------------------------------------------- dose scaling
_MASS_TO_MG = {"mg": 1.0, "milligram": 1.0, "g": 1000.0, "gram": 1000.0, "mcg": 1e-3, "ug": 1e-3, "microgram": 1e-3, "kg": 1e6}
_WEIGHT_TO_KG = {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "lb": 0.4536, "lbs": 0.4536, "pound": 0.4536, "pounds": 0.4536,
                 "g": 1e-3, "gram": 1e-3}


def mg_per_kg(dose_value: Optional[float], dose_unit: Optional[str], weight_min: Optional[float],
              weight_max: Optional[float], weight_unit: Optional[str], dose_denominator_unit: Optional[str] = None) -> Optional[float]:
    """Dose in mg/kg from CVM fields; ``None`` when units are unknown or the dose is already per kg."""
    if dose_value is None or dose_unit is None:
        return None
    u = str(dose_unit).strip().lower()
    if u not in _MASS_TO_MG:
        return None
    mg = float(dose_value) * _MASS_TO_MG[u]
    if dose_denominator_unit and str(dose_denominator_unit).strip().lower() in ("kg", "kilogram"):
        return mg  # already mg/kg
    w = [x for x in (weight_min, weight_max) if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not w or weight_unit is None:
        return None
    wu = str(weight_unit).strip().lower()
    if wu not in _WEIGHT_TO_KG:
        return None
    kg = float(np.mean([float(x) for x in w])) * _WEIGHT_TO_KG[wu]
    return mg / kg if kg > 0 else None


def allometric_dose_ratio(human_mg_per_kg: float, animal_mg_per_kg: float, human_kg: float = 70.0,
                          animal_kg: float = 20.0, exponent: float = 0.75) -> float:
    """Animal/human dose ratio after body-surface-area style scaling (mg/kg^exponent).

    A ratio > 1 means the animal receives more drug per unit metabolic mass than the human."""
    h = human_mg_per_kg * human_kg / human_kg ** exponent
    a = animal_mg_per_kg * animal_kg / animal_kg ** exponent
    return float(a / h)
