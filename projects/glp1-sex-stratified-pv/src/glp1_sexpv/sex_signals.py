"""Sex-stratified disproportionality with sex-specific backgrounds.

The two mistakes this module is designed to avoid:

1. Reporting "65 % of GLP-1 reports are from women" as a sex effect. That
   share is mostly exposure (who takes the drug) and reporting propensity;
   see :mod:`glp1_sexpv.denominators` for the rate-based treatment.
2. Computing a sex-stratified ROR against a *pooled* background. Here every
   2 x 2 cell is restricted to one sex, so ROR_f compares GLP-1 reports from
   women with non-GLP-1 reports from women.

Main entry points

* :func:`sex_tables` - female and male 2 x 2 tables for one PT.
* :func:`ratio_of_ror` - ROR_f / ROR_m with Wald CI and a likelihood-ratio
  test of the drug x sex term from a weighted logistic model.
* :func:`sex_difference_atlas` - the screen over all PTs with BH.
* :func:`adjusted_sex_interaction` - drug x sex term adjusted for indication
  (and any other categorical strata) via a weighted logistic model on
  aggregated counts; this is the indication-confounding control.
* :func:`mh_ror_by_sex` - Mantel-Haenszel ROR per sex pooled across strata.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def ror(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """ROR with Wald 95% CI; Haldane correction when any cell is zero."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    est = (a * d) / (b * c)
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    z = np.log(est) / se
    return {"ror": float(est), "ror_lo": float(np.exp(np.log(est) - 1.96 * se)), "ror_hi": float(np.exp(np.log(est) + 1.96 * se)), "log_ror": float(np.log(est)), "se": se, "p": float(2 * stats.norm.sf(abs(z)))}


def sex_tables(reports: Sequence[Dict], event: str, exposed: Callable[[Dict], bool]) -> Dict[str, Dict[str, int]]:
    """Female and male 2x2 tables (a, b, c, d) for ``event`` with sex-specific cells."""
    out: Dict[str, Dict[str, int]] = {}
    for sex in ("female", "male"):
        a = b = c = d = 0
        for r in reports:
            if r.get("sex") != sex:
                continue
            e = event in r.get("reactions", [])
            if exposed(r):
                a += e
                b += not e
            else:
                c += e
                d += not e
        out[sex] = {"a": a, "b": b, "c": c, "d": d}
    return out


def _weighted_logit(df: pd.DataFrame, cols: Sequence[str]):
    X = sm.add_constant(df[list(cols)].astype(float), has_constant="add")
    return sm.GLM(df["event"].astype(float), X, family=sm.families.Binomial(), freq_weights=df["w"].astype(float)).fit()


def ratio_of_ror(tables: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """ROR_f / ROR_m with Wald CI and LRT for the drug x sex interaction."""
    rf, rm = ror(**tables["female"]), ror(**tables["male"])
    diff = rf["log_ror"] - rm["log_ror"]
    se = float(np.sqrt(rf["se"] ** 2 + rm["se"] ** 2))
    rows = []
    for sex_val, key in ((1, "female"), (0, "male")):
        t = tables[key]
        rows += [
            {"drug": 1, "sex": sex_val, "event": 1, "w": t["a"]},
            {"drug": 1, "sex": sex_val, "event": 0, "w": t["b"]},
            {"drug": 0, "sex": sex_val, "event": 1, "w": t["c"]},
            {"drug": 0, "sex": sex_val, "event": 0, "w": t["d"]},
        ]
    df = pd.DataFrame(rows)
    df = df[df["w"] > 0].copy()
    df["drug_sex"] = df["drug"] * df["sex"]
    p_lrt = float("nan")
    if df["event"].nunique() == 2 and len(df) >= 6:
        m0 = _weighted_logit(df, ["drug", "sex"])
        m1 = _weighted_logit(df, ["drug", "sex", "drug_sex"])
        p_lrt = float(stats.chi2.sf(max(2 * (m1.llf - m0.llf), 0.0), 1))
    return {
        "ror_female": rf["ror"],
        "ror_female_lo": rf["ror_lo"],
        "ror_male": rm["ror"],
        "ror_male_lo": rm["ror_lo"],
        "ratio": float(np.exp(diff)),
        "ratio_lo": float(np.exp(diff - 1.96 * se)),
        "ratio_hi": float(np.exp(diff + 1.96 * se)),
        "p_wald": float(2 * stats.norm.sf(abs(diff / se))),
        "p_lrt": p_lrt,
        "a_female": tables["female"]["a"],
        "a_male": tables["male"]["a"],
    }


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def sex_difference_atlas(reports: Sequence[Dict], exposed: Callable[[Dict], bool], min_a: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    """Screen every PT reported >= ``min_a`` times in exposed women AND men.

    Returns one row per PT with sex-specific RORs, ratio, CIs, p-values, BH
    q-values (on the LRT p) and flags: ``signal_female`` / ``signal_male``
    (ROR lower bound > 1 and a >= min_a) and ``sex_differs`` (q <= alpha).
    """
    cnt = {"female": Counter(), "male": Counter()}
    for r in reports:
        if exposed(r) and r.get("sex") in cnt:
            cnt[r["sex"]].update(set(r.get("reactions", [])))
    events = sorted(e for e in set(cnt["female"]) | set(cnt["male"]) if cnt["female"][e] >= min_a and cnt["male"][e] >= min_a)
    rows = []
    for e in events:
        res = ratio_of_ror(sex_tables(reports, e, exposed))
        res["event"] = e
        rows.append(res)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = benjamini_hochberg(df["p_lrt"].fillna(1.0).values)
    df["signal_female"] = (df["ror_female_lo"] > 1) & (df["a_female"] >= min_a)
    df["signal_male"] = (df["ror_male_lo"] > 1) & (df["a_male"] >= min_a)
    df["sex_differs"] = df["q"] <= alpha
    cols = ["event", "a_female", "a_male", "ror_female", "ror_male", "ratio", "ratio_lo", "ratio_hi", "p_wald", "p_lrt", "q", "signal_female", "signal_male", "sex_differs"]
    return df[cols].sort_values("q").reset_index(drop=True)


def aggregate_counts(reports: Sequence[Dict], event: str, exposed: Callable[[Dict], bool], strata: Sequence[str] = ("indication",)) -> pd.DataFrame:
    """Aggregate reports to (drug, sex, strata..., event) cells with weights."""
    c: Counter = Counter()
    for r in reports:
        if r.get("sex") not in ("female", "male"):
            continue
        key = (int(exposed(r)), int(r["sex"] == "female"), *[str(r.get(s, "unknown")) for s in strata], int(event in r.get("reactions", [])))
        c[key] += 1
    rows = [dict(zip(["drug", "sex", *strata, "event"], k), w=v) for k, v in c.items()]
    return pd.DataFrame(rows)


def adjusted_sex_interaction(agg: pd.DataFrame, strata: Sequence[str] = ("indication",)) -> Dict[str, float]:
    """Drug x sex interaction adjusted for strata (e.g. indication, age band).

    Fits ``event ~ drug*sex + C(stratum) + drug:C(stratum) + sex:C(stratum)``
    on aggregated counts and returns exp(beta_drug:sex) with CI and LRT p.
    The crude (unadjusted) ratio is returned alongside for comparison.
    """
    import statsmodels.formula.api as smf

    df = agg.copy()
    if df.empty or df["event"].nunique() < 2:
        return {"ratio_adj": float("nan"), "ratio_crude": float("nan"), "p_lrt": float("nan")}
    strata_terms = " + ".join(f"C({s}) + drug:C({s}) + sex:C({s})" for s in strata)
    f1 = f"event ~ drug*sex + {strata_terms}"
    f0 = f"event ~ drug + sex + {strata_terms}"
    fam = sm.families.Binomial()
    m1 = smf.glm(f1, data=df, family=fam, freq_weights=df["w"]).fit()
    m0 = smf.glm(f0, data=df, family=fam, freq_weights=df["w"]).fit()
    crude1 = smf.glm("event ~ drug*sex", data=df, family=fam, freq_weights=df["w"]).fit()
    beta = float(m1.params["drug:sex"])
    se = float(m1.bse["drug:sex"])
    return {
        "ratio_adj": float(np.exp(beta)),
        "ratio_adj_lo": float(np.exp(beta - 1.96 * se)),
        "ratio_adj_hi": float(np.exp(beta + 1.96 * se)),
        "ratio_crude": float(np.exp(crude1.params["drug:sex"])),
        "p_lrt": float(stats.chi2.sf(max(2 * (m1.llf - m0.llf), 0.0), 1)),
    }


def mh_ror_by_sex(agg: pd.DataFrame, stratum: str = "indication") -> Dict[str, Dict[str, float]]:
    """Mantel-Haenszel ROR for each sex pooled across ``stratum`` levels."""
    out: Dict[str, Dict[str, float]] = {}
    for sex_val, name in ((1, "female"), (0, "male")):
        num = den = 0.0
        sub = agg[agg["sex"] == sex_val]
        for _, g in sub.groupby(stratum):
            def w(drug: int, event: int) -> float:
                m = (g["drug"] == drug) & (g["event"] == event)
                return float(g.loc[m, "w"].sum())

            a, b, c, d = w(1, 1), w(1, 0), w(0, 1), w(0, 0)
            n = a + b + c + d
            if n == 0:
                continue
            num += a * d / n
            den += b * c / n
        out[name] = {"ror_mh": float(num / den) if den > 0 else float("nan")}
    if all(k in out for k in ("female", "male")):
        out["ratio_mh"] = {"ratio": out["female"]["ror_mh"] / out["male"]["ror_mh"] if out["male"]["ror_mh"] else float("nan")}
    return out
