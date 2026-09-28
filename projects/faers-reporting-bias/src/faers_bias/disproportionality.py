"""Disproportionality measures, sex-stratified signals and bias adjustment.

Notation for a drug-event 2x2 table (report counts):

    a = drug & event      b = drug & other events
    c = other drugs & event   d = other drugs & other events

Implemented:

* ``ror``  reporting odds ratio with Wald 95% CI
* ``prr``  proportional reporting ratio with CI and the classical
  Evans (2001) criterion (PRR >= 2, chi2 >= 4, a >= 3)
* ``information_component``  BCPNN IC with the Noren et al. (2013) shrinkage
  (IC = log2((a + 0.5) / (E + 0.5)), E = (a+b)(a+c)/N) and IC025 lower bound
* ``sex_stratified_ror``  ROR in women and men separately, from a report-level
  frame or pre-computed stratum tables
* ``sex_interaction_test``  Wald test on log(ROR_f) - log(ROR_m) and a
  logistic-regression likelihood-ratio test for the drug x sex interaction
* ``bias_adjusted_ror``  logistic regression ``event ~ drug + reporter + post_dsc
  + drug:post_dsc [+ sex + drug:sex]`` that returns the drug coefficient net of
  reporter-type and stimulated-reporting effects, optionally with an exposure
  offset (denominator-aware)
* ``benjamini_hochberg``  FDR control for signal screens
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# --------------------------------------------------------------------- basics
def _cells(a: float, b: float, c: float, d: float, cc: float = 0.5) -> Tuple[float, float, float, float]:
    """Add a continuity correction only when a zero cell is present."""
    if min(a, b, c, d) == 0:
        return a + cc, b + cc, c + cc, d + cc
    return float(a), float(b), float(c), float(d)


def ror(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """Reporting odds ratio with Wald 95% CI and two-sided p-value."""
    a_, b_, c_, d_ = _cells(a, b, c, d)
    est = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    z = np.log(est) / se
    return {
        "ror": float(est),
        "ror_lo": float(np.exp(np.log(est) - 1.96 * se)),
        "ror_hi": float(np.exp(np.log(est) + 1.96 * se)),
        "log_ror": float(np.log(est)),
        "se_log_ror": float(se),
        "p": float(2 * stats.norm.sf(abs(z))),
        "n": float(a),
    }


def prr(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """Proportional reporting ratio, CI, chi-square and the Evans criterion."""
    a_, b_, c_, d_ = _cells(a, b, c, d)
    est = (a_ / (a_ + b_)) / (c_ / (c_ + d_))
    se = np.sqrt(1 / a_ - 1 / (a_ + b_) + 1 / c_ - 1 / (c_ + d_))
    chi2, p, _, _ = stats.chi2_contingency(np.array([[a_, b_], [c_, d_]]), correction=True)
    return {
        "prr": float(est),
        "prr_lo": float(np.exp(np.log(est) - 1.96 * se)),
        "prr_hi": float(np.exp(np.log(est) + 1.96 * se)),
        "chi2": float(chi2),
        "p": float(p),
        "evans_signal": bool(est >= 2 and chi2 >= 4 and a >= 3),
        "n": float(a),
    }


def information_component(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """BCPNN information component with shrinkage (Noren et al. 2013)."""
    n = a + b + c + d
    expected = (a + b) * (a + c) / n if n > 0 else 0.0
    ic = np.log2((a + 0.5) / (expected + 0.5))
    # approximate 95% credibility bounds (Noren 2013 uses gamma quantiles; the
    # normal approximation below is standard in screening code)
    se = 1.0 / np.sqrt(a + 0.5) if a >= 0 else np.nan
    return {"ic": float(ic), "ic025": float(ic - 1.96 * se), "ic975": float(ic + 1.96 * se), "expected": float(expected), "n": float(a)}


def two_by_two_from_frame(
    df: pd.DataFrame,
    drug: str,
    event: str,
    drug_col: str = "drug",
    event_col: str = "event",
    report_col: str = "safetyreportid",
) -> Dict[str, int]:
    """Build a 2x2 table from a long (report, drug, event) frame.

    Each report contributes once to each margin regardless of how many drugs /
    events it lists.
    """
    reports = df[report_col].unique()
    drug_reports = set(df.loc[df[drug_col] == drug, report_col])
    event_reports = set(df.loc[df[event_col] == event, report_col])
    a = len(drug_reports & event_reports)
    b = len(drug_reports) - a
    c = len(event_reports) - a
    d = len(reports) - a - b - c
    return {"a": a, "b": b, "c": c, "d": d}


# ------------------------------------------------------------ sex-stratified
def sex_stratified_ror(tables: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """ROR per sex stratum from ``{"female": {a,b,c,d}, "male": {a,b,c,d}}``.

    The stratum tables must be built *within* each sex (all four cells
    restricted to that sex) so that the comparator background is sex-specific.
    """
    rows = []
    for sex, t in tables.items():
        r = ror(t["a"], t["b"], t["c"], t["d"])
        r.update(information_component(t["a"], t["b"], t["c"], t["d"]))
        r["sex"] = sex
        rows.append(r)
    return pd.DataFrame(rows).set_index("sex")


def sex_interaction_test(tables: Dict[str, Dict[str, float]], f: str = "female", m: str = "male") -> Dict[str, float]:
    """Test whether the drug-event ROR differs between women and men.

    Returns the ratio of RORs (women / men) with CI and two tests:

    * Wald z on ``log ROR_f - log ROR_m`` (independent strata), and
    * a likelihood-ratio test comparing logistic models
      ``event ~ drug + sex`` vs ``event ~ drug * sex`` fitted to the aggregated
      2x2x2 counts (identical to the Breslow-Day idea for two strata).
    """
    rf = ror(**tables[f])
    rm = ror(**tables[m])
    diff = rf["log_ror"] - rm["log_ror"]
    se = np.sqrt(rf["se_log_ror"] ** 2 + rm["se_log_ror"] ** 2)
    z = diff / se
    # LRT via aggregated logistic regression
    rows = []
    for sex_val, key in ((1, f), (0, m)):
        t = tables[key]
        rows += [
            {"drug": 1, "sex": sex_val, "event": 1, "w": t["a"]},
            {"drug": 1, "sex": sex_val, "event": 0, "w": t["b"]},
            {"drug": 0, "sex": sex_val, "event": 1, "w": t["c"]},
            {"drug": 0, "sex": sex_val, "event": 0, "w": t["d"]},
        ]
    agg = pd.DataFrame(rows)
    agg = agg[agg["w"] > 0]
    X0 = sm.add_constant(agg[["drug", "sex"]].astype(float))
    X1 = X0.copy()
    X1["drug_x_sex"] = X1["drug"] * X1["sex"]
    fam = sm.families.Binomial()
    m0 = sm.GLM(agg["event"].astype(float), X0, family=fam, freq_weights=agg["w"].astype(float)).fit()
    m1 = sm.GLM(agg["event"].astype(float), X1, family=fam, freq_weights=agg["w"].astype(float)).fit()
    lr = 2 * (m1.llf - m0.llf)
    p_lrt = float(stats.chi2.sf(lr, 1))
    return {
        "ror_female": rf["ror"],
        "ror_male": rm["ror"],
        "ratio_of_ror": float(np.exp(diff)),
        "ratio_lo": float(np.exp(diff - 1.96 * se)),
        "ratio_hi": float(np.exp(diff + 1.96 * se)),
        "z": float(z),
        "p_wald": float(2 * stats.norm.sf(abs(z))),
        "lr_stat": float(lr),
        "p_lrt": p_lrt,
        "interaction_coef": float(m1.params["drug_x_sex"]),
    }


# ----------------------------------------------------------- bias adjustment
def bias_adjusted_ror(
    reports: pd.DataFrame,
    drug_col: str = "drug",
    event_col: str = "event",
    reporter_col: Optional[str] = "reporter",
    post_dsc_col: Optional[str] = "post_dsc",
    sex_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    exposure_offset_col: Optional[str] = None,
) -> pd.DataFrame:
    """Logistic-regression disproportionality net of known reporting biases.

    Fits ``event ~ drug + reporter + post_dsc + drug:post_dsc [+ sex + drug:sex]``
    on report-level rows (0/1 ``drug`` and ``event`` indicators). The exp of the
    ``drug`` coefficient is the adjusted ROR for reports from the *reference*
    reporter type (physician) outside stimulated-reporting windows; the
    ``drug:post_dsc`` term quantifies how much the signal is inflated after an
    FDA communication.

    If ``exposure_offset_col`` is given (log exposed persons for the report's
    drug x sex x year stratum, from ``denominators.merge_denominators``), the
    model becomes a Poisson rate model on aggregated counts, i.e. a
    denominator-aware disproportionality.
    """
    df = reports.copy()
    if reporter_col and reporter_col in df:
        df[reporter_col] = pd.Categorical(df[reporter_col])
        ref = "physician" if "physician" in df[reporter_col].cat.categories else df[reporter_col].cat.categories[0]
        df[reporter_col] = df[reporter_col].cat.reorder_categories([ref] + [c for c in df[reporter_col].cat.categories if c != ref])
    terms = [drug_col]
    if reporter_col and reporter_col in df:
        terms.append(f"C({reporter_col})")
    if post_dsc_col and post_dsc_col in df:
        terms += [post_dsc_col, f"{drug_col}:{post_dsc_col}"]
    if sex_col and sex_col in df:
        terms += [f"C({sex_col})", f"{drug_col}:C({sex_col})"]
    formula = f"{event_col} ~ " + " + ".join(terms)
    import statsmodels.formula.api as smf

    if exposure_offset_col:
        model = smf.glm(formula, data=df, family=sm.families.Poisson(), offset=df[exposure_offset_col])
    elif weight_col:
        model = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df[weight_col])
    else:
        model = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = model.fit()
    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "coef": res.params,
            "ratio": np.exp(res.params),
            "ratio_lo": np.exp(ci[0]),
            "ratio_hi": np.exp(ci[1]),
            "p": res.pvalues,
        }
    )
    out.index.name = "term"
    return out


def reporter_stratified_signal(tables: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """ROR / IC per reporter type (``{"consumer": {a,b,c,d}, "physician": ...}``).

    A drug-event pair whose signal is carried mainly by consumer or lawyer
    reports (large consumer ROR, null physician ROR) is flagged with
    ``reporter_discordant``.
    """
    df = sex_stratified_ror(tables)  # same computation, different strata
    df.index.name = "reporter"
    if {"consumer", "physician"}.issubset(df.index):
        df["reporter_discordant"] = (df.loc["consumer", "ror_lo"] > 1) & (df.loc["physician", "ror_hi"] < 1.5)
    return df


# ------------------------------------------------------------ multiplicity
def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """BH-adjusted q-values and rejection flags."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.minimum(q, 1.0)
    return pd.DataFrame({"p": p, "q": q_full, "reject": q_full <= alpha})


def screen_pairs(tables: pd.DataFrame, min_a: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    """Vectorised ROR / PRR / IC screen over a frame with columns a, b, c, d.

    Returns the input plus estimates, BH q-values on the ROR p-value and the
    conventional flags (ROR025 > 1 & a >= 3; IC025 > 0).
    """
    rows = []
    for _, r in tables.iterrows():
        d = ror(r["a"], r["b"], r["c"], r["d"])
        d.update({k: v for k, v in prr(r["a"], r["b"], r["c"], r["d"]).items() if k != "p" and k != "n"})
        d.update(information_component(r["a"], r["b"], r["c"], r["d"]))
        rows.append(d)
    est = pd.DataFrame(rows, index=tables.index)
    out = pd.concat([tables, est], axis=1)
    out["q"] = benjamini_hochberg(out["p"].fillna(1.0).values, alpha)["q"].values
    out["ror_signal"] = (out["ror_lo"] > 1) & (out["a"] >= min_a)
    out["ic_signal"] = out["ic025"] > 0
    return out
