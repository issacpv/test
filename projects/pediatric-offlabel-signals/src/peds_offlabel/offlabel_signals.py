"""Signal comparison for age-off-label vs on-label paediatric reports.

Design: a below-floor report is by definition younger than an on-label
report of the same drug, so a naive within-drug comparison confounds
"off-label" with "younger". Every disproportionality here therefore uses an
**age-band-specific background**: the ROR of (drug, event) among below-floor
reports is computed against all other reports *in the same age band*, and
likewise for on-label reports; the two age-adjusted RORs are then compared
(ratio with Wald CI). Seriousness is modelled at report level with age band,
sex and reporter type as covariates. An interrupted-time-series helper
(segmented Poisson) supports the natural-experiment analysis around FDA
paediatric labelling changes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .label_ages import classify_age


def ror(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    est = (a * d) / (b * c)
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    return {"ror": float(est), "ror_lo": float(np.exp(np.log(est) - 1.96 * se)), "ror_hi": float(np.exp(np.log(est) + 1.96 * se)), "log_ror": float(np.log(est)), "se": se}


def classify_reports(reports: Sequence[Dict], floors: Dict[str, Dict[str, Any]], role: str = "suspect") -> pd.DataFrame:
    """One row per (report, drug) for drugs with a known floor.

    ``floors`` maps upper-case generic name -> output of
    :func:`peds_offlabel.label_ages.floor_from_label`. ``role`` selects the
    drug list (``"suspect"`` or ``"all"``).
    """
    rows = []
    for i, r in enumerate(reports):
        names = r.get("suspect", []) if role == "suspect" else [d["name"] for d in r.get("drugs", [])]
        for name in set(names):
            if name not in floors:
                continue
            rows.append(
                {
                    "row": i,
                    "safetyreportid": r.get("safetyreportid"),
                    "drug": name,
                    "age_years": r.get("age_years"),
                    "age_band": r.get("age_band"),
                    "sex": r.get("sex"),
                    "qualification": r.get("qualification"),
                    "serious": bool(r.get("serious")),
                    "death": bool(r.get("death")),
                    "label_class": classify_age(r.get("age_years"), floors[name]),
                    "n_reactions": len(r.get("reactions", [])),
                }
            )
    return pd.DataFrame(rows)


def age_stratum(r: Dict, years_per_bin: float = 1.0) -> Optional[str]:
    """Matching stratum for age-specific backgrounds.

    Numeric ages are binned into ``years_per_bin``-year bins (default 1 year,
    which is finer than any labelled floor); reports without numeric age fall
    back to their ICH band. A floor that falls *inside* a coarse band (e.g.
    6 years inside "child" 2-12) would otherwise leave residual age
    confounding, so band-level matching alone is not sufficient.
    """
    age = r.get("age_years")
    if age is not None:
        return f"y{int(age // years_per_bin)}"
    return r.get("age_band")


def age_specific_two_by_two(reports: Sequence[Dict], stratum: str, event: str, exposed_rows: set, stratum_fn: Callable[[Dict], Optional[str]] = age_stratum) -> Tuple[int, int, int, int]:
    """2x2 for ``event`` in one age stratum: exposed rows vs all other reports in it."""
    a = b = c = d = 0
    for i, r in enumerate(reports):
        if stratum_fn(r) != stratum:
            continue
        e = event in r.get("reactions", [])
        if i in exposed_rows:
            a += e
            b += not e
        else:
            c += e
            d += not e
    return a, b, c, d


def offlabel_signal_table(
    reports: Sequence[Dict],
    classified: pd.DataFrame,
    drug: str,
    min_a: int = 3,
    alpha: float = 0.05,
    stratum_fn: Callable[[Dict], Optional[str]] = age_stratum,
) -> pd.DataFrame:
    """Per-PT comparison of below-floor vs on-label reports for one drug.

    For each PT with >= ``min_a`` reports in either class, computes the
    age-stratum-specific ROR (Mantel-Haenszel across strata given by
    ``stratum_fn``, default 1-year bins) of the drug within the below-floor
    reports and within the on-label reports, their ratio (below / on-label)
    with Wald CI, and BH q-values.
    """
    sub = classified[classified["drug"] == drug]
    groups = {cls: set(sub.loc[sub["label_class"] == cls, "row"]) for cls in ("below_floor", "on_label_age")}
    counts = {cls: Counter() for cls in groups}
    for cls, rows in groups.items():
        for i in rows:
            counts[cls].update(set(reports[i].get("reactions", [])))
    events = sorted(e for e in set(counts["below_floor"]) | set(counts["on_label_age"]) if max(counts["below_floor"][e], counts["on_label_age"][e]) >= min_a)
    strata = sorted({stratum_fn(reports[i]) for rows in groups.values() for i in rows if stratum_fn(reports[i])})
    out = []
    for e in events:
        res: Dict[str, Any] = {"event": e, "a_below": counts["below_floor"][e], "a_on": counts["on_label_age"][e]}
        logs = {}
        for cls, rows in groups.items():
            num = den = 0.0
            var_terms = []
            for stratum in strata:
                a, b, c, d = age_specific_two_by_two(reports, stratum, e, rows, stratum_fn)
                n = a + b + c + d
                if n == 0 or (a + b) == 0:
                    continue
                num += a * d / n
                den += b * c / n
                var_terms.append((a, b, c, d))
            if den > 0 and num > 0:
                mh = num / den
                # Robins-Breslow-Greenland variance is heavy; use pooled-table Wald SE as approximation
                A = sum(t[0] for t in var_terms)
                B = sum(t[1] for t in var_terms)
                C = sum(t[2] for t in var_terms)
                D = sum(t[3] for t in var_terms)
                se = float(np.sqrt(sum(1 / max(x, 0.5) for x in (A, B, C, D))))
                logs[cls] = (float(np.log(mh)), se)
                res[f"ror_{cls}"] = float(mh)
            else:
                res[f"ror_{cls}"] = float("nan")
        if "below_floor" in logs and "on_label_age" in logs:
            diff = logs["below_floor"][0] - logs["on_label_age"][0]
            se = float(np.sqrt(logs["below_floor"][1] ** 2 + logs["on_label_age"][1] ** 2))
            res.update({"ratio": float(np.exp(diff)), "ratio_lo": float(np.exp(diff - 1.96 * se)), "ratio_hi": float(np.exp(diff + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(diff / se)))})
        else:
            res.update({"ratio": float("nan"), "ratio_lo": float("nan"), "ratio_hi": float("nan"), "p": float("nan")})
        out.append(res)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["q"] = benjamini_hochberg(df["p"].fillna(1.0).values)
    df["offlabel_excess"] = (df["ratio_lo"] > 1) & (df["q"] <= alpha)
    return df.sort_values("q").reset_index(drop=True)


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


def seriousness_model(classified: pd.DataFrame, outcome: str = "serious") -> pd.DataFrame:
    """Logistic ``outcome ~ below_floor + C(age_band) + C(sex) + C(qualification) + C(drug)``.

    Returns the odds-ratio table; the ``below_floor`` row is the adjusted OR
    of a serious outcome for age-off-label vs on-label paediatric reports.
    """
    import statsmodels.formula.api as smf

    df = classified[classified["label_class"].isin(["below_floor", "on_label_age"])].copy()
    df["below_floor"] = (df["label_class"] == "below_floor").astype(int)
    df["y"] = df[outcome].astype(int)
    for c in ("age_band", "sex", "qualification", "drug"):
        df[c] = df[c].fillna("unknown").astype(str)
    terms = ["below_floor"] + [f"C({c})" for c in ("age_band", "sex", "qualification", "drug") if df[c].nunique() > 1]
    res = smf.glm("y ~ " + " + ".join(terms), data=df, family=sm.families.Binomial()).fit()
    ci = res.conf_int()
    return pd.DataFrame({"or": np.exp(res.params), "or_lo": np.exp(ci[0]), "or_hi": np.exp(ci[1]), "p": res.pvalues})


def segmented_poisson_its(counts: Sequence[float], change_index: int, offset: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Interrupted time series: ``log E[y_t] = b0 + b1 t + b2 post_t + b3 (t - T0) post_t``.

    ``counts`` are monthly (or quarterly) report counts, ``change_index`` the
    first period after the intervention (e.g. an FDA paediatric labelling
    change), ``offset`` an optional exposure (e.g. log of all paediatric
    reports that period). Returns level (``exp(b2)``) and slope
    (``exp(b3)``) changes with CIs, plus a scale (overdispersion) estimate;
    use quasi-Poisson-style inference by multiplying SEs by sqrt(scale) when
    ``scale`` >> 1.
    """
    y = np.asarray(counts, dtype=float)
    t = np.arange(len(y), dtype=float)
    post = (t >= change_index).astype(float)
    X = pd.DataFrame({"t": t, "post": post, "t_post": (t - change_index) * post})
    X = sm.add_constant(X, has_constant="add")
    off = np.asarray(offset, dtype=float) if offset is not None else None
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=off)
    res = model.fit()
    pearson = float(np.sum(res.resid_pearson**2) / max(res.df_resid, 1))
    ci = res.conf_int()
    return {
        "level_ratio": float(np.exp(res.params["post"])),
        "level_lo": float(np.exp(ci.loc["post", 0])),
        "level_hi": float(np.exp(ci.loc["post", 1])),
        "slope_ratio": float(np.exp(res.params["t_post"])),
        "slope_lo": float(np.exp(ci.loc["t_post", 0])),
        "slope_hi": float(np.exp(ci.loc["t_post", 1])),
        "p_level": float(res.pvalues["post"]),
        "p_slope": float(res.pvalues["t_post"]),
        "scale": pearson,
    }


def class_shares_by_period(classified: pd.DataFrame, reports: Sequence[Dict], freq: str = "Q") -> pd.DataFrame:
    """Counts of below-floor / on-label / no-labelling reports per period."""
    dates = pd.to_datetime([reports[i].get("receivedate") for i in classified["row"]], format="%Y%m%d", errors="coerce")
    df = classified.assign(period=dates.to_period(freq))
    tab = df.groupby(["period", "label_class"]).size().unstack(fill_value=0)
    tab["share_below_floor"] = tab.get("below_floor", 0) / tab.sum(axis=1).replace(0, np.nan)
    return tab.reset_index()
