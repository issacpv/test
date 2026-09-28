"""Ingredient-level signal detection in CAERS with cross-system checks.

* :func:`screen` - ROR, PRR (Evans criterion) and BCPNN IC with shrinkage
  for every (ingredient, PT) pair inside the supplement stratum, BH q-values.
* :func:`serious_fraction_model` - logistic model of a serious outcome on
  ingredient class, sex, age band and period (the mandatory serious-report
  rule for supplements took effect in December 2007).
* :func:`segmented_poisson_its` - structural-break / stimulated-reporting
  test for monthly series (Dec 2007 mandatory reporting; product-specific
  media episodes).
* :func:`poisson_cusum` - Poisson CUSUM for emerging signals.
* :func:`cross_system_concordance` - CAERS vs FAERS agreement of
  ingredient-level signals (Spearman on log-ROR, Cohen's kappa on flags).
* :func:`reference_evaluation` - AUROC / AP of any score against a curated
  reference set (e.g., LiverTox herbal hepatotoxicity, FDA tainted-products).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ------------------------------------------------------------------ measures
def ror(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    est = (a * d) / (b * c)
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    z = np.log(est) / se
    return {"ror": float(est), "ror_lo": float(np.exp(np.log(est) - 1.96 * se)), "ror_hi": float(np.exp(np.log(est) + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(z)))}


def prr(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    if min(a, b, c, d) == 0:
        a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a2, b2, c2, d2 = a, b, c, d
    est = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
    chi2, _, _, _ = stats.chi2_contingency(np.array([[a2, b2], [c2, d2]]), correction=True)
    return {"prr": float(est), "chi2": float(chi2), "evans_signal": bool(est >= 2 and chi2 >= 4 and a >= 3)}


def information_component(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """BCPNN IC with the Noren et al. (2013) shrinkage; IC025 approximation."""
    n = a + b + c + d
    expected = (a + b) * (a + c) / n if n > 0 else 0.0
    ic = float(np.log2((a + 0.5) / (expected + 0.5)))
    ic025 = ic - 3.3 * (a + 0.5) ** (-0.5) - 2.0 * (a + 0.5) ** (-1.5)
    return {"ic": ic, "ic025": float(ic025), "expected": float(expected)}


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


# ------------------------------------------------------------------- screen
def index_reports(reports: Sequence[Dict], exposure_key: str = "ingredients") -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
    exp_idx: Dict[str, Set[int]] = {}
    ev_idx: Dict[str, Set[int]] = {}
    for i, r in enumerate(reports):
        for x in set(r.get(exposure_key, [])):
            exp_idx.setdefault(x, set()).add(i)
        for e in set(r.get("reactions", [])):
            ev_idx.setdefault(e, set()).add(i)
    return exp_idx, ev_idx


def screen(reports: Sequence[Dict], exposure_key: str = "ingredients", min_a: int = 3, alpha: float = 0.05, exposures: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """ROR / PRR / IC for every (exposure, PT) pair with a >= min_a.

    The background is all reports in ``reports`` (pass the supplement stratum
    for a within-supplement comparison, or supplements + foods for a broader
    background; both are reported in the study).
    """
    exp_idx, ev_idx = index_reports(reports, exposure_key)
    n = len(reports)
    rows = []
    for x, xr in exp_idx.items():
        if exposures is not None and x not in exposures:
            continue
        ev_counts = Counter()
        for i in xr:
            ev_counts.update(set(reports[i].get("reactions", [])))
        for e, a in ev_counts.items():
            if a < min_a:
                continue
            b = len(xr) - a
            c = len(ev_idx[e]) - a
            d = n - a - b - c
            row = {"exposure": x, "event": e, "a": a, "b": b, "c": c, "d": d}
            row.update(ror(a, b, c, d))
            row.update(prr(a, b, c, d))
            row.update(information_component(a, b, c, d))
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = benjamini_hochberg(df["p"].values)
    df["ror_signal"] = (df["ror_lo"] > 1) & (df["q"] <= alpha)
    df["ic_signal"] = df["ic025"] > 0
    return df.sort_values(["exposure", "q"]).reset_index(drop=True)


def serious_fraction_model(reports: Sequence[Dict], exposure: str, exposure_key: str = "ingredients", mandatory_date: str = "20071222") -> pd.DataFrame:
    """Logistic ``serious ~ exposed + sex + age_band + post_mandatory`` on reports.

    ``mandatory_date`` is the effective date of the Dietary Supplement and
    Nonprescription Drug Consumer Protection Act reporting requirement
    (serious AE reports from manufacturers mandatory from December 2007).
    """
    import statsmodels.formula.api as smf

    rows = []
    for r in reports:
        age = r.get("age_years")
        band = "unknown" if age is None else ("child" if age < 18 else ("adult" if age < 65 else "elderly"))
        rows.append({"serious": int(bool(r.get("serious"))), "exposed": int(exposure in r.get(exposure_key, [])), "sex": r.get("sex", "unknown"), "age_band": band, "post": int(str(r.get("date_created") or "0") >= mandatory_date)})
    df = pd.DataFrame(rows)
    terms = ["exposed"] + [f"C({c})" for c in ("sex", "age_band") if df[c].nunique() > 1] + (["post"] if df["post"].nunique() > 1 else [])
    res = smf.glm("serious ~ " + " + ".join(terms), data=df, family=sm.families.Binomial()).fit()
    ci = res.conf_int()
    return pd.DataFrame({"or": np.exp(res.params), "or_lo": np.exp(ci[0]), "or_hi": np.exp(ci[1]), "p": res.pvalues})


# --------------------------------------------------------------------- time
def monthly_series(reports: Sequence[Dict], date_key: str = "date_created", mask: Optional[Callable[[Dict], bool]] = None) -> pd.Series:
    """Monthly report counts (complete month index, zeros filled)."""
    dates = [str(r.get(date_key) or "") for r in reports if (mask is None or mask(r))]
    idx = pd.to_datetime([d for d in dates if len(d) >= 6], format="%Y%m%d", errors="coerce").dropna()
    if len(idx) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(1, index=idx).resample("MS").sum()
    return s.astype(float)


def segmented_poisson_its(counts: Sequence[float], change_index: int, offset: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Level and slope change at ``change_index`` (segmented Poisson regression)."""
    y = np.asarray(counts, dtype=float)
    t = np.arange(len(y), dtype=float)
    post = (t >= change_index).astype(float)
    X = sm.add_constant(pd.DataFrame({"t": t, "post": post, "t_post": (t - change_index) * post}), has_constant="add")
    off = np.asarray(offset, dtype=float) if offset is not None else None
    res = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
    ci = res.conf_int()
    scale = float(np.sum(res.resid_pearson**2) / max(res.df_resid, 1))
    return {
        "level_ratio": float(np.exp(res.params["post"])),
        "level_lo": float(np.exp(ci.loc["post", 0])),
        "level_hi": float(np.exp(ci.loc["post", 1])),
        "slope_ratio": float(np.exp(res.params["t_post"])),
        "p_level": float(res.pvalues["post"]),
        "p_slope": float(res.pvalues["t_post"]),
        "scale": scale,
    }


def poisson_cusum(counts: Sequence[float], expected: float, k: Optional[float] = None, h: float = 5.0) -> Dict[str, object]:
    """One-sided Poisson CUSUM for an increase from rate ``expected``.

    Reference value ``k`` defaults to the optimal value for detecting a
    doubling: (2m - m) / ln(2m / m) = m / ln 2. Returns the CUSUM path, the
    first alarm index (or -1) and the number of alarms after resets.
    """
    m = float(expected)
    k = m / np.log(2.0) if k is None else float(k)
    s = 0.0
    path = []
    alarms = []
    for i, y in enumerate(counts):
        s = max(0.0, s + float(y) - k)
        path.append(s)
        if s > h:
            alarms.append(i)
            s = 0.0
    return {"path": path, "first_alarm": alarms[0] if alarms else -1, "n_alarms": len(alarms), "k": k, "h": h}


# --------------------------------------------------------- cross-system
def cross_system_concordance(caers: pd.DataFrame, faers: pd.DataFrame, min_a: int = 3) -> Dict[str, float]:
    """Agreement of (exposure, event) signals between two ``screen`` tables."""
    m = caers.merge(faers, on=["exposure", "event"], suffixes=("_caers", "_faers"))
    m = m[(m["a_caers"] >= min_a) & (m["a_faers"] >= min_a)]
    if len(m) < 3:
        return {"n_pairs": int(len(m)), "spearman": float("nan"), "kappa": float("nan"), "jaccard_signals": float("nan")}
    rho, p = stats.spearmanr(np.log(m["ror_caers"]), np.log(m["ror_faers"]))
    f1, f2 = m["ror_signal_caers"].astype(int).values, m["ror_signal_faers"].astype(int).values
    po = np.mean(f1 == f2)
    pe = np.mean(f1) * np.mean(f2) + (1 - np.mean(f1)) * (1 - np.mean(f2))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    both = np.sum((f1 == 1) & (f2 == 1))
    either = np.sum((f1 == 1) | (f2 == 1))
    return {"n_pairs": int(len(m)), "spearman": float(rho), "spearman_p": float(p), "kappa": float(kappa), "jaccard_signals": float(both / either) if either else float("nan")}


# ------------------------------------------------------------ reference set
def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def reference_evaluation(table: pd.DataFrame, positives: Iterable[Tuple[str, str]], negatives: Iterable[Tuple[str, str]], score_cols: Sequence[str] = ("ror", "prr", "ic"), n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """AUROC (with bootstrap CI) of score columns against labelled (exposure, event) pairs.

    Pairs absent from ``table`` count as score = minimum observed (they were
    never reported >= min_a times, i.e. not detectable).
    """
    pos, neg = set(positives), set(negatives)
    keyed = table.set_index(["exposure", "event"])
    rng = np.random.default_rng(seed)
    rows = []
    for col in score_cols:
        floor = float(keyed[col].min()) if len(keyed) else 0.0
        s, y = [], []
        for pair, lab in [(p, 1) for p in pos] + [(p, 0) for p in neg]:
            s.append(float(keyed.loc[pair, col]) if pair in keyed.index else floor)
            y.append(lab)
        s_arr, y_arr = np.array(s), np.array(y)
        auc = roc_auc(s_arr, y_arr)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_arr), len(y_arr))
            boots.append(roc_auc(s_arr[idx], y_arr[idx]))
        boots = np.array([b for b in boots if not np.isnan(b)])
        rows.append({"score": col, "auroc": auc, "auroc_lo": float(np.percentile(boots, 2.5)) if len(boots) else np.nan, "auroc_hi": float(np.percentile(boots, 97.5)) if len(boots) else np.nan, "n_pos": int(y_arr.sum()), "n_neg": int((1 - y_arr).sum())})
    return pd.DataFrame(rows)


#: Starter hepatotoxicity reference (herbal/dietary supplement ingredients with
#: documented liver injury in LiverTox / DILIN; PT panel to pair with them).
HEPATOTOXIC_INGREDIENTS: Tuple[str, ...] = ("GREEN TEA EXTRACT", "KAVA", "KRATOM", "ASHWAGANDHA", "TURMERIC/CURCUMIN", "GARCINIA CAMBOGIA", "RED YEAST RICE", "NIACIN", "BODYBUILDING/SARM", "BLACK COHOSH")
HEPATIC_PTS: Tuple[str, ...] = ("HEPATOTOXICITY", "DRUG-INDUCED LIVER INJURY", "HEPATITIS", "LIVER INJURY", "JAUNDICE", "ALANINE AMINOTRANSFERASE INCREASED", "HEPATIC FAILURE", "LIVER FUNCTION TEST ABNORMAL", "HEPATIC ENZYME INCREASED")
