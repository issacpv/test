"""Adverse-event profile comparison: biosimilar vs originator, launch-aligned.

Design choices encoded here:

* **Comparator-restricted disproportionality.** The background for a
  biosimilar's ROR is the *originator's* reports in the same calendar window
  (not the whole of FAERS), so indication, route and population are matched
  by construction (``comparator_ror``).
* **Launch alignment.** Reporting is not stationary after a launch (Weber
  effect: Weber, 1984; Hoffman et al., 2014, *Drug Saf*). Two alignments are
  provided: calendar-time windows (originator reports received in the same
  months as the biosimilar's first N months on the market) and
  time-since-launch windows (biosimilar-to-biosimilar comparisons at equal
  months since launch).
* **Profile distance.** Jensen-Shannon divergence between PT frequency
  vectors with a report-level permutation null, which tests "is the whole
  profile different?" before any PT-level testing.
* **Mechanism panels.** PT panels for effectiveness/nocebo, injection-site,
  device, hypersensitivity/immunogenicity and product-substitution events,
  whose pooled counts are the pre-specified secondary endpoints.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PANELS: Dict[str, Tuple[str, ...]] = {
    "effectiveness": ("DRUG INEFFECTIVE", "TREATMENT FAILURE", "DRUG EFFECT DECREASED", "CONDITION AGGRAVATED", "DISEASE RECURRENCE", "DISEASE PROGRESSION", "THERAPEUTIC PRODUCT EFFECT DECREASED"),
    "injection_site": ("INJECTION SITE PAIN", "INJECTION SITE ERYTHEMA", "INJECTION SITE REACTION", "INJECTION SITE SWELLING", "INJECTION SITE BRUISING", "INJECTION SITE PRURITUS", "INJECTION SITE HAEMORRHAGE"),
    "device": ("DEVICE MALFUNCTION", "DEVICE USE ISSUE", "DEVICE DIFFICULT TO USE", "NEEDLE ISSUE", "PRODUCT DOSE OMISSION ISSUE", "INJECTION SITE DISCOMFORT", "DEVICE DELIVERY SYSTEM ISSUE"),
    "hypersensitivity": ("HYPERSENSITIVITY", "ANAPHYLACTIC REACTION", "INFUSION RELATED REACTION", "URTICARIA", "ANGIOEDEMA", "DRUG SPECIFIC ANTIBODY PRESENT"),
    "substitution": ("PRODUCT SUBSTITUTION ISSUE", "WRONG PRODUCT ADMINISTERED", "PRODUCT PRESCRIBING ISSUE", "INTERCHANGE OF PRODUCTS"),
}


# ------------------------------------------------------------ 2x2 and ROR
def ror(a: float, b: float, c: float, d: float) -> Dict[str, float]:
    """Reporting odds ratio with Wald 95% CI (Haldane 0.5 correction on zeros)."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    est = (a * d) / (b * c)
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    z = np.log(est) / se
    return {"ror": float(est), "ror_lo": float(np.exp(np.log(est) - 1.96 * se)), "ror_hi": float(np.exp(np.log(est) + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(z)))}


def pt_counts(reports: Iterable[Dict], key: str = "reactions") -> Tuple[Counter, int]:
    """Report-level PT counts (a PT counts once per report) and number of reports."""
    cnt: Counter = Counter()
    n = 0
    for r in reports:
        n += 1
        cnt.update(set(r.get(key, [])))
    return cnt, n


def comparator_ror(exposed: Sequence[Dict], comparator: Sequence[Dict], min_a: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    """ROR of every PT for ``exposed`` (e.g. biosimilar) vs ``comparator``
    (e.g. originator) reports, with BH q-values.

    Columns: event, a, b, c, d, ror, ror_lo, ror_hi, p, q, signal.
    """
    ce, ne = pt_counts(exposed)
    cc, nc = pt_counts(comparator)
    rows = []
    for pt, a in ce.items():
        if a < min_a:
            continue
        c = cc.get(pt, 0)
        r = ror(a, ne - a, c, nc - c)
        rows.append({"event": pt, "a": a, "b": ne - a, "c": c, "d": nc - c, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = benjamini_hochberg(df["p"].values)
    df["signal"] = (df["ror_lo"] > 1) & (df["q"] <= alpha)
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


# ---------------------------------------------------------- profile distance
def profile_vector(reports: Sequence[Dict], vocab: Sequence[str]) -> np.ndarray:
    cnt, n = pt_counts(reports)
    v = np.array([cnt.get(t, 0) for t in vocab], dtype=float)
    return v / v.sum() if v.sum() > 0 else v


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits (0 identical ... 1 disjoint)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def profile_distance_test(group_a: Sequence[Dict], group_b: Sequence[Dict], top_k: int = 200, n_perm: int = 500, seed: int = 0) -> Dict[str, float]:
    """JS divergence between two report sets with a report-level permutation null.

    The vocabulary is the ``top_k`` PTs in the pooled reports. The p-value is
    the fraction of label permutations with divergence >= observed.
    """
    pooled = list(group_a) + list(group_b)
    cnt, _ = pt_counts(pooled)
    vocab = [t for t, _ in cnt.most_common(top_k)]
    obs = jensen_shannon(profile_vector(group_a, vocab), profile_vector(group_b, vocab))
    rng = np.random.default_rng(seed)
    na = len(group_a)
    null = np.empty(n_perm)
    idx = np.arange(len(pooled))
    for i in range(n_perm):
        rng.shuffle(idx)
        pa = [pooled[j] for j in idx[:na]]
        pb = [pooled[j] for j in idx[na:]]
        null[i] = jensen_shannon(profile_vector(pa, vocab), profile_vector(pb, vocab))
    return {"js": obs, "null_mean": float(null.mean()), "p_perm": float((np.sum(null >= obs) + 1) / (n_perm + 1)), "n_a": na, "n_b": len(group_b)}


# ------------------------------------------------------------ launch windows
def _month_index(receivedate: Optional[str]) -> Optional[int]:
    if not receivedate or len(str(receivedate)) < 6:
        return None
    s = str(receivedate)
    return int(s[:4]) * 12 + int(s[4:6]) - 1


def months_since_launch(reports: Sequence[Dict], launch: str) -> List[Optional[int]]:
    """Months between the biosimilar launch (``YYYY-MM``) and each report's receipt."""
    y, m = int(launch[:4]), int(launch[5:7])
    l0 = y * 12 + m - 1
    out = []
    for r in reports:
        mi = _month_index(r.get("receivedate"))
        out.append(None if mi is None else mi - l0)
    return out


def calendar_window(reports: Sequence[Dict], launch: str, n_months: int) -> List[Dict]:
    """Reports received in the ``n_months`` starting at ``launch`` (calendar alignment)."""
    ms = months_since_launch(reports, launch)
    return [r for r, k in zip(reports, ms) if k is not None and 0 <= k < n_months]


def launch_curve(reports: Sequence[Dict], launch: str, n_months: int = 48) -> pd.DataFrame:
    """Monthly report counts by months since launch (0..n_months-1)."""
    ms = months_since_launch(reports, launch)
    cnt = Counter(k for k in ms if k is not None and 0 <= k < n_months)
    return pd.DataFrame({"month": range(n_months), "count": [cnt.get(k, 0) for k in range(n_months)]})


def weber_index(curve: pd.DataFrame) -> Dict[str, float]:
    """Crude Weber-effect descriptors from a launch curve.

    ``peak_month``: month of maximum count; ``early_late_ratio``: mean monthly
    count in months 0-23 divided by months 24-47 (> 1 indicates the classical
    early peak); ``frac_first_year``: fraction of reports in year 1.
    """
    c = curve["count"].values.astype(float)
    total = c.sum()
    early = c[:24].mean() if len(c) >= 24 else c.mean()
    late = c[24:48].mean() if len(c) >= 48 and c[24:48].sum() > 0 else np.nan
    return {
        "peak_month": int(np.argmax(c)) if total > 0 else -1,
        "early_late_ratio": float(early / late) if late and not np.isnan(late) else float("nan"),
        "frac_first_year": float(c[:12].sum() / total) if total > 0 else float("nan"),
    }


# ------------------------------------------------------------------- panels
def panel_counts(reports: Sequence[Dict], panels: Dict[str, Tuple[str, ...]] = PANELS) -> pd.DataFrame:
    """Number and fraction of reports mentioning at least one PT of each panel."""
    n = len(reports)
    rows = []
    for name, pts in panels.items():
        s = set(pts)
        k = sum(1 for r in reports if s & set(r.get("reactions", [])))
        rows.append({"panel": name, "n_reports": k, "fraction": k / n if n else float("nan")})
    return pd.DataFrame(rows)


def panel_comparison(exposed: Sequence[Dict], comparator: Sequence[Dict], panels: Dict[str, Tuple[str, ...]] = PANELS) -> pd.DataFrame:
    """Panel-level ROR (exposed vs comparator) with Fisher exact p-values."""
    rows = []
    ne, nc = len(exposed), len(comparator)
    for name, pts in panels.items():
        s = set(pts)
        a = sum(1 for r in exposed if s & set(r.get("reactions", [])))
        c = sum(1 for r in comparator if s & set(r.get("reactions", [])))
        r = ror(a, ne - a, c, nc - c)
        _, p_f = stats.fisher_exact([[a, ne - a], [c, nc - c]])
        rows.append({"panel": name, "a": a, "n_exposed": ne, "c": c, "n_comparator": nc, **r, "p_fisher": float(p_f)})
    return pd.DataFrame(rows)


def stratified_ror(exposed: Sequence[Dict], comparator: Sequence[Dict], event: str, strata_key: str) -> Dict[str, float]:
    """Mantel-Haenszel ROR across strata (e.g. reporter type or sex) for one PT."""
    strata = sorted({r.get(strata_key) for r in list(exposed) + list(comparator)})
    num = den = 0.0
    for s in strata:
        e = [r for r in exposed if r.get(strata_key) == s]
        c = [r for r in comparator if r.get(strata_key) == s]
        n = len(e) + len(c)
        if n == 0:
            continue
        a = sum(event in set(r.get("reactions", [])) for r in e)
        b = len(e) - a
        cc = sum(event in set(r.get("reactions", [])) for r in c)
        d = len(c) - cc
        num += a * d / n
        den += b * cc / n
    return {"ror_mh": float(num / den) if den > 0 else float("nan"), "n_strata": len(strata)}
