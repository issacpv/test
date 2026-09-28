"""QC exclusion as a hidden multiverse.

Given a participant-level table with IQMs, a group label and an outcome, this
module enumerates plausible QC exclusion rules (thresholds on one or more
IQMs, applied in either direction) and reports for every cell of the grid:

* retained n per group and the *differential* exclusion rate
  (risk ratio of exclusion, patients vs controls);
* the group difference in the outcome (Hedges' g with CI, Welch t-test);
* achieved power for the retained n at the full-sample effect size.

The rules mirror thresholds used in the literature, e.g. mean framewise
displacement > 0.2 / 0.25 / 0.5 mm (Power et al., 2012; Satterthwaite et al.,
2012), fd_perc > 20 %, CJV > 0.45 (Ganzetti et al., 2016), or *relative*
rules such as "worst 10 % per dataset".  Combining cells into a specification
curve (Simonsohn et al., 2020) shows how much the reported effect depends on a
choice that is rarely reported (Elyounssi-style comparisons of QC pipelines in
pediatric cohorts found that different QC approaches exclude different
participants and that excluded participants differ systematically).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower


@dataclass(frozen=True)
class QCRule:
    """Exclude rows where ``iqm`` is beyond ``threshold``.

    ``higher_is_worse=True`` excludes ``iqm > threshold``; otherwise
    ``iqm < threshold``.  ``relative=True`` interprets ``threshold`` as a
    quantile (0-1) computed within ``by`` groups (e.g. per dataset).
    """

    iqm: str
    threshold: float
    higher_is_worse: bool = True
    relative: bool = False
    by: Optional[str] = None

    def label(self) -> str:
        op = ">" if self.higher_is_worse else "<"
        kind = f"q{self.threshold:g}" if self.relative else f"{self.threshold:g}"
        return f"{self.iqm}{op}{kind}"


def _cutoff(series: pd.Series, rule: QCRule, df: pd.DataFrame) -> pd.Series:
    if not rule.relative:
        return pd.Series(rule.threshold, index=series.index)
    q = rule.threshold if rule.higher_is_worse else 1 - rule.threshold
    if rule.by and rule.by in df.columns:
        return df.groupby(rule.by)[rule.iqm].transform(lambda s: s.quantile(q))
    return pd.Series(series.quantile(q), index=series.index)


def apply_rules(df: pd.DataFrame, rules: Iterable[QCRule]) -> pd.Series:
    """Boolean mask of rows *retained* after applying all rules (AND)."""
    keep = pd.Series(True, index=df.index)
    for r in rules:
        s = pd.to_numeric(df[r.iqm], errors="coerce")
        cut = _cutoff(s, r, df)
        bad = (s > cut) if r.higher_is_worse else (s < cut)
        keep &= ~bad.fillna(False)
    return keep


def effect_size(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Hedges' g (bias-corrected Cohen's d) with 95 % CI and Welch t-test."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return {"g": np.nan, "g_lo": np.nan, "g_hi": np.nan, "t": np.nan, "p": np.nan, "n1": n1, "n2": n2}
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * J
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"g": float(g), "g_lo": float(g - 1.96 * se), "g_hi": float(g + 1.96 * se),
            "t": float(t), "p": float(p), "n1": int(n1), "n2": int(n2)}


def exclusion_risk_ratio(keep: pd.Series, group: pd.Series, case_label, control_label) -> Dict[str, float]:
    """Risk ratio of being excluded (cases vs controls) with log-RR 95 % CI."""
    excl = ~keep
    a = int((excl & (group == case_label)).sum()); n1 = int((group == case_label).sum())
    b = int((excl & (group == control_label)).sum()); n2 = int((group == control_label).sum())
    p1 = (a + 0.5) / (n1 + 1); p2 = (b + 0.5) / (n2 + 1)
    rr = p1 / p2
    se = np.sqrt(1 / (a + 0.5) - 1 / (n1 + 1) + 1 / (b + 0.5) - 1 / (n2 + 1))
    return {"excl_rate_case": a / n1 if n1 else np.nan, "excl_rate_control": b / n2 if n2 else np.nan,
            "rr": float(rr), "rr_lo": float(np.exp(np.log(rr) - 1.96 * se)), "rr_hi": float(np.exp(np.log(rr) + 1.96 * se))}


def two_sample_power(g: float, n1: int, n2: int, solver: Optional[TTestIndPower] = None, alpha: float = 0.05) -> float:
    """Achieved power of a two-sample t-test at effect size ``g``.

    Uses ``statsmodels`` (noncentral t) and falls back to the normal
    approximation when the noncentral-t evaluation overflows (large n).
    """
    if n1 < 2 or n2 < 2 or g is None or np.isnan(g) or g == 0:
        return np.nan
    solver = solver or TTestIndPower()
    try:
        pw = float(solver.power(effect_size=abs(g), nobs1=n1, ratio=n2 / n1, alpha=alpha))
    except (ValueError, FloatingPointError):
        pw = np.nan
    if np.isnan(pw):
        ncp = abs(g) * np.sqrt(n1 * n2 / (n1 + n2))
        pw = float(stats.norm.sf(stats.norm.ppf(1 - alpha / 2) - ncp))
    return float(np.clip(pw, 0.0, 1.0))


def qc_multiverse(
    df: pd.DataFrame,
    rule_grid: Dict[str, Sequence[QCRule]],
    group_col: str,
    outcome_col: str,
    case_label="patient",
    control_label="control",
    covariates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Enumerate the Cartesian product of QC choices and score each cell.

    Parameters
    ----------
    rule_grid
        ``{"motion": [None, QCRule(...), ...], "cjv": [...]}``; ``None`` means
        "no rule on that axis".  The product over axes defines the multiverse.
    covariates
        Optional columns; if given, the outcome is residualised on them (OLS
        within the retained sample) before the group comparison.

    Returns
    -------
    One row per cell with the rule labels, retained n, exclusion RR, Hedges' g
    and its CI, p-value, and post-hoc power at the full-sample g.
    """
    full = effect_size(df.loc[df[group_col] == case_label, outcome_col],
                       df.loc[df[group_col] == control_label, outcome_col])
    power = TTestIndPower()
    axes = list(rule_grid)
    rows = []
    for combo in itertools.product(*[rule_grid[a] for a in axes]):
        rules = [r for r in combo if r is not None]
        keep = apply_rules(df, rules)
        sub = df[keep]
        y = pd.to_numeric(sub[outcome_col], errors="coerce")
        if covariates:
            X = np.column_stack([np.ones(len(sub))] + [pd.to_numeric(sub[c], errors="coerce").to_numpy() for c in covariates])
            ok = ~np.isnan(X).any(axis=1) & ~y.isna().to_numpy()
            beta = np.linalg.lstsq(X[ok], y.to_numpy()[ok], rcond=None)[0]
            y = pd.Series(y.to_numpy() - X @ beta, index=sub.index)
        es = effect_size(y[sub[group_col] == case_label], y[sub[group_col] == control_label])
        rr = exclusion_risk_ratio(keep, df[group_col], case_label, control_label)
        pw = two_sample_power(full["g"], es["n1"], es["n2"], power)
        row = {a: (r.label() if r is not None else "none") for a, r in zip(axes, combo)}
        row.update({"n_retained": int(keep.sum()), "frac_retained": float(keep.mean()), **rr, **es,
                    "power_at_full_g": pw, "g_full_sample": full["g"]})
        rows.append(row)
    return pd.DataFrame(rows)


def specification_summary(results: pd.DataFrame, g_col: str = "g", p_col: str = "p") -> Dict[str, float]:
    """Summaries for a specification curve of the multiverse."""
    g = results[g_col].dropna()
    return {
        "n_specs": int(len(results)),
        "g_median": float(g.median()),
        "g_min": float(g.min()),
        "g_max": float(g.max()),
        "g_range": float(g.max() - g.min()),
        "frac_significant": float((results[p_col] < 0.05).mean()),
        "frac_sign_consistent": float(max((g > 0).mean(), (g < 0).mean())),
        "n_retained_min": int(results["n_retained"].min()),
        "n_retained_max": int(results["n_retained"].max()),
    }


def standard_rule_grid() -> Dict[str, List[Optional[QCRule]]]:
    """A literature-derived default grid for T1w (cjv, snr) and BOLD (fd)."""
    return {
        "fd_mean": [None, QCRule("fd_mean", 0.2), QCRule("fd_mean", 0.25), QCRule("fd_mean", 0.5)],
        "fd_perc": [None, QCRule("fd_perc", 20.0)],
        "cjv": [None, QCRule("cjv", 0.45), QCRule("cjv", 0.9, relative=True, by="dataset_id")],
        "snr_total": [None, QCRule("snr_total", 0.1, higher_is_worse=False, relative=True, by="dataset_id")],
    }
