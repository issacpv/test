"""Evaluation: AUROC with cluster bootstrap, gray zone, attributable response, stratified AUROC."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def auroc_ci(y: Sequence[int], score: Sequence[float], groups: Optional[Sequence] = None, n_boot: int = 1000, alpha: float = 0.05, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """AUROC with a percentile bootstrap CI; resamples clusters (patients) when ``groups`` is given."""
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    g = np.arange(len(y)) if groups is None else np.asarray(groups)[ok]
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": int(len(y))}
    point = float(roc_auc_score(y, s))
    keys = np.unique(g)
    idx_of = {k: np.where(g == k)[0] for k in keys}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([idx_of[keys[i]] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = (np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if boots else (np.nan, np.nan))
    return {"auroc": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(y)), "n_clusters": int(len(keys))}


def gray_zone(score: Sequence[float], y: Sequence[int], sens_target: float = 0.9, spec_target: float = 0.9, n_boot: int = 500, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Gray zone of a continuous index (Cannesson et al., 2011): the interval between the cut-off giving
    ``sens_target`` sensitivity and the cut-off giving ``spec_target`` specificity, with bootstrap CIs of each bound
    and the fraction of observations that fall inside.

    Higher score is assumed to indicate responders.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = np.asarray(score, dtype=float)
    yy = np.asarray(y, dtype=int)
    ok = np.isfinite(s)
    s, yy = s[ok], yy[ok]

    def bounds(sv: np.ndarray, yv: np.ndarray) -> tuple:
        fpr, tpr, thr = roc_curve(yv, sv)
        # threshold with sensitivity >= target (largest threshold keeping tpr >= target)
        lo_candidates = thr[tpr >= sens_target]
        lo = float(np.max(lo_candidates)) if lo_candidates.size else float(np.min(sv))
        hi_candidates = thr[(1 - fpr) >= spec_target]
        hi = float(np.min(hi_candidates)) if hi_candidates.size else float(np.max(sv))
        lo, hi = min(lo, hi), max(lo, hi)
        return lo, hi

    lo, hi = bounds(s, yy)
    los, his = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        if len(np.unique(yy[idx])) < 2:
            continue
        l, h = bounds(s[idx], yy[idx])
        los.append(l)
        his.append(h)
    inside = float(np.mean((s >= lo) & (s <= hi)))
    return {
        "lower": lo, "upper": hi, "fraction_inside": inside,
        "lower_ci": (float(np.quantile(los, 0.025)), float(np.quantile(los, 0.975))) if los else (np.nan, np.nan),
        "upper_ci": (float(np.quantile(his, 0.025)), float(np.quantile(his, 0.975))) if his else (np.nan, np.nan),
    }


def attributable_response(n_resp_bolus: int, n_bolus: int, n_resp_sham: int, n_sham: int) -> Dict[str, float]:
    """Bolus response rate minus sham (regression-to-the-mean) response rate, with a risk-difference CI and z-test.

    ``attributable_fraction`` = (p_bolus - p_sham) / p_bolus, the share of observed responses not expected
    under no treatment.
    """
    p1, p0 = n_resp_bolus / n_bolus, n_resp_sham / n_sham
    se = np.sqrt(p1 * (1 - p1) / n_bolus + p0 * (1 - p0) / n_sham)
    z = (p1 - p0) / se if se > 0 else np.nan
    return {
        "p_bolus": p1, "p_sham": p0, "risk_difference": p1 - p0,
        "rd_ci_lo": p1 - p0 - 1.96 * se, "rd_ci_hi": p1 - p0 + 1.96 * se,
        "attributable_fraction": (p1 - p0) / p1 if p1 > 0 else np.nan,
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan,
    }


def stratified_auroc(df: pd.DataFrame, score_col: str, y_col: str, strata_col: str, group_col: Optional[str] = None, n_boot: int = 500, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """AUROC (with CI) of ``score_col`` for ``y_col`` within each level of ``strata_col``."""
    rows = []
    for level, g in df.groupby(strata_col):
        res = auroc_ci(g[y_col], g[score_col], None if group_col is None else g[group_col], n_boot=n_boot, rng=rng)
        rows.append({strata_col: level, **res, "prevalence": float(g[y_col].mean())})
    return pd.DataFrame(rows)


def bland_altman(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Bias and 95% limits of agreement of ``a - b`` (for VitalDB device-vs-waveform validation)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return {"bias": float(d.mean()), "sd": float(d.std(ddof=1)), "loa_lo": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_hi": float(d.mean() + 1.96 * d.std(ddof=1)), "n": int(d.size)}
