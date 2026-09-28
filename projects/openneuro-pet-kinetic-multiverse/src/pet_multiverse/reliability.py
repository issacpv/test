"""Test-retest reliability metrics per specification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def icc(Y: np.ndarray, kind: str = "2,1") -> float:
    """ICC(2,1) (absolute agreement) or ICC(3,1) (consistency) for an n×k matrix (Shrout & Fleiss)."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    if kind == "3,1":
        return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def icc_ci(Y: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate F-based CI for ICC(3,1) (used as a conservative interval for reporting)."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    F = ms_r / ms_e
    df1, df2 = n - 1, (n - 1) * (k - 1)
    fl = F / stats.f.ppf(1 - alpha / 2, df1, df2)
    fu = F * stats.f.ppf(1 - alpha / 2, df2, df1)
    return float((fl - 1) / (fl + k - 1)), float((fu - 1) / (fu + k - 1))


def within_subject_cv(Y: np.ndarray) -> float:
    """Mean within-subject coefficient of variation (%)."""
    Y = np.asarray(Y, float)
    return float(100 * np.mean(Y.std(axis=1, ddof=1) / np.abs(Y.mean(axis=1))))


def absolute_variability(Y: np.ndarray) -> float:
    """Mean |test − retest| / mean(test, retest) in % (two sessions)."""
    Y = np.asarray(Y, float)
    return float(100 * np.mean(np.abs(Y[:, 0] - Y[:, 1]) / np.abs(Y[:, :2].mean(axis=1))))


def bland_altman(Y: np.ndarray) -> dict[str, float]:
    Y = np.asarray(Y, float)
    diff = Y[:, 0] - Y[:, 1]
    return {"bias": float(diff.mean()), "loa_low": float(diff.mean() - 1.96 * diff.std(ddof=1)),
            "loa_high": float(diff.mean() + 1.96 * diff.std(ddof=1))}


def reliability_by_specification(results: pd.DataFrame, value: str = "bp", min_subjects: int = 4) -> pd.DataFrame:
    """ICC / CV / AV / Bland-Altman per (specification, region) for subjects with 2 sessions."""
    rows = []
    for (spec, region), g in results.groupby(["spec", "region"]):
        wide = g.pivot_table(index="subject", columns="session", values=value).dropna()
        wide = wide[np.isfinite(wide).all(axis=1)]
        if len(wide) < min_subjects or wide.shape[1] < 2:
            continue
        Y = wide.to_numpy()[:, :2]
        lo, hi = icc_ci(Y)
        rows.append({"spec": spec, "region": region, "n": len(wide), "icc21": icc(Y, "2,1"), "icc31": icc(Y, "3,1"),
                     "icc_ci_low": lo, "icc_ci_high": hi, "wscv_pct": within_subject_cv(Y), "av_pct": absolute_variability(Y),
                     **bland_altman(Y), "mean_bp": float(Y.mean())})
    out = pd.DataFrame(rows)
    return out.sort_values(["region", "icc21"], ascending=[True, False]).reset_index(drop=True)


def bootstrap_icc_difference(results: pd.DataFrame, spec_a: str, spec_b: str, region: str, value: str = "bp",
                             n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Bootstrap (over subjects) CI for ICC(spec_b) − ICC(spec_a) in a region."""
    rng = np.random.default_rng(seed)
    sub = results[results["region"] == region]
    wa = sub[sub["spec"] == spec_a].pivot_table(index="subject", columns="session", values=value).dropna()
    wb = sub[sub["spec"] == spec_b].pivot_table(index="subject", columns="session", values=value).dropna()
    common = wa.index.intersection(wb.index)
    A, B = wa.loc[common].to_numpy()[:, :2], wb.loc[common].to_numpy()[:, :2]
    point = icc(B) - icc(A)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(common), len(common))
        if len(np.unique(idx)) < 3:
            continue
        boots.append(icc(B[idx]) - icc(A[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_icc": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(common))}
