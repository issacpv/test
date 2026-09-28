"""Longitudinal evaluation of brain-age delta against clinical trajectories.

OASIS-3 gives repeated MRI sessions and repeated clinical assessments (CDR,
MMSE) per participant.  Rather than treating each scan as independent, the
audit asks two questions for each (harmonisation x bias-correction) pipeline:

1. *Within-person* change: does corrected delta change track CDR / MMSE
   change, and is the estimated "acceleration" stable across pipelines?
2. Reliability: what is the test-retest ICC of delta for scans acquired
   close in time (e.g. < 6 months) on the same versus different scanners?

Mixed models are fitted with ``statsmodels.MixedLM`` (random intercept +
optional random slope per subject).  Clinical scores are aligned to the
nearest MRI session with a configurable tolerance (OASIS-3 convention: within
+-365 days of the scan).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def align_clinical_to_scans(
    scans: pd.DataFrame, clinical: pd.DataFrame, tolerance_days: int = 365,
    score_cols: Sequence[str] = ("cdr", "mmse"),
) -> pd.DataFrame:
    """Attach the nearest-in-time clinical assessment to each scan.

    Parameters
    ----------
    scans
        Columns ``subject``, ``days`` (days from baseline, OASIS-3 ``d0123``
        convention) and any delta columns.
    clinical
        Columns ``subject``, ``days`` and the score columns.
    tolerance_days
        Maximum |scan day - assessment day| to accept; otherwise NaN.
    """
    out = []
    clin = clinical.sort_values("days")
    for subj, grp in scans.groupby("subject"):
        c = clin[clin["subject"] == subj]
        for _, row in grp.iterrows():
            rec = row.to_dict()
            if len(c):
                j = int(np.argmin(np.abs(c["days"].values - row["days"])))
                if abs(c["days"].values[j] - row["days"]) <= tolerance_days:
                    for s in score_cols:
                        rec[s] = c.iloc[j][s]
                    rec["clin_lag_days"] = float(c["days"].values[j] - row["days"])
                else:
                    for s in score_cols:
                        rec[s] = np.nan
                    rec["clin_lag_days"] = np.nan
            out.append(rec)
    return pd.DataFrame(out)


def fit_delta_trajectory_model(
    df: pd.DataFrame,
    delta_col: str = "delta",
    time_col: str = "years",
    group_col: Optional[str] = "converter",
    covariates: Sequence[str] = ("age_baseline", "sex"),
    random_slope: bool = False,
):
    """Random-intercept (optionally random-slope) model of delta over time.

    Model: ``delta ~ years * group + covariates``, random effects by subject.
    The ``years:group`` interaction is the difference in delta slope between
    e.g. CDR-converters and stable participants: the headline longitudinal
    effect whose sign/magnitude should be compared across pipelines.

    Returns the fitted ``MixedLMResults``.
    """
    rhs = [time_col]
    if group_col:
        rhs = [f"{time_col} * {group_col}"]
    rhs += list(covariates)
    formula = f"{delta_col} ~ " + " + ".join(rhs)
    kwargs = {"groups": df["subject"]}
    if random_slope:
        kwargs["re_formula"] = f"~{time_col}"
    model = smf.mixedlm(formula, df.dropna(subset=[delta_col, time_col]), **kwargs)
    return model.fit(reml=True, method=["lbfgs"])


def within_person_change(df: pd.DataFrame, delta_col: str = "delta", time_col: str = "years") -> pd.DataFrame:
    """Per-subject OLS slope of delta over time (needs >= 2 visits).

    Returns ``subject``, ``n_visits``, ``span_years``, ``slope`` (years of
    delta per year), and the baseline delta.
    """
    rows = []
    for subj, g in df.dropna(subset=[delta_col, time_col]).groupby("subject"):
        if len(g) < 2 or g[time_col].nunique() < 2:
            continue
        slope, intercept = np.polyfit(g[time_col].values, g[delta_col].values, 1)
        rows.append({"subject": subj, "n_visits": len(g),
                     "span_years": float(g[time_col].max() - g[time_col].min()),
                     "slope": float(slope), "baseline_delta": float(g.sort_values(time_col)[delta_col].iloc[0])})
    return pd.DataFrame(rows)


def icc_test_retest(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, float]:
    """ICC(2,1) (two-way random, absolute agreement, single measure).

    Follows Shrout & Fleiss (1979).  Use for pairs of scans of the same
    subjects acquired close in time (same or different scanner).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n, k = a.size, 2
    data = np.vstack([a, b]).T
    grand = data.mean()
    ms_r = k * np.sum((data.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((data.mean(axis=0) - grand) ** 2) / (k - 1)
    resid = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_e = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    return {"icc21": float(icc), "n_pairs": int(n), "mean_abs_diff": float(np.mean(np.abs(a - b)))}


def simulate_longitudinal_cohort(
    n_subjects: int = 150, visits: int = 3, interval_years: float = 2.0,
    frac_converters: float = 0.3, converter_accel: float = 1.0, noise_sd: float = 2.0, seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OASIS-3-like longitudinal brain-age data.

    Each subject has a baseline delta ~ N(0, 3); converters gain
    ``converter_accel`` years of delta per year; CDR rises for converters.
    Columns: ``subject, visit, years, age, age_baseline, sex, converter, delta, cdr, mmse``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        conv = int(rng.random() < frac_converters)
        age0 = rng.uniform(55, 85)
        d0 = rng.normal(0, 3)
        sex = int(rng.random() < 0.5)
        for v in range(visits):
            t = v * interval_years + rng.normal(0, 0.2)
            t = max(t, 0.0)
            delta = d0 + conv * converter_accel * t + rng.normal(0, noise_sd)
            cdr = 0.5 * conv * (t > interval_years) + rng.choice([0, 0.5], p=[0.9, 0.1]) * (1 - conv)
            mmse = 29 - 1.5 * conv * t + rng.normal(0, 1)
            rows.append({"subject": f"S{s:04d}", "visit": v, "years": t, "age": age0 + t,
                         "age_baseline": age0, "sex": sex, "converter": conv,
                         "delta": delta, "cdr": cdr, "mmse": float(np.clip(mmse, 0, 30))})
    return pd.DataFrame(rows)


def pipeline_comparison_table(fits: Dict[str, object], term: str = "years:converter") -> pd.DataFrame:
    """Collect a coefficient (default: slope difference converters vs stable)
    across fitted mixed models keyed by pipeline name."""
    rows = []
    for name, res in fits.items():
        rows.append({"pipeline": name, "term": term,
                     "coef": float(res.params.get(term, np.nan)),
                     "se": float(res.bse.get(term, np.nan)),
                     "p": float(res.pvalues.get(term, np.nan))})
    return pd.DataFrame(rows)
