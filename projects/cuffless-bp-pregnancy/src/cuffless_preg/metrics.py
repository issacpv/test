"""Validation-standard metrics for cuffless BP: ISO 81060-2, BHS, IEEE 1708-style drift.

Errors are defined as estimate - reference in mmHg.  Subject-level bootstrap
is provided because the unit of inference is the subject (record), not the
beat.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# ISO 81060-2:2018 criterion 2: maximum permissible SD of the per-subject mean
# errors as a function of the overall mean error (|ME| in mmHg).
_ISO_C2_ME = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_ISO_C2_SD = np.array([6.95, 6.87, 6.65, 6.25, 5.64, 4.79])


def iso81060_criterion1(err: np.ndarray) -> dict[str, float | bool]:
    """Criterion 1: |mean error| <= 5 mmHg and SD of errors <= 8 mmHg (pooled over beats/readings)."""
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    me, sd = float(np.mean(err)), float(np.std(err, ddof=1)) if err.size > 1 else np.nan
    return {"n": int(err.size), "me": me, "sd": sd, "mae": float(np.mean(np.abs(err))), "pass": bool(abs(me) <= 5.0 and sd <= 8.0)}


def iso81060_criterion2(err: np.ndarray, subject: np.ndarray) -> dict[str, float | bool]:
    """Criterion 2: SD of per-subject mean errors must not exceed the ISO table value for the overall |ME|."""
    df = pd.DataFrame({"err": np.asarray(err, float), "s": np.asarray(subject)}).dropna()
    per = df.groupby("s")["err"].mean()
    me = float(df["err"].mean())
    sd_subj = float(per.std(ddof=1)) if len(per) > 1 else np.nan
    limit = float(np.interp(min(abs(me), 5.0), _ISO_C2_ME, _ISO_C2_SD)) if abs(me) <= 5.0 else 0.0
    return {"n_subjects": int(len(per)), "me": me, "sd_subject_means": sd_subj, "sd_limit": limit, "pass": bool(abs(me) <= 5.0 and sd_subj <= limit)}


def bhs_grade(err: np.ndarray) -> dict[str, float | str]:
    """British Hypertension Society grade from cumulative % of |errors| within 5/10/15 mmHg."""
    a = np.abs(np.asarray(err, float))
    a = a[np.isfinite(a)]
    p5, p10, p15 = (100.0 * np.mean(a <= k) for k in (5, 10, 15))
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        g = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        g = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        g = "C"
    else:
        g = "D"
    return {"pct_within_5": p5, "pct_within_10": p10, "pct_within_15": p15, "grade": g}


def calibration_drift(err: np.ndarray, t_since_calib_s: np.ndarray, bin_s: float = 300.0) -> pd.DataFrame:
    """IEEE 1708-style drift: mean and SD of error in time bins since calibration, plus the OLS slope of |err| per hour."""
    df = pd.DataFrame({"err": np.asarray(err, float), "t": np.asarray(t_since_calib_s, float)}).dropna()
    df = df[df["t"] >= 0]
    df["bin"] = (df["t"] // bin_s).astype(int)
    out = df.groupby("bin")["err"].agg(n="size", me="mean", sd="std", mae=lambda e: float(np.mean(np.abs(e)))).reset_index()
    out["t_start_s"] = out["bin"] * bin_s
    if len(df) > 2:
        X = np.column_stack([np.ones(len(df)), df["t"] / 3600.0])
        beta, *_ = np.linalg.lstsq(X, np.abs(df["err"]), rcond=None)
        out.attrs["abs_err_slope_per_hour"] = float(beta[1])
    return out


def bootstrap_by_subject(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], subject_col: str = "subject_id", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Cluster (subject) bootstrap of any scalar statistic of a beat table."""
    rng = np.random.default_rng(seed)
    subjects = df[subject_col].unique()
    groups = {s: g for s, g in df.groupby(subject_col)}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        vals.append(stat(pd.concat([groups[s] for s in draw], ignore_index=True)))
    vals = np.asarray(vals, float)
    return {"estimate": float(stat(df)), "ci_low": float(np.nanquantile(vals, alpha / 2)), "ci_high": float(np.nanquantile(vals, 1 - alpha / 2)), "n_boot": n_boot}


def permutation_group_difference(per_subject: pd.DataFrame, value: str, group_col: str, a: str, b: str, n_perm: int = 2000, seed: int = 0) -> dict[str, float]:
    """Permutation test for a difference of subject-level means between groups a and b."""
    rng = np.random.default_rng(seed)
    d = per_subject[per_subject[group_col].isin([a, b])].dropna(subset=[value])
    x = d[value].to_numpy()
    lab = (d[group_col] == a).to_numpy()
    obs = x[lab].mean() - x[~lab].mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(lab)
        null[i] = x[p].mean() - x[~p].mean()
    return {"observed": float(obs), "p_value": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)), "n_a": int(lab.sum()), "n_b": int((~lab).sum())}
