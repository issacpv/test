"""Subject-independent splits, leakage audit and standards-based evaluation.

Standards implemented (as evaluation *frames*, not as a claim of device validation):

* AAMI / ISO 81060-2:2018 criterion 1: mean error |ME| <= 5 mmHg and SD <= 8 mmHg over all
  paired readings, with >= 85 subjects; criterion 2: SD of the per-subject mean errors must not
  exceed a bound that depends on |ME| (table values 6.95, 6.87, 6.65, 6.29, 5.79, 5.14 for
  |ME| = 0..5 mmHg, linearly interpolated).
* IEEE 1708-2014 / 1708a-2019: grade from the mean absolute difference (MAD): A <= 5, B <= 6,
  C <= 7, D > 7 mmHg; the standard also prescribes calibration-interval testing, which is what
  :func:`time_blocked_split` is for.
* BHS (O'Brien et al., 1990): cumulative percentage of errors within 5 / 10 / 15 mmHg;
  A = 60/85/95, B = 50/75/90, C = 40/65/85.
* Bland-Altman limits of agreement for repeated measures per subject (Bland & Altman, 2007).

Change tracking (:func:`change_tracking`, :func:`bolus_response_table`) evaluates *within-subject
delta BP*, which is what the bolus-locked analysis needs.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# splits and leakage
# ----------------------------------------------------------------------------------------------


def subject_independent_split(subject_ids: Sequence, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold with subjects shuffled (sklearn's GroupKFold is not shuffled)."""
    s = np.asarray(subject_ids)
    uniq = np.unique(s)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_splits)
    out = []
    for f in folds:
        test = np.isin(s, f)
        out.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return out


def segment_level_split(n: int, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The *leaky* random split used by most published papers (for RQ5 only)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * test_frac))
    return perm[k:], perm[:k]


def time_blocked_split(subject_ids: Sequence, times: Sequence[float], train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Within each subject, the earliest ``train_frac`` of windows train, the rest test.

    Used for calibration-interval analysis (IEEE 1708): the model may see a subject's early data
    (as a calibration) and is tested on later data; time since the last training window is the
    calibration age.
    """
    s = np.asarray(subject_ids)
    t = np.asarray(times, dtype=float)
    train, test = [], []
    for u in np.unique(s):
        idx = np.flatnonzero(s == u)
        idx = idx[np.argsort(t[idx])]
        k = int(np.floor(len(idx) * train_frac))
        train.extend(idx[:k])
        test.extend(idx[k:])
    return np.array(train, int), np.array(test, int)


def leakage_audit(train_idx: np.ndarray, test_idx: np.ndarray, subject_ids: Sequence, record_ids: Sequence | None = None,
                  times: Sequence[float] | None = None, min_gap_s: float = 0.0, strict: bool = True) -> dict:
    """Check that train and test share no subject (and, optionally, no record / adjacent time).

    Raises ``ValueError`` in strict mode when any subject appears on both sides.
    """
    s = np.asarray(subject_ids)
    shared_subjects = np.intersect1d(np.unique(s[train_idx]), np.unique(s[test_idx]))
    out = {"n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
           "n_shared_subjects": int(len(shared_subjects)), "shared_subjects": shared_subjects.tolist()[:20]}
    if record_ids is not None:
        r = np.asarray(record_ids)
        out["n_shared_records"] = int(len(np.intersect1d(np.unique(r[train_idx]), np.unique(r[test_idx]))))
    if times is not None and record_ids is not None:
        t = np.asarray(times, dtype=float)
        r = np.asarray(record_ids)
        min_gap = np.inf
        for rec in np.intersect1d(np.unique(r[train_idx]), np.unique(r[test_idx])):
            tr = t[train_idx][r[train_idx] == rec]
            te = t[test_idx][r[test_idx] == rec]
            if len(tr) and len(te):
                min_gap = min(min_gap, float(np.min(np.abs(tr[:, None] - te[None, :]))))
        out["min_time_gap_s"] = min_gap
        out["temporal_adjacency"] = bool(min_gap < min_gap_s)
    if strict and out["n_shared_subjects"] > 0:
        raise ValueError(f"leakage: {out['n_shared_subjects']} subjects appear in both train and test")
    return out


# ----------------------------------------------------------------------------------------------
# standards
# ----------------------------------------------------------------------------------------------

_ISO_C2 = np.array([[0, 6.95], [1, 6.87], [2, 6.65], [3, 6.29], [4, 5.79], [5, 5.14]], dtype=float)


def aami_iso_evaluation(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: Sequence, min_subjects: int = 85,
                        min_per_subject: int = 3) -> dict:
    """AAMI / ISO 81060-2 criteria 1 and 2 plus the usual MAE / RMSE / correlation."""
    y, p, s = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(subject_ids)
    err = p - y
    me, sd = float(np.mean(err)), float(np.std(err, ddof=1))
    df = pd.DataFrame({"s": s, "e": err})
    per = df.groupby("s")["e"].agg(["mean", "count"])
    per = per[per["count"] >= min_per_subject]
    me_subj = float(per["mean"].mean()) if len(per) else np.nan
    sd_subj = float(per["mean"].std(ddof=1)) if len(per) > 1 else np.nan
    bound = float(np.interp(min(abs(me_subj), 5.0), _ISO_C2[:, 0], _ISO_C2[:, 1])) if np.isfinite(me_subj) else np.nan
    return {"n": int(len(y)), "n_subjects": int(len(per)), "enough_subjects": bool(len(per) >= min_subjects),
            "ME": me, "SD": sd, "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "r": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
            "criterion1_pass": bool(abs(me) <= 5 and sd <= 8),
            "ME_subject": me_subj, "SD_subject": sd_subj, "criterion2_bound": bound,
            "criterion2_pass": bool(np.isfinite(sd_subj) and abs(me_subj) <= 5 and sd_subj <= bound)}


def ieee_1708_grade(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """IEEE 1708 grade from the mean absolute difference."""
    mad = float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))
    grade = "A" if mad <= 5 else "B" if mad <= 6 else "C" if mad <= 7 else "D"
    return {"MAD": mad, "grade": grade}


def bhs_grade(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ae = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    p5, p10, p15 = (float(np.mean(ae <= k) * 100) for k in (5, 10, 15))
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        g = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        g = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        g = "C"
    else:
        g = "D"
    return {"pct_within_5": p5, "pct_within_10": p10, "pct_within_15": p15, "grade": g}


def bland_altman_repeated(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: Sequence) -> dict:
    """Bias and 95% limits of agreement with multiple observations per subject (Bland & Altman, 2007).

    Total variance = between-subject variance of the mean differences + within-subject variance,
    estimated from a one-way ANOVA on the differences.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.asarray(subject_ids)
    df = pd.DataFrame({"s": s, "d": d})
    g = df.groupby("s")["d"]
    n_i = g.count().to_numpy(dtype=float)
    k = len(n_i)
    N = n_i.sum()
    grand = df["d"].mean()
    ss_between = float(np.sum(n_i * (g.mean().to_numpy() - grand) ** 2))
    ss_within = float(np.sum((df["d"] - g.transform("mean")) ** 2))
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(N - k, 1)
    n0 = (N - np.sum(n_i ** 2) / N) / max(k - 1, 1)
    var_between = max((ms_between - ms_within) / n0, 0.0)
    total_sd = float(np.sqrt(var_between + ms_within))
    return {"bias": float(grand), "sd_total": total_sd, "loa_low": float(grand - 1.96 * total_sd),
            "loa_high": float(grand + 1.96 * total_sd), "sd_within": float(np.sqrt(ms_within)),
            "sd_between": float(np.sqrt(var_between))}


def trivial_baselines(y_train: np.ndarray, subjects_train: Sequence, y_test: np.ndarray, subjects_test: Sequence,
                      calib_idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Floors every model must beat: population mean (calibration-free) and per-subject
    calibration (mean of the subject's ``calib_idx`` test windows, else population mean)."""
    pop = float(np.mean(y_train))
    pred_pop = np.full(len(y_test), pop)
    pred_cal = pred_pop.copy()
    if calib_idx is not None:
        s = np.asarray(subjects_test)
        y = np.asarray(y_test, float)
        for u in np.unique(s[calib_idx]):
            m = calib_idx[s[calib_idx] == u]
            pred_cal[s == u] = float(np.mean(y[m]))
    return {"population_mean": pred_pop, "subject_calibrated": pred_cal}


# ----------------------------------------------------------------------------------------------
# change tracking
# ----------------------------------------------------------------------------------------------


def change_tracking(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: Sequence, min_change: float = 10.0,
                    min_windows: int = 5) -> dict:
    """Within-subject delta-BP tracking.

    Deltas are taken relative to the subject's own mean (equivalently: subject-centred values).
    Reports the pooled within-subject correlation, the median per-subject correlation, the
    within-subject regression slope of predicted on true deltas, and the direction agreement for
    true |delta| >= ``min_change`` relative to the subject's first window.
    """
    y, p, s = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(subject_ids)
    df = pd.DataFrame({"s": s, "y": y, "p": p})
    cnt = df.groupby("s")["y"].transform("count")
    df = df[cnt >= min_windows]
    if df.empty:
        return {"r_within_pooled": np.nan, "r_within_median": np.nan, "slope_within": np.nan,
                "direction_agreement": np.nan, "n_subjects": 0}
    dy = df["y"] - df.groupby("s")["y"].transform("mean")
    dp = df["p"] - df.groupby("s")["p"].transform("mean")
    r_pooled = float(np.corrcoef(dy, dp)[0, 1]) if dy.std() > 0 and dp.std() > 0 else np.nan
    per = []
    for _, g in df.groupby("s"):
        if g["y"].std() > 0 and g["p"].std() > 0:
            per.append(np.corrcoef(g["y"], g["p"])[0, 1])
    slope = float(np.sum(dy * dp) / np.sum(dy ** 2)) if np.sum(dy ** 2) > 0 else np.nan
    first_y = df.groupby("s")["y"].transform("first")
    first_p = df.groupby("s")["p"].transform("first")
    big = (df["y"] - first_y).abs() >= min_change
    agree = np.sign(df["y"] - first_y)[big] == np.sign(df["p"] - first_p)[big]
    return {"r_within_pooled": r_pooled, "r_within_median": float(np.median(per)) if per else np.nan,
            "slope_within": slope, "direction_agreement": float(agree.mean()) if big.any() else np.nan,
            "n_subjects": int(df["s"].nunique()), "n_big_changes": int(big.sum())}


def bolus_response_table(pred_df: pd.DataFrame, events: pd.DataFrame, pre: tuple[float, float] = (-180.0, -30.0),
                         post: tuple[float, float] = (60.0, 300.0), min_windows: int = 3,
                         value_cols: tuple[str, str] = ("sbp", "sbp_pred")) -> pd.DataFrame:
    """Event-locked pre/post medians for each bolus.

    ``pred_df`` has one row per accepted window with columns ``subject_id``, ``timestamp``
    (datetime) and the true / predicted values in ``value_cols``. ``events`` has ``subject_id``,
    ``starttime`` (datetime) and ``label`` (drug). Windows are assigned by seconds relative to the
    bolus start; each side needs >= ``min_windows`` accepted windows.
    Returns one row per event with delta_true, delta_pred and the coverage.
    """
    yt, yp = value_cols
    rows = []
    for ev in events.itertuples(index=False):
        d = pred_df[pred_df["subject_id"] == ev.subject_id]
        if d.empty:
            continue
        rel = (pd.to_datetime(d["timestamp"]) - pd.Timestamp(ev.starttime)).dt.total_seconds()
        a = d[(rel >= pre[0]) & (rel <= pre[1])]
        b = d[(rel >= post[0]) & (rel <= post[1])]
        if len(a) < min_windows or len(b) < min_windows:
            continue
        rows.append({"subject_id": ev.subject_id, "label": ev.label, "starttime": ev.starttime,
                     "n_pre": len(a), "n_post": len(b),
                     "pre_true": a[yt].median(), "post_true": b[yt].median(),
                     "pre_pred": a[yp].median(), "post_pred": b[yp].median(),
                     "delta_true": b[yt].median() - a[yt].median(), "delta_pred": b[yp].median() - a[yp].median()})
    return pd.DataFrame(rows)


def bolus_tracking_summary(tbl: pd.DataFrame, min_change: float = 10.0) -> dict:
    """Correlation, slope and sign agreement of predicted vs true bolus responses (per drug if
    ``label`` is present, plus pooled)."""
    def one(t: pd.DataFrame) -> dict:
        if len(t) < 3:
            return {"n": len(t), "r": np.nan, "slope": np.nan, "sign_agreement": np.nan}
        x, y = t["delta_true"].to_numpy(), t["delta_pred"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan
        slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)) if x.std() > 0 else np.nan
        big = np.abs(x) >= min_change
        sa = float(np.mean(np.sign(x[big]) == np.sign(y[big]))) if big.any() else np.nan
        return {"n": int(len(t)), "r": r, "slope": slope, "sign_agreement": sa}

    out = {"all": one(tbl)}
    if "label" in tbl:
        for lab, g in tbl.groupby("label"):
            out[str(lab)] = one(g)
    return out


# ----------------------------------------------------------------------------------------------
# uncertainty
# ----------------------------------------------------------------------------------------------


def cluster_bootstrap(metric_fn: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_pred: np.ndarray,
                      subject_ids: Sequence, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile CI of a scalar metric by resampling subjects (clusters) with replacement."""
    y, p, s = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(subject_ids)
    uniq = np.unique(s)
    groups = {u: np.flatnonzero(s == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        vals.append(metric_fn(y[idx], p[idx]))
    vals = np.asarray(vals, float)
    return {"estimate": float(metric_fn(y, p)), "ci_low": float(np.nanpercentile(vals, 2.5)),
            "ci_high": float(np.nanpercentile(vals, 97.5))}


def full_report(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: Sequence) -> dict:
    """All standard metrics in one dict (one BP component at a time)."""
    return {**aami_iso_evaluation(y_true, y_pred, subject_ids), **{f"ieee_{k}": v for k, v in ieee_1708_grade(y_true, y_pred).items()},
            **{f"bhs_{k}": v for k, v in bhs_grade(y_true, y_pred).items()},
            **{f"ba_{k}": v for k, v in bland_altman_repeated(y_true, y_pred, subject_ids).items()},
            **{f"track_{k}": v for k, v in change_tracking(y_true, y_pred, subject_ids).items()}}
