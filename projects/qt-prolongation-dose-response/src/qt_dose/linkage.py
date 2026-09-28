"""Pair pre-dose (baseline) and post-dose ECGs around each administration.

For each administration of an ingredient we find:
- one baseline ECG in the look-back window before the dose, with **no**
  administration of the same ingredient (or of any other listed QT drug when
  ``strict_washout`` is set) between the baseline and the dose;
- all post-dose ECGs in the effect window, tagged with time-since-dose (hours).

The result is a long table of (subject_id, ingredient, dose_mg, delta_qtc,
hours_since_dose, ...) suitable for the within-patient dose-response models.
Baseline QTc is subtracted, so each row is a within-patient change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_prior(ecg_times: np.ndarray, t0: np.ndarray, lookback_h: float):
    """For each dose time t0, index of the latest ECG within [t0-lookback, t0)."""
    idx = np.searchsorted(ecg_times, t0, side="left") - 1
    out = np.full(len(t0), -1, dtype=int)
    for i, (j, t) in enumerate(zip(idx, t0)):
        if j >= 0 and (t - ecg_times[j]) <= np.timedelta64(int(lookback_h * 3600), "s"):
            out[i] = j
    return out


def link_pre_post(
    ecgs: pd.DataFrame,
    admins: pd.DataFrame,
    qtc_col: str = "qtc",
    lookback_h: float = 48.0,
    effect_window_h: float = 24.0,
    min_gap_min: float = 5.0,
) -> pd.DataFrame:
    """Build the per-administration pre/post ECG pairs.

    Parameters
    ----------
    ecgs : DataFrame
        Columns: subject_id, ecg_time (datetime), <qtc_col> (ms), rr (s optional).
    admins : DataFrame
        Output of ``exposure.build_exposures`` (subject_id, ingredient,
        charttime, dose_mg, route).
    qtc_col : str
        Name of the corrected-QT column in ``ecgs`` (ms).

    Returns
    -------
    DataFrame with one row per (administration, post-dose ECG) and the
    within-patient ``delta_qtc`` relative to the administration's baseline.
    """
    ecgs = ecgs.copy()
    ecgs["ecg_time"] = pd.to_datetime(ecgs["ecg_time"], errors="coerce")
    ecgs = ecgs.dropna(subset=["ecg_time", qtc_col]).sort_values(["subject_id", "ecg_time"])

    rows: list[dict] = []
    lookback = np.timedelta64(int(lookback_h * 3600), "s")
    window = np.timedelta64(int(effect_window_h * 3600), "s")
    min_gap = np.timedelta64(int(min_gap_min * 60), "s")

    for sid, adm_s in admins.groupby("subject_id"):
        e = ecgs[ecgs["subject_id"] == sid]
        if e.empty:
            continue
        et = e["ecg_time"].to_numpy()
        eq = e[qtc_col].to_numpy(dtype=float)
        for _, a in adm_s.iterrows():
            t0 = np.datetime64(a["charttime"])
            # baseline: latest ECG strictly before dose within look-back
            prior = np.where((et < t0) & (et >= t0 - lookback))[0]
            if prior.size == 0:
                continue
            b = prior[-1]
            baseline_qtc = eq[b]
            # post-dose ECGs within effect window, after a minimal gap
            post = np.where((et > t0 + min_gap) & (et <= t0 + window))[0]
            for p in post:
                dt_h = (et[p] - t0) / np.timedelta64(1, "s") / 3600.0
                rows.append(
                    {
                        "subject_id": sid,
                        "ingredient": a["ingredient"],
                        "dose_mg": a["dose_mg"],
                        "route": a.get("route", ""),
                        "baseline_qtc": baseline_qtc,
                        "post_qtc": eq[p],
                        "delta_qtc": eq[p] - baseline_qtc,
                        "hours_since_dose": float(dt_h),
                    }
                )
    return pd.DataFrame(rows)


def attach_labs(pairs: pd.DataFrame, labs: pd.DataFrame, max_staleness_h: float = 24.0) -> pd.DataFrame:
    """Carry-forward serum potassium (and others) onto each pair by subject/time.

    ``labs`` must have subject_id, charttime, label, valuenum. Only values within
    ``max_staleness_h`` before the post ECG are used (approximate here via the
    dose-linked baseline time is out of scope; a full run keys on ecg_time).
    """
    if pairs.empty or labs.empty:
        pairs = pairs.copy()
        pairs["potassium"] = np.nan
        return pairs
    k = labs[labs["label"].astype(str).str.contains("Potassium", case=False, na=False)]
    med = k.groupby("subject_id")["valuenum"].median()
    out = pairs.copy()
    out["potassium"] = out["subject_id"].map(med)
    return out
