"""Flush-free damping features from routine beats and a damping-class classifier.

The features are the morphological consequences of under/over-damping: ringing
on the systolic upstroke (extra inflections, exaggerated normalised dP/dt,
narrow systolic peak), excess high-frequency power, and a blunted or absent
dicrotic notch in over-damping. Physiologic plausibility flags follow the
logic of the signal abnormality index of Sun, Reisner & Mark (Computers in
Cardiology 2006).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

PLAUSIBLE = {"sbp": (40.0, 300.0), "dbp": (20.0, 200.0), "pp": (10.0, 200.0), "hr": (25.0, 220.0)}
FEATURE_NAMES = ("sbp", "dbp", "map", "pp", "hr", "dpdt_max_norm", "sys_width50", "n_sys_inflections",
                 "notch_prominence", "hf_ratio", "rise_time")


def detect_onsets(abp: np.ndarray, fs: float, min_rr_s: float = 0.3) -> np.ndarray:
    """Beat feet from the slope-sum function (windowed sum of positive slopes) with an adaptive threshold."""
    d = np.diff(abp, prepend=abp[0])
    d[d < 0] = 0
    w = max(1, int(0.128 * fs))
    ssf = np.convolve(d, np.ones(w), mode="full")[: len(abp)]
    peaks, _ = find_peaks(ssf, height=0.4 * np.percentile(ssf, 98), distance=int(min_rr_s * fs))
    back = int(0.25 * fs)
    return np.unique([max(0, p - back) + int(np.argmin(abp[max(0, p - back): p + 1])) for p in peaks]).astype(int)


def _hf_ratio(beat: np.ndarray, fs: float, lo: float = 8.0, hi: float = 25.0) -> float:
    if beat.size < 16:
        return np.nan
    f, p = welch(beat - beat.mean(), fs=fs, nperseg=min(beat.size, 128))
    band = (f >= lo) & (f <= hi)
    total = (f >= 0.5) & (f <= hi)
    return float(p[band].sum() / p[total].sum()) if p[total].sum() > 0 else np.nan


def beat_damping_features(abp: np.ndarray, fs: float, onsets: np.ndarray | None = None) -> pd.DataFrame:
    """One row per beat with the columns in ``FEATURE_NAMES`` plus ``onset`` and ``plausible``."""
    if onsets is None:
        onsets = detect_onsets(abp, fs)
    rows = []
    for i in range(len(onsets) - 1):
        a, b = int(onsets[i]), int(onsets[i + 1])
        beat = np.asarray(abp[a:b], float)
        if beat.size < int(0.2 * fs):
            continue
        dbp, pk = float(beat[0]), int(np.argmax(beat))
        sbp = float(beat[pk])
        pp = sbp - dbp
        if pp <= 0:
            continue
        d1 = np.gradient(beat) * fs
        dpdt_norm = float(d1[: pk + 1].max() / pp) if pk > 0 else np.nan
        # width of the systolic peak at 50% of pulse pressure
        above = np.flatnonzero(beat >= dbp + 0.5 * pp)
        width50 = float((above.max() - above.min() + 1) / fs) if above.size else np.nan
        # inflections (sign changes of dP/dt) in the upper half of systole before the notch window
        sys_end = min(beat.size - 1, pk + int(0.1 * fs))
        upper = np.flatnonzero(beat[:sys_end] >= dbp + 0.5 * pp)
        n_infl = int(np.sum(np.diff(np.sign(d1[upper])) != 0)) if upper.size > 2 else 0
        # dicrotic notch prominence: deepest local minimum between peak+50 ms and peak+450 ms relative to PP
        lo_i, hi_i = min(beat.size - 1, pk + int(0.05 * fs)), min(beat.size, pk + int(0.45 * fs))
        notch_prom = np.nan
        if hi_i - lo_i > 3:
            mins, props = find_peaks(-beat[lo_i:hi_i], prominence=0)
            notch_prom = float(props["prominences"].max() / pp) if mins.size else 0.0
        rise = float(pk / fs)
        hr = 60.0 * fs / (b - a)
        row = {"onset": a, "sbp": sbp, "dbp": dbp, "map": float(beat.mean()), "pp": pp, "hr": hr,
               "dpdt_max_norm": dpdt_norm, "sys_width50": width50, "n_sys_inflections": n_infl,
               "notch_prominence": notch_prom, "hf_ratio": _hf_ratio(beat, fs), "rise_time": rise}
        row["plausible"] = all(PLAUSIBLE[k][0] <= row[k] <= PLAUSIBLE[k][1] for k in PLAUSIBLE)
        rows.append(row)
    return pd.DataFrame(rows)


def window_features(beats: pd.DataFrame, fs: float, window_s: float = 60.0) -> pd.DataFrame:
    """Median of beat features per window (plus beat count and plausible fraction)."""
    if beats.empty:
        return pd.DataFrame(columns=list(FEATURE_NAMES) + ["t_center", "n_beats", "plausible_frac"])
    w = np.floor(beats["onset"].to_numpy() / fs / window_s).astype(int)
    g = beats.assign(_w=w).groupby("_w")
    out = g[list(FEATURE_NAMES)].median()
    out["n_beats"] = g.size()
    out["plausible_frac"] = g["plausible"].mean()
    out["t_center"] = (out.index + 0.5) * window_s
    return out.reset_index(drop=True)


def fit_damping_classifier(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    """Gradient-boosted classifier from window features to flush-derived damping class."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
    return clf.fit(X[list(FEATURE_NAMES)], y)


def cv_auroc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 3, seed: int = 0) -> float:
    """Grouped cross-validated AUROC (binary y) so beats from one record never straddle folds."""
    from sklearn.model_selection import GroupKFold

    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
    p = cross_val_predict(clf, X[list(FEATURE_NAMES)], y, groups=groups, cv=GroupKFold(n_splits), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
