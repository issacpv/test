"""Flush-free damping features from routine beats and a damping-class classifier.

The features are the morphological consequences of under/over-damping: ringing
on the systolic upstroke (extra inflections, exaggerated normalised dP/dt,
systolic overshoot above the low-passed pulse), a resonant bump in the pulse
spectrum, and a slow rise with a blunted dicrotic notch in over-damping.
Physiologic plausibility flags follow the logic of the signal abnormality index
of Sun, Reisner & Mark (Computers in Cardiology 2006).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

PLAUSIBLE = {"sbp": (40.0, 300.0), "dbp": (20.0, 200.0), "pp": (10.0, 200.0), "hr": (25.0, 220.0)}
BEAT_FEATURES = ("sbp", "dbp", "map", "pp", "hr", "dpdt_max_norm", "sys_width50", "n_sys_inflections",
                 "notch_prominence", "overshoot", "rise_time")
WINDOW_FEATURES = ("spectral_peakiness", "hf_ratio")
FEATURE_NAMES = BEAT_FEATURES + WINDOW_FEATURES


def detect_onsets(abp: np.ndarray, fs: float, min_rr_s: float = 0.3) -> np.ndarray:
    """Beat feet from the slope-sum function (windowed sum of positive slopes) with an adaptive threshold."""
    d = np.diff(abp, prepend=abp[0])
    d[d < 0] = 0
    w = max(1, int(0.128 * fs))
    ssf = np.convolve(d, np.ones(w), mode="full")[: len(abp)]
    peaks, _ = find_peaks(ssf, height=0.4 * np.percentile(ssf, 98), distance=int(min_rr_s * fs))
    back = int(0.25 * fs)
    return np.unique([max(0, p - back) + int(np.argmin(abp[max(0, p - back): p + 1])) for p in peaks]).astype(int)


def _lowpass(x: np.ndarray, fs: float, fc: float = 5.0) -> np.ndarray:
    b, a = butter(2, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, padlen=min(len(x) - 1, 12))


def beat_damping_features(abp: np.ndarray, fs: float, onsets: np.ndarray | None = None) -> pd.DataFrame:
    """One row per beat with the columns in ``BEAT_FEATURES`` plus ``onset`` and ``plausible``."""
    abp = np.asarray(abp, float)
    if onsets is None:
        onsets = detect_onsets(abp, fs)
    smooth = _lowpass(abp, fs) if abp.size > 30 else abp
    rows = []
    for i in range(len(onsets) - 1):
        a, b = int(onsets[i]), int(onsets[i + 1])
        beat = abp[a:b]
        if beat.size < int(0.2 * fs):
            continue
        dbp, pk = float(beat[0]), int(np.argmax(beat))
        sbp = float(beat[pk])
        pp = sbp - dbp
        if pp <= 0:
            continue
        d1 = np.gradient(beat) * fs
        dpdt_norm = float(d1[: pk + 1].max() / pp) if pk > 0 else np.nan
        above = np.flatnonzero(beat >= dbp + 0.5 * pp)
        width50 = float((above.max() - above.min() + 1) / fs) if above.size else np.nan
        sys_end = min(beat.size - 1, pk + int(0.1 * fs))
        upper = np.flatnonzero(beat[:sys_end] >= dbp + 0.5 * pp)
        n_infl = int(np.sum(np.diff(np.sign(d1[upper])) != 0)) if upper.size > 2 else 0
        lo_i, hi_i = min(beat.size - 1, pk + int(0.05 * fs)), min(beat.size, pk + int(0.45 * fs))
        notch_prom = np.nan
        if hi_i - lo_i > 3:
            mins, props = find_peaks(-beat[lo_i:hi_i], prominence=0)
            notch_prom = float(props["prominences"].max() / pp) if mins.size else 0.0
        overshoot = float((sbp - smooth[a:b].max()) / pp)
        row = {"onset": a, "sbp": sbp, "dbp": dbp, "map": float(beat.mean()), "pp": pp, "hr": 60.0 * fs / (b - a),
               "dpdt_max_norm": dpdt_norm, "sys_width50": width50, "n_sys_inflections": n_infl,
               "notch_prominence": notch_prom, "overshoot": overshoot, "rise_time": float(pk / fs)}
        row["plausible"] = all(PLAUSIBLE[k][0] <= row[k] <= PLAUSIBLE[k][1] for k in PLAUSIBLE)
        rows.append(row)
    return pd.DataFrame(rows)


def spectral_features(x: np.ndarray, fs: float, band: tuple[float, float] = (5.0, 30.0)) -> dict[str, float]:
    """Resonance features of a multi-beat segment.

    ``spectral_peakiness``: max of f^2 * PSD in ``band`` relative to its median over 2-30 Hz
    (f^2 compensates the natural 1/f^2 roll-off of the pulse spectrum, so a resonant bump stands out).
    ``hf_ratio``: power fraction in 8-25 Hz relative to 0.5-25 Hz.
    """
    x = np.asarray(x, float)
    if x.size < 64:
        return {"spectral_peakiness": np.nan, "hf_ratio": np.nan}
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(x.size, 512))
    comp = p * np.maximum(f, 0.5) ** 2
    ref = (f >= 2.0) & (f <= 30.0)
    inb = (f >= band[0]) & (f <= band[1])
    peak = float(comp[inb].max() / np.median(comp[ref])) if inb.any() and np.median(comp[ref]) > 0 else np.nan
    hf = (f >= 8.0) & (f <= 25.0)
    tot = (f >= 0.5) & (f <= 25.0)
    return {"spectral_peakiness": peak, "hf_ratio": float(p[hf].sum() / p[tot].sum()) if p[tot].sum() > 0 else np.nan}


def window_features(abp: np.ndarray, fs: float, beats: pd.DataFrame, window_s: float = 60.0) -> pd.DataFrame:
    """Median beat features per window plus spectral features of the window's raw signal."""
    cols = list(FEATURE_NAMES) + ["t_center", "n_beats", "plausible_frac"]
    if beats.empty:
        return pd.DataFrame(columns=cols)
    w = np.floor(beats["onset"].to_numpy() / fs / window_s).astype(int)
    g = beats.assign(_w=w).groupby("_w")
    out = g[list(BEAT_FEATURES)].median()
    out["n_beats"] = g.size()
    out["plausible_frac"] = g["plausible"].mean()
    out["t_center"] = (out.index + 0.5) * window_s
    spec = []
    for k in out.index:
        seg = np.asarray(abp[int(k * window_s * fs): int((k + 1) * window_s * fs)], float)
        spec.append(spectral_features(seg, fs))
    spec_df = pd.DataFrame(spec, index=out.index)
    out = pd.concat([out, spec_df], axis=1)
    return out.reset_index(drop=True)[cols]


def fit_damping_classifier(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    """Gradient-boosted classifier from window features to flush-derived damping class."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
    return clf.fit(X[list(FEATURE_NAMES)], y)


def cv_auroc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 3, seed: int = 0) -> float:
    """Grouped cross-validated AUROC (binary y) so windows from one record never straddle folds."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
    p = cross_val_predict(clf, X[list(FEATURE_NAMES)], y, groups=groups, cv=GroupKFold(n_splits), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
