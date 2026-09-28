"""Morphology features of candidate discharges mapped to the six IFCN criteria (Kural et al., 2020).

Criterion -> feature
1. di-/triphasic sharp or spiky morphology       -> ``sharpness`` (normalized second derivative at the peak)
2. wave duration different from background       -> ``duration_ms`` (width at half prominence; spikes 20-70 ms,
                                                    sharp waves 70-200 ms) vs background dominant period
3. asymmetry of the waveform                     -> ``asymmetry`` (log rise-time / fall-time)
4. followed by a slow after-wave                 -> ``slow_wave`` (opposite-polarity area 50-400 ms after the peak,
                                                    normalized by prominence)
5. background activity disrupted                 -> ``bg_disruption`` (RMS 0.4-1.0 s after / 1.0 s before)
6. field over the scalp / neighbouring contacts  -> ``field_n`` (channels with a correlated deflection)

Amplitude relative to background (``rel_amplitude``) is kept as an extra feature.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURE_NAMES = ["duration_ms", "sharpness", "asymmetry", "rel_amplitude", "slow_wave", "bg_disruption",
                 "field_n", "field_maxcorr", "polarity"]


def _half_width(x: np.ndarray, i: int, base: float) -> Tuple[int, int]:
    """Indices where the deflection at ``i`` returns to half its height above ``base`` on each side."""
    h = x[i] - base
    half = base + 0.5 * h
    l = i
    while l > 0 and (x[l] - base) * np.sign(h) > (half - base) * np.sign(h):
        l -= 1
    r = i
    n = x.size
    while r < n - 1 and (x[r] - base) * np.sign(h) > (half - base) * np.sign(h):
        r += 1
    return l, r


def ifcn_features(x: np.ndarray, fs: float, idx: int) -> Dict[str, float]:
    """Single-channel morphology features for a candidate peak at sample ``idx``."""
    x = np.asarray(x, float)
    n = x.size
    idx = int(np.clip(idx, 1, n - 2))
    w_bg = int(1.0 * fs)
    excl = int(0.1 * fs)
    lo, hi = max(0, idx - w_bg), min(n, idx + w_bg)
    bg = np.r_[x[lo:max(lo, idx - excl)], x[min(hi, idx + excl):hi]]
    base = float(np.median(bg)) if bg.size else float(np.median(x))
    bg_rms = float(np.sqrt(np.mean((bg - base) ** 2))) + 1e-9 if bg.size else 1e-9
    prom = x[idx] - base
    pol = float(np.sign(prom)) if prom != 0 else 1.0
    l, r = _half_width(x, idx, base)
    duration_ms = 1000.0 * (r - l) / fs
    rise = max(idx - l, 1) / fs
    fall = max(r - idx, 1) / fs
    asym = float(np.log(rise / fall))
    d2 = abs(x[idx - 1] - 2 * x[idx] + x[idx + 1]) * fs ** 2
    sharp = float(np.log(d2 / (abs(prom) + 1e-9) + 1e-9))
    # slow after-wave: mean opposite-polarity deflection 50-400 ms after the peak, relative to the
    # spike prominence (an unclipped mean is unbiased under zero-mean background noise)
    a, b = idx + int(0.05 * fs), min(n, idx + int(0.4 * fs))
    seg = (x[a:b] - base) * (-pol) if b > a else np.array([0.0])
    slow = float(np.mean(seg) / (abs(prom) + 1e-9))
    # background disruption: RMS after (0.4-1.0 s) vs before (-1.0 .. -0.1 s)
    post = x[min(n, idx + int(0.4 * fs)):min(n, idx + int(1.0 * fs))]
    pre = x[max(0, idx - int(1.0 * fs)):max(0, idx - int(0.1 * fs))]
    rms_post = np.sqrt(np.mean((post - base) ** 2)) if post.size else np.nan
    rms_pre = np.sqrt(np.mean((pre - base) ** 2)) if pre.size else np.nan
    bg_dis = float(np.log((rms_post + 1e-9) / (rms_pre + 1e-9))) if np.isfinite(rms_post) and np.isfinite(rms_pre) else 0.0
    return dict(duration_ms=float(duration_ms), sharpness=sharp, asymmetry=asym,
                rel_amplitude=float(abs(prom) / bg_rms), slow_wave=slow, bg_disruption=bg_dis, polarity=pol)


def spatial_field(X: np.ndarray, fs: float, idx: int, ch: int, win_ms: float = 20.0, corr_thr: float = 0.5) -> Tuple[int, float]:
    """Number of *other* channels with a correlated deflection within +/- ``win_ms`` and the max |corr|."""
    X = np.asarray(X, float)
    w = max(2, int(win_ms / 1000.0 * fs))
    lo, hi = max(0, idx - w), min(X.shape[1], idx + w + 1)
    ref = X[ch, lo:hi] - X[ch, lo:hi].mean()
    if ref.std() == 0:
        return 0, 0.0
    n, mx = 0, 0.0
    for c in range(X.shape[0]):
        if c == ch:
            continue
        seg = X[c, lo:hi] - X[c, lo:hi].mean()
        if seg.std() == 0:
            continue
        r = abs(float(np.corrcoef(ref, seg)[0, 1]))
        mx = max(mx, r)
        if r >= corr_thr:
            n += 1
    return n, mx


def features_matrix(X: np.ndarray, fs: float, candidates: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Feature matrix (n_candidates, len(FEATURE_NAMES)) for (channel, sample) candidates on a (n_ch, n) array."""
    X = np.asarray(X, float)
    rows: List[List[float]] = []
    for ch, idx in candidates:
        f = ifcn_features(X[ch], fs, idx)
        fn, fmax = spatial_field(X, fs, idx, ch)
        f["field_n"], f["field_maxcorr"] = float(fn), float(fmax)
        rows.append([f[k] for k in FEATURE_NAMES])
    return np.asarray(rows, float).reshape(len(rows), len(FEATURE_NAMES))


def ifcn_criteria_flags(f: Dict[str, float], bg_dominant_period_ms: float = 100.0) -> Dict[str, bool]:
    """Binary criteria from a feature dict (thresholds are starting points, to be calibrated on annotated data)."""
    return {
        "c1_sharp": f["sharpness"] > np.log(2e4),                       # steep second derivative relative to height
        "c2_duration": 20.0 <= f["duration_ms"] <= 200.0 and abs(f["duration_ms"] - bg_dominant_period_ms) > 20.0,
        "c3_asymmetry": abs(f["asymmetry"]) > np.log(1.3),
        "c4_slow_wave": f["slow_wave"] > 0.05,
        "c5_bg_disruption": abs(f["bg_disruption"]) > np.log(1.3),
        "c6_field": f.get("field_n", 0) >= 1,
    }


def criteria_count(f: Dict[str, float]) -> int:
    return int(sum(ifcn_criteria_flags(f).values()))
