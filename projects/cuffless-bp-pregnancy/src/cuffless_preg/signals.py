"""Beat-level signal processing for ECG / PPG / ABP records.

All functions operate on 1-D numpy arrays sampled at ``fs`` Hz and return
sample indices or per-beat tables.  They are deliberately simple (no learned
components) so that they behave identically on MIMIC (125 Hz) and VitalDB
(500 Hz ECG resampled, 100 Hz PLETH/ART) after resampling to a common rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal as sps


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(hi, 0.95 * nyq)
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def _lowpass(x: np.ndarray, fs: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = sps.butter(order, min(hi, 0.95 * nyq) / nyq, btype="low")
    return sps.filtfilt(b, a, x)


def detect_r_peaks(ecg: np.ndarray, fs: float, min_rr_s: float = 0.3) -> np.ndarray:
    """Pan-Tompkins-style R-peak detector.

    Band-pass 5-20 Hz, derivative, squaring, moving-window integration
    (150 ms), then peak picking with an adaptive threshold (40% of the rolling
    95th percentile of the integrated signal).  Peaks are refined to the local
    maximum of the band-passed ECG within +-50 ms.

    Returns sample indices of R peaks.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.size < int(2 * fs):
        return np.array([], dtype=int)
    filt = _bandpass(ecg - np.nanmean(ecg), fs, 5.0, 20.0)
    der = np.gradient(filt)
    sq = der**2
    win = max(1, int(0.15 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = 0.4 * np.percentile(integ, 95)
    peaks, _ = sps.find_peaks(integ, height=thr, distance=int(min_rr_s * fs))
    half = int(0.05 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(filt), p + half + 1)
        refined.append(lo + int(np.argmax(filt[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


@dataclass
class PulseFiducials:
    """Per-beat fiducial indices for a pulsatile signal (PPG or ABP)."""

    feet: np.ndarray
    peaks: np.ndarray
    max_slope: np.ndarray


def detect_pulse_fiducials(
    x: np.ndarray, fs: float, min_rr_s: float = 0.3, lp_hz: float = 8.0, lp_hz_foot: float = 20.0
) -> PulseFiducials:
    """Find foot (onset), systolic peak and maximum-upslope point of every pulse.

    Peaks are found on a heavily low-passed copy (``lp_hz``, robust peak
    picking); the foot is the minimum in the 0.4 s window preceding each peak
    located on a lightly low-passed copy (``lp_hz_foot``) so that the onset is
    not smeared backwards by the filter; the max-slope point is the maximum of
    the first derivative between foot and peak.  Only beats with all three
    fiducials are returned, aligned element-wise.
    """
    x = np.asarray(x, dtype=float)
    if x.size < int(2 * fs):
        e = np.array([], dtype=int)
        return PulseFiducials(e, e, e)
    xc = x - np.nanmean(x)
    xf = _lowpass(xc, fs, lp_hz)
    xl = _lowpass(xc, fs, lp_hz_foot)
    prom = 0.3 * np.nanstd(xf)
    peaks, _ = sps.find_peaks(xf, distance=int(min_rr_s * fs), prominence=prom)
    dx = np.gradient(xl)
    back = int(0.4 * fs)
    feet, pks, slopes = [], [], []
    for p in peaks:
        lo = max(0, p - back)
        if p - lo < 3:
            continue
        f = lo + int(np.argmin(xl[lo:p]))
        if p - f < 2:
            continue
        s = f + int(np.argmax(dx[f:p]))
        feet.append(f)
        pks.append(p)
        slopes.append(s)
    return PulseFiducials(np.asarray(feet, int), np.asarray(pks, int), np.asarray(slopes, int))


def pulse_arrival_times(
    r_idx: np.ndarray, foot_idx: np.ndarray, fs: float, min_pat_s: float = 0.10, max_pat_s: float = 0.60
) -> pd.DataFrame:
    """Pair each R peak with the first pulse foot inside [min_pat, max_pat].

    Returns a DataFrame with columns ``r_idx``, ``foot_idx``, ``t_s`` (time of
    the R peak) and ``pat_s``.  Beats with no foot in the window are dropped.
    """
    r_idx = np.asarray(r_idx, int)
    foot_idx = np.asarray(foot_idx, int)
    rows = []
    j = 0
    for r in r_idx:
        while j < len(foot_idx) and foot_idx[j] < r + int(min_pat_s * fs):
            j += 1
        if j >= len(foot_idx):
            break
        d = (foot_idx[j] - r) / fs
        if d <= max_pat_s:
            rows.append((int(r), int(foot_idx[j]), r / fs, float(d)))
    return pd.DataFrame(rows, columns=["r_idx", "foot_idx", "t_s", "pat_s"])


def abp_beat_table(abp: np.ndarray, fs: float) -> pd.DataFrame:
    """Beat-wise SBP / DBP / MAP from an invasive ABP waveform (mmHg).

    DBP is the foot value, SBP the systolic peak value, MAP the mean of the
    waveform between consecutive feet.  Beats with SBP-DBP < 15 mmHg or
    SBP > 300 mmHg are flagged ``valid=False`` (damping / artefact screen).
    """
    abp = np.asarray(abp, dtype=float)
    fid = detect_pulse_fiducials(abp, fs)
    rows = []
    for i, (f, p) in enumerate(zip(fid.feet, fid.peaks)):
        nxt = fid.feet[i + 1] if i + 1 < len(fid.feet) else min(len(abp), f + int(1.5 * fs))
        seg = abp[f:nxt]
        sbp, dbp = float(abp[p]), float(abp[f])
        rows.append(
            {
                "foot_idx": int(f),
                "t_s": f / fs,
                "sbp": sbp,
                "dbp": dbp,
                "map": float(np.mean(seg)) if seg.size else np.nan,
                "valid": (15.0 <= sbp - dbp) and (sbp <= 300.0) and (dbp >= 20.0),
            }
        )
    return pd.DataFrame(rows)


def template_sqi(x: np.ndarray, feet: np.ndarray, fs: float, beat_len_s: float = 0.8) -> np.ndarray:
    """Per-beat signal quality: Pearson correlation of each beat with the mean beat template.

    Beats are resampled to a fixed length starting at the foot.  Returns an
    array aligned with ``feet`` (NaN where the beat runs off the record).
    """
    x = np.asarray(x, dtype=float)
    n = int(beat_len_s * fs)
    beats = []
    for f in feet:
        seg = x[f : f + n]
        if seg.size == n:
            seg = (seg - seg.mean()) / (seg.std() + 1e-9)
            beats.append(seg)
        else:
            beats.append(None)
    valid = [b for b in beats if b is not None]
    if len(valid) < 3:
        return np.full(len(feet), np.nan)
    template = np.mean(np.vstack(valid), axis=0)
    out = np.full(len(feet), np.nan)
    for i, b in enumerate(beats):
        if b is not None:
            out[i] = float(np.corrcoef(b, template)[0, 1])
    return out


def ppg_morphology(ppg: np.ndarray, fid: PulseFiducials, fs: float) -> pd.DataFrame:
    """Simple PPG morphology features per beat: amplitude, rise time, width at half height."""
    ppg = np.asarray(ppg, dtype=float)
    rows = []
    for f, p in zip(fid.feet, fid.peaks):
        amp = ppg[p] - ppg[f]
        rise = (p - f) / fs
        half = ppg[f] + 0.5 * amp
        # width at half height: from first crossing before peak to first crossing after
        left = p
        while left > f and ppg[left] > half:
            left -= 1
        right = p
        lim = min(len(ppg) - 1, p + int(1.0 * fs))
        while right < lim and ppg[right] > half:
            right += 1
        rows.append({"foot_idx": int(f), "amp": float(amp), "rise_s": rise, "width50_s": (right - left) / fs})
    return pd.DataFrame(rows)


def build_beat_table(
    ecg: np.ndarray, ppg: np.ndarray, abp: np.ndarray | None, fs: float, sqi_min: float = 0.8
) -> pd.DataFrame:
    """End-to-end beat table for one record segment.

    Columns: t_s, pat_s (R -> PPG foot), hr_bpm, sqi, amp, rise_s, width50_s and,
    when ABP is given, sbp/dbp/map matched to the nearest ABP beat within 0.5 s
    and pat_prox_s (R -> ABP foot, proximal component).
    """
    r = detect_r_peaks(ecg, fs)
    pf = detect_pulse_fiducials(ppg, fs)
    pat = pulse_arrival_times(r, pf.feet, fs)
    if pat.empty:
        return pat
    rr = np.diff(r) / fs
    hr = pd.Series(60.0 / rr, index=r[1:])
    pat["hr_bpm"] = pat["r_idx"].map(hr)
    sqi = pd.Series(template_sqi(ppg, pf.feet, fs), index=pf.feet)
    pat["sqi"] = pat["foot_idx"].map(sqi)
    morph = ppg_morphology(ppg, pf, fs).set_index("foot_idx")
    pat = pat.join(morph, on="foot_idx")
    if abp is not None:
        ab = abp_beat_table(abp, fs)
        ab = ab[ab["valid"]].drop(columns="valid")
        pat = pd.merge_asof(
            pat.sort_values("t_s"),
            ab.sort_values("t_s").rename(columns={"t_s": "t_abp_s", "foot_idx": "abp_foot_idx"}),
            left_on="t_s",
            right_on="t_abp_s",
            direction="forward",
            tolerance=0.5,
        )
        pat["pat_prox_s"] = (pat["abp_foot_idx"] - pat["r_idx"]) / fs
    pat["good"] = pat["sqi"].ge(sqi_min).fillna(False)
    return pat.reset_index(drop=True)
