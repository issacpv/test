"""Breath segmentation, per-breath features and rule-based asynchrony detectors on flow/pressure.

The rules follow the waveform criteria used in the clinical literature and in
automated systems such as Better Care (Blanch et al., Intensive Care Med 2012/2015):

* ineffective effort   - a positive deflection of expiratory flow (flow rises toward zero and
                         falls again) that does not trigger a breath;
* double triggering    - two consecutive delivered inspirations separated by an expiratory
                         time shorter than half the median inspiratory time;
* asynchrony index     - (ineffective efforts + asynchronous breaths) / (breaths + ineffective efforts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def segment_breaths(flow: np.ndarray, fs: float, insp_threshold: float = 0.05, min_ti_s: float = 0.15,
                    min_te_s: float = 0.1) -> pd.DataFrame:
    """Delivered breaths from the flow signal: ``start_idx, insp_end_idx, end_idx``.

    Inspiration = flow above ``insp_threshold`` (L/s) for at least ``min_ti_s``; it ends when flow
    drops below the threshold. A breath ends at the next inspiration start.
    """
    above = np.asarray(flow) > insp_threshold
    edges = np.diff(above.astype(int), prepend=0, append=0)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= int(min_ti_s * fs)]
    # merge inspirations separated by a very short dip (flow noise), not a real expiration
    merged: list[tuple[int, int]] = []
    for s, e in keep:
        if merged and s - merged[-1][1] < int(min_te_s * fs):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    rows = [{"start_idx": s, "insp_end_idx": e, "end_idx": merged[k + 1][0] if k + 1 < len(merged) else len(flow)}
            for k, (s, e) in enumerate(merged)]
    return pd.DataFrame(rows, columns=["start_idx", "insp_end_idx", "end_idx"])


def breath_features(flow: np.ndarray, paw: np.ndarray, fs: float, breaths: pd.DataFrame) -> pd.DataFrame:
    """Per-breath ``ti, te, vt_insp, vt_exp, pip, peep, rr_inst, t_start, n_exp_notches``."""
    dt = 1.0 / fs
    rows = []
    for _, b in breaths.iterrows():
        s, ie, e = int(b["start_idx"]), int(b["insp_end_idx"]), int(b["end_idx"])
        q_in, q_ex = flow[s:ie], flow[ie:e]
        rows.append({
            "t_start": s / fs, "ti": (ie - s) * dt, "te": (e - ie) * dt,
            "vt_insp": float(np.sum(np.clip(q_in, 0, None)) * dt),
            "vt_exp": float(-np.sum(np.clip(q_ex, None, 0)) * dt),
            "pip": float(paw[s:ie].max()) if ie > s else np.nan,
            "peep": float(np.median(paw[max(ie, e - int(0.1 * fs)):e])) if e > ie else np.nan,
            "rr_inst": 60.0 / ((e - s) * dt) if e > s else np.nan,
            "n_exp_notches": int(len(_exp_notches(q_ex, fs)[0])),
        })
    return pd.DataFrame(rows)


def _exp_notches(q_ex: np.ndarray, fs: float, min_rise_lps: float = 0.04, min_gap_s: float = 0.15) -> tuple[np.ndarray, dict]:
    """Local maxima of expiratory flow (positive deflections toward zero) with prominence >= min_rise."""
    if q_ex.size < int(0.2 * fs):
        return np.array([], dtype=int), {}
    seg = q_ex[int(0.05 * fs):]  # skip the cycling transient
    pk, props = find_peaks(seg, prominence=min_rise_lps, distance=max(1, int(min_gap_s * fs)))
    return pk + int(0.05 * fs), props


def detect_ineffective_efforts(flow: np.ndarray, fs: float, breaths: pd.DataFrame,
                               min_rise_lps: float = 0.04) -> pd.DataFrame:
    """Ineffective efforts as expiratory-flow notches; returns ``idx, breath_row, prominence``."""
    rows = []
    for k, b in breaths.iterrows():
        ie, e = int(b["insp_end_idx"]), int(b["end_idx"])
        pk, props = _exp_notches(flow[ie:e], fs, min_rise_lps)
        for j, p in enumerate(pk):
            rows.append({"idx": ie + int(p), "breath_row": int(k), "prominence": float(props["prominences"][j])})
    return pd.DataFrame(rows, columns=["idx", "breath_row", "prominence"])


def detect_double_triggering(features: pd.DataFrame, te_factor: float = 0.5, max_te_s: float | None = None) -> np.ndarray:
    """Rows (of ``features``) that are the *second* breath of a double-trigger pair."""
    if features.empty:
        return np.array([], dtype=int)
    thr = te_factor * float(np.median(features["ti"]))
    if max_te_s is not None:
        thr = min(thr, max_te_s)
    te_prev = features["te"].shift(1)
    return np.flatnonzero((te_prev < thr).to_numpy())


def asynchrony_index(n_breaths: int, n_ineffective: int, n_async_breaths: int) -> float:
    denom = n_breaths + n_ineffective
    return 100.0 * (n_ineffective + n_async_breaths) / denom if denom else float("nan")


def match_events(detected_idx: np.ndarray, true_idx: np.ndarray, fs: float, tol_s: float = 0.5) -> dict[str, float]:
    """Recall / precision of detected event indices against ground truth within +-tol_s."""
    detected_idx, true_idx = np.asarray(detected_idx, int), np.asarray(true_idx, int)
    tol = int(tol_s * fs)
    tp_true = sum(np.any(np.abs(detected_idx - t) <= tol) for t in true_idx) if detected_idx.size else 0
    tp_det = sum(np.any(np.abs(true_idx - d) <= tol) for d in detected_idx) if true_idx.size else 0
    recall = tp_true / len(true_idx) if len(true_idx) else float("nan")
    precision = tp_det / len(detected_idx) if len(detected_idx) else float("nan")
    return {"recall": float(recall), "precision": float(precision), "n_true": int(len(true_idx)), "n_detected": int(len(detected_idx))}
