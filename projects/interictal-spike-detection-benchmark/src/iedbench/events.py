"""Event representation and event-level scoring for sub-second discharges.

An event is (channel, centre time in s, duration in s, score).  Matching is
greedy one-to-one: predictions are visited in descending score, each claims the
nearest unclaimed reference event on the same channel (or any channel when
``per_channel=False``) whose centre is within ``tol_s``.  Unclaimed predictions
are false positives, unclaimed references are misses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    channel: str
    t_s: float
    duration_s: float = 0.0
    score: float = 1.0
    label: str = "ied"


def events_from_mask(mask: np.ndarray, fs: float, channel_names: Sequence[str], scores: Optional[np.ndarray] = None,
                     min_dur_s: float = 0.0, max_dur_s: float = np.inf) -> List[Event]:
    """Convert a boolean (n_channels, n_samples) mask into events (one per contiguous run)."""
    mask = np.asarray(mask, bool)
    out: List[Event] = []
    for c in range(mask.shape[0]):
        m = np.r_[False, mask[c], False].astype(int)
        d = np.diff(m)
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            dur = (e - s) / fs
            if dur < min_dur_s or dur > max_dur_s:
                continue
            if scores is not None:
                seg = scores[c, s:e]
                peak = s + int(np.argmax(seg))
                sc = float(seg.max())
            else:
                peak, sc = s + (e - s) // 2, 1.0
            out.append(Event(str(channel_names[c]), peak / fs, dur, sc))
    return out


def match_events(pred: Sequence[Event], ref: Sequence[Event], tol_s: float = 0.1,
                 per_channel: bool = True) -> Dict[str, object]:
    """Greedy one-to-one matching within ``tol_s``. Returns counts, metrics and matched index pairs."""
    pred = list(pred)
    ref = list(ref)
    ref_t = np.asarray([r.t_s for r in ref], float)
    ref_c = np.asarray([r.channel for r in ref])
    claimed = np.zeros(len(ref), dtype=bool)
    pairs: List[Tuple[int, int]] = []
    order = np.argsort([-p.score for p in pred])
    for i in order:
        p = pred[i]
        if ref_t.size == 0:
            break
        ok = ~claimed & (np.abs(ref_t - p.t_s) <= tol_s)
        if per_channel:
            ok &= ref_c == p.channel
        idx = np.where(ok)[0]
        if idx.size:
            j = idx[np.argmin(np.abs(ref_t[idx] - p.t_s))]
            claimed[j] = True
            pairs.append((int(i), int(j)))
    tp = len(pairs)
    fp = len(pred) - tp
    fn = len(ref) - tp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp and tp + fn and prec + rec) else (0.0 if tp + fp + fn else float("nan"))
    matched_pred = {i for i, _ in pairs}
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, pairs=pairs,
                fp_indices=[i for i in range(len(pred)) if i not in matched_pred])


def fp_per_minute(n_fp: int, duration_s: float, n_channels: int = 1) -> float:
    """False positives per channel-minute."""
    return float(n_fp / (duration_s / 60.0) / n_channels) if duration_s > 0 else float("nan")


def threshold_sweep(pred: Sequence[Event], ref: Sequence[Event], duration_s: float, n_channels: int,
                    thresholds: Iterable[float], tol_s: float = 0.1, per_channel: bool = True) -> pd.DataFrame:
    """Precision / recall / F1 / FP-per-channel-minute at each score threshold."""
    rows = []
    for thr in thresholds:
        sub = [p for p in pred if p.score >= thr]
        m = match_events(sub, ref, tol_s, per_channel)
        rows.append(dict(threshold=float(thr), n_pred=len(sub), precision=m["precision"], recall=m["recall"],
                         f1=m["f1"], fp_per_min=fp_per_minute(int(m["fp"]), duration_s, n_channels)))
    return pd.DataFrame(rows)


def sensitivity_at_fp_rate(sweep: pd.DataFrame, fp_per_min_max: float) -> Tuple[float, float]:
    """Highest recall among operating points with FP/min <= ``fp_per_min_max`` and the threshold used."""
    ok = sweep[sweep["fp_per_min"] <= fp_per_min_max]
    if ok.empty:
        return float("nan"), float("nan")
    row = ok.loc[ok["recall"].idxmax()]
    return float(row["recall"]), float(row["threshold"])


def shuffle_reference_times(ref: Sequence[Event], duration_s: float, seed: int = 0) -> List[Event]:
    """Null reference: same channels and count, uniformly random times (chance level for F1)."""
    rng = np.random.default_rng(seed)
    return [Event(r.channel, float(rng.uniform(0, duration_s)), r.duration_s, r.score, r.label) for r in ref]


def events_to_frame(events: Sequence[Event]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in events]) if events else pd.DataFrame(
        columns=["channel", "t_s", "duration_s", "score", "label"])
