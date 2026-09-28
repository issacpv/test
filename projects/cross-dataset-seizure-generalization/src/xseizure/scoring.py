"""Event-based seizure scoring: OVLP, TAES (NEDC-style) and SzCORE-style.

Events are ``(onset_s, offset_s)`` pairs. All scorers return a
:class:`ScoreResult` with counts and derived metrics (sensitivity, precision,
F1, false positives per 24 h).

References
----------
* Shah, Golmohammadi, Obeid & Picone (2021). Objective evaluation metrics for
  automatic classification of EEG events. In *Biomedical Signal Processing*.
  (defines OVLP and TAES; the TAES implemented here follows the published
  definition: fractional TP credit per reference event, FN = 1 - TP, and
  fractional FP for hypothesis time outside the reference.)
* Dan et al. (2024). SzCORE. *Epilepsia*. (event scoring with pre/post onset
  tolerance, minimum overlap, maximum event duration and merging of close events.)

The exact NEDC and SzCORE reference implementations have additional edge-case
handling; use them for official numbers and this module for fast, dependency-free
audits. Unit tests pin the behaviour on simple cases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Event = Tuple[float, float]


@dataclass
class ScoreResult:
    tp: float
    fp: float
    fn: float
    duration_s: float
    scorer: str

    @property
    def sensitivity(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else float("nan")

    @property
    def f1(self) -> float:
        s, p = self.sensitivity, self.precision
        if np.isnan(s) or np.isnan(p) or (s + p) == 0:
            return 0.0 if not (np.isnan(s) and np.isnan(p)) else float("nan")
        return 2 * s * p / (s + p)

    @property
    def fp_per_24h(self) -> float:
        return self.fp / max(self.duration_s, 1e-9) * 86400.0

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.update(sensitivity=self.sensitivity, precision=self.precision, f1=self.f1,
                 fp_per_24h=self.fp_per_24h)
        return d


# --------------------------------------------------------------------------- #
# event utilities
# --------------------------------------------------------------------------- #
def _clean(events: Sequence[Event]) -> List[Event]:
    ev = sorted((float(a), float(b)) for a, b in events if b > a)
    return ev


def merge_events(events: Sequence[Event], min_gap_s: float = 0.0) -> List[Event]:
    """Merge events separated by less than ``min_gap_s`` (and overlapping ones)."""
    ev = _clean(events)
    out: List[Event] = []
    for on, off in ev:
        if out and on - out[-1][1] < min_gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], off))
        else:
            out.append((on, off))
    return out


def split_long_events(events: Sequence[Event], max_dur_s: float) -> List[Event]:
    """Split events longer than ``max_dur_s`` into consecutive chunks of at most that length."""
    out: List[Event] = []
    for on, off in _clean(events):
        while off - on > max_dur_s:
            out.append((on, on + max_dur_s))
            on += max_dur_s
        out.append((on, off))
    return out


def events_from_mask(mask: np.ndarray, fs: float) -> List[Event]:
    """Convert a binary per-sample (or per-window) mask into events in seconds."""
    m = np.asarray(mask).astype(int)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    on = np.where(d == 1)[0]
    off = np.where(d == -1)[0]
    return [(o / fs, f / fs) for o, f in zip(on, off)]


def mask_from_events(events: Sequence[Event], duration_s: float, fs: float) -> np.ndarray:
    n = int(round(duration_s * fs))
    m = np.zeros(n, dtype=bool)
    for on, off in _clean(events):
        m[int(on * fs): int(np.ceil(off * fs))] = True
    return m


def _overlap(a: Event, b: Event) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def probabilities_to_events(prob: np.ndarray, step_s: float, win_s: float, threshold: float = 0.5,
                            min_duration_s: float = 5.0, merge_gap_s: float = 10.0) -> List[Event]:
    """Threshold per-window probabilities and post-process into hypothesis events.

    Window ``i`` covers ``[i*step_s, i*step_s + win_s)``. Consecutive positive
    windows are joined, events closer than ``merge_gap_s`` are merged, and
    events shorter than ``min_duration_s`` are removed.
    """
    p = np.asarray(prob, float)
    pos = p >= threshold
    raw: List[Event] = []
    for on_i, off_i in events_from_mask(pos, 1.0):
        raw.append((on_i * step_s, (off_i - 1) * step_s + win_s))
    merged = merge_events(raw, merge_gap_s)
    return [(a, b) for a, b in merged if b - a >= min_duration_s]


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #
def sample_score(ref_mask: np.ndarray, hyp_mask: np.ndarray, fs: float) -> Dict[str, float]:
    """Sample-based sensitivity/specificity/precision/F1 on aligned binary masks."""
    r = np.asarray(ref_mask, bool)
    h = np.asarray(hyp_mask, bool)
    if r.shape != h.shape:
        raise ValueError("masks must have the same shape")
    tp = float(np.sum(r & h)); fp = float(np.sum(~r & h))
    fn = float(np.sum(r & ~h)); tn = float(np.sum(~r & ~h))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "fp_per_24h": fp / fs / max(len(r) / fs, 1e-9) * 86400.0}


def ovlp_score(ref: Sequence[Event], hyp: Sequence[Event], duration_s: float) -> ScoreResult:
    """Any-overlap event scoring (NEDC OVLP).

    A reference event is a TP if any hypothesis event overlaps it (else FN);
    a hypothesis event that overlaps no reference event is an FP. Multiple
    hypotheses on one reference count once.
    """
    R, H = _clean(ref), _clean(hyp)
    tp = sum(1 for r in R if any(_overlap(r, h) > 0 for h in H))
    fn = len(R) - tp
    fp = sum(1 for h in H if not any(_overlap(r, h) > 0 for r in R))
    return ScoreResult(tp, fp, fn, duration_s, "ovlp")


def taes_score(ref: Sequence[Event], hyp: Sequence[Event], duration_s: float) -> ScoreResult:
    """Time-Aligned Event Scoring (TAES).

    For each reference event ``r``: TP credit = (total hypothesis time inside
    ``r``) / |r|, capped at 1; FN credit = 1 - TP credit. FP credit = hypothesis
    time *outside* ``r`` from hypotheses that overlap ``r``, divided by |r|
    (so over-extended detections are penalised proportionally). Hypothesis
    events overlapping no reference count as one FP each.
    """
    R, H = _clean(ref), _clean(hyp)
    tp = fn = fp = 0.0
    matched = np.zeros(len(H), dtype=bool)
    for r in R:
        rlen = r[1] - r[0]
        inside = 0.0
        outside = 0.0
        for j, h in enumerate(H):
            ov = _overlap(r, h)
            if ov > 0:
                matched[j] = True
                inside += ov
                outside += (h[1] - h[0]) - ov
        credit = min(inside / rlen, 1.0)
        tp += credit
        fn += 1.0 - credit
        fp += outside / rlen
    fp += float(np.sum(~matched))
    return ScoreResult(tp, fp, fn, duration_s, "taes")


def szcore_event_score(ref: Sequence[Event], hyp: Sequence[Event], duration_s: float,
                       tolerance_start_s: float = 30.0, tolerance_end_s: float = 60.0,
                       min_overlap: float = 0.0, max_event_duration_s: float = 300.0,
                       min_duration_between_events_s: float = 90.0) -> ScoreResult:
    """SzCORE-style event scoring (Dan et al., 2024).

    Steps: (1) merge reference and hypothesis events closer than
    ``min_duration_between_events_s``; (2) split events longer than
    ``max_event_duration_s``; (3) extend each reference event by the pre-onset /
    post-offset tolerances; (4) a reference is TP if some hypothesis overlaps its
    extended window by at least ``min_overlap`` of the reference duration; (5)
    a hypothesis is FP if it overlaps no extended reference.
    """
    R = split_long_events(merge_events(ref, min_duration_between_events_s), max_event_duration_s)
    H = split_long_events(merge_events(hyp, min_duration_between_events_s), max_event_duration_s)
    Rext = [(max(0.0, a - tolerance_start_s), min(duration_s, b + tolerance_end_s)) for a, b in R]
    tp = 0
    for (a, b), (ea, eb) in zip(R, Rext):
        ov = sum(_overlap((ea, eb), h) for h in H)
        need = min_overlap * (b - a)
        if ov > 0 and ov >= need:
            tp += 1
    fn = len(R) - tp
    fp = sum(1 for h in H if not any(_overlap(e, h) > 0 for e in Rext))
    return ScoreResult(tp, fp, fn, duration_s, "szcore")


def score_all(ref: Sequence[Event], hyp: Sequence[Event], duration_s: float) -> Dict[str, Dict[str, float]]:
    """Run OVLP, TAES and SzCORE scoring and return a nested dict of metrics."""
    return {
        "ovlp": ovlp_score(ref, hyp, duration_s).as_dict(),
        "taes": taes_score(ref, hyp, duration_s).as_dict(),
        "szcore": szcore_event_score(ref, hyp, duration_s).as_dict(),
    }


def pool_results(results: Sequence[ScoreResult]) -> ScoreResult:
    """Pool counts over records (micro-average) for one scorer."""
    if not results:
        raise ValueError("no results to pool")
    name = results[0].scorer
    return ScoreResult(sum(r.tp for r in results), sum(r.fp for r in results),
                       sum(r.fn for r in results), sum(r.duration_s for r in results), name)


def sensitivity_at_fp_rate(ref_by_rec: Dict[str, Sequence[Event]], prob_by_rec: Dict[str, np.ndarray],
                           duration_by_rec: Dict[str, float], step_s: float, win_s: float,
                           target_fp_per_24h: float = 1.0, thresholds: Sequence[float] = tuple(np.linspace(0.05, 0.95, 19)),
                           scorer=szcore_event_score, **post) -> Tuple[float, float]:
    """Sweep thresholds and return ``(sensitivity, threshold)`` at the highest threshold meeting the FP budget."""
    best = (float("nan"), float("nan"))
    for thr in sorted(thresholds):
        res = []
        for rid, p in prob_by_rec.items():
            hyp = probabilities_to_events(p, step_s, win_s, threshold=thr, **post)
            res.append(scorer(ref_by_rec[rid], hyp, duration_by_rec[rid]))
        pooled = pool_results(res)
        if pooled.fp_per_24h <= target_fp_per_24h:
            return pooled.sensitivity, float(thr)
    return best
