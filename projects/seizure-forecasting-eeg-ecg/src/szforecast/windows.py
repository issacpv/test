"""Pre-ictal / inter-ictal window labelling, patient-wise splits and seizure-time surrogates.

Conventions (Maiwald et al., 2004; Winterhalder et al., 2003):

* SPH  seizure prediction horizon: minimum gap between an alarm and the onset.
* SOP  seizure occurrence period: the window after SPH during which the seizure
       is expected.  A window is *pre-ictal* if it lies in
       ``[onset - SPH - SOP, onset - SPH)``.
* Ictal windows and a post-ictal exclusion period are removed; windows closer
  than ``interictal_gap`` to any seizure (either side) but not pre-ictal are
  also removed (label -1), so inter-ictal windows are unambiguous.

Seizure-time surrogates (Andrzejak et al., 2003) keep the number of seizures
and the distribution of inter-seizure intervals within a record but destroy
their alignment with the signal, giving a null for any forecasting statistic.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

PREICTAL, INTERICTAL, EXCLUDED = 1, 0, -1


def lead_seizures(onsets: Sequence[float], offsets: Sequence[float], min_gap_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only *lead* seizures: those preceded by at least ``min_gap_s`` of seizure-free time."""
    on = np.asarray(onsets, float)
    off = np.asarray(offsets, float)
    order = np.argsort(on)
    on, off = on[order], off[order]
    keep = [0] if on.size else []
    for i in range(1, on.size):
        if on[i] - off[keep[-1]] >= min_gap_s:
            keep.append(i)
    return on[keep], off[keep]


def label_windows(window_starts: np.ndarray, window_len_s: float, onsets: Sequence[float],
                  offsets: Sequence[float], sph_s: float = 300.0, sop_s: float = 1800.0,
                  postictal_s: float = 3600.0, interictal_gap_s: float = 7200.0,
                  use_lead_only: bool = True) -> np.ndarray:
    """Assign PREICTAL (1) / INTERICTAL (0) / EXCLUDED (-1) to windows.

    A window counts as pre-ictal if its *end* falls inside the pre-ictal interval of
    a (lead) seizure and its start is not earlier than the interval start; windows
    overlapping the SPH gap, the seizure or the post-ictal period are excluded.
    """
    starts = np.asarray(window_starts, float)
    ends = starts + window_len_s
    labels = np.full(starts.size, INTERICTAL, dtype=int)
    on, off = (lead_seizures(onsets, offsets, sop_s + sph_s) if use_lead_only
               else (np.asarray(onsets, float), np.asarray(offsets, float)))
    for o, f in zip(on, off):
        pre_lo, pre_hi = o - sph_s - sop_s, o - sph_s
        excl_lo, excl_hi = o - sph_s, f + postictal_s
        # anything overlapping [SPH gap .. post-ictal] is excluded
        labels[(ends > excl_lo) & (starts < excl_hi)] = EXCLUDED
        # near-seizure but not pre-ictal: ambiguous -> excluded
        near = (ends > o - interictal_gap_s) & (starts < f + interictal_gap_s)
        labels[near & (labels != EXCLUDED)] = EXCLUDED
        # pre-ictal windows
        pre = (starts >= pre_lo) & (ends <= pre_hi)
        labels[pre] = PREICTAL
    # windows that are pre-ictal for one seizure but excluded by another keep EXCLUDED only if ictal overlap
    return labels


def seizure_surrogates(onsets: Sequence[float], offsets: Sequence[float], record_duration_s: float,
                       n: int = 200, seed: int = 0, min_gap_s: float = 60.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Constrained seizure-time surrogates for one record.

    Each surrogate permutes the inter-seizure intervals and applies a random
    circular shift, keeping durations, so that the number of seizures and the
    inter-seizure-interval distribution are preserved while the alignment with the
    signal is destroyed (Andrzejak et al., 2003, *Phys. Rev. E*).
    """
    rng = np.random.default_rng(seed)
    on = np.asarray(onsets, float)
    off = np.asarray(offsets, float)
    order = np.argsort(on)
    on, off = on[order], off[order]
    dur = off - on
    out = []
    if on.size == 0:
        return [(on.copy(), off.copy()) for _ in range(n)]
    gaps = np.diff(on) if on.size > 1 else np.array([])
    for _ in range(n):
        g = rng.permutation(gaps) if gaps.size else gaps
        new_on = np.concatenate([[0.0], np.cumsum(g)]) if gaps.size else np.array([0.0])
        shift = rng.uniform(0, record_duration_s)
        new_on = (new_on + shift) % record_duration_s
        d = rng.permutation(dur)
        idx = np.argsort(new_on)
        new_on, d = new_on[idx], d[idx]
        new_off = np.minimum(new_on + d, record_duration_s)
        # enforce a minimal gap to avoid overlapping surrogate events
        for i in range(1, new_on.size):
            if new_on[i] < new_off[i - 1] + min_gap_s:
                new_on[i] = min(new_off[i - 1] + min_gap_s, record_duration_s)
                new_off[i] = min(new_on[i] + d[i], record_duration_s)
        out.append((new_on, new_off))
    return out


def patient_folds(subject_ids: Sequence[str], n_folds: int = 5, seed: int = 0,
                  seizure_counts: Optional[Sequence[int]] = None) -> np.ndarray:
    """Fold index per window with all windows of a subject in the same fold.

    If ``seizure_counts`` (per window, e.g. 1 for pre-ictal windows) is given, subjects
    are assigned greedily to balance the pre-ictal count across folds.
    """
    subs = np.asarray(subject_ids)
    uniq = np.unique(subs)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {}
    if seizure_counts is None:
        for i, s in enumerate(uniq):
            fold_of[s] = i % n_folds
    else:
        cnt = np.asarray(seizure_counts, float)
        per_sub = {s: cnt[subs == s].sum() for s in uniq}
        load = np.zeros(n_folds)
        for s in sorted(uniq, key=lambda k: -per_sub[k]):
            f = int(np.argmin(load))
            fold_of[s] = f
            load[f] += per_sub[s]
    return np.asarray([fold_of[s] for s in subs], dtype=int)


def assert_no_subject_leakage(train_subjects: Iterable[str], test_subjects: Iterable[str]) -> None:
    """Raise if any subject appears in both sets."""
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"subject leakage between train and test: {sorted(overlap)[:5]} ...")


def clock_covariates(start_clock_s: float, window_starts: np.ndarray) -> np.ndarray:
    """Sine/cosine of the time of day for each window (columns: sin, cos), from the record start clock (s since midnight)."""
    tod = (start_clock_s + np.asarray(window_starts, float)) % 86400.0
    ang = 2 * np.pi * tod / 86400.0
    return np.column_stack([np.sin(ang), np.cos(ang)])
