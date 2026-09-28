"""Split regimes for the leakage audit.

Four regimes are compared on the *same* windows:

* window-wise  : random windows (worst case; adjacent windows overlap in time)
* record-wise  : whole records are assigned to folds but a subject's records may
                 straddle folds (identity leakage)
* patient-wise : all records of a subject in one fold (deployment-realistic)
* leave-one-patient-out

All splitters return lists of ``(train_idx, test_idx)`` index arrays into the
window table, so they are interchangeable in an evaluation loop.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Fold = Tuple[np.ndarray, np.ndarray]


def _as_array(x: Sequence) -> np.ndarray:
    return np.asarray(list(x))


def window_wise_split(n_windows: int, n_folds: int = 5, seed: int = 0) -> List[Fold]:
    """Random assignment of individual windows to folds (maximally leaky)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows)
    folds = np.array_split(perm, n_folds)
    return [(np.setdiff1d(perm, f), np.sort(f)) for f in folds]


def _balanced_group_assignment(groups: np.ndarray, weights: np.ndarray, n_folds: int,
                               seed: int) -> Dict[str, int]:
    """Greedy assignment of groups to folds balancing total weight (e.g. seizure count).

    Groups are visited in decreasing weight with random tie-breaking and each is
    placed in the currently lightest fold.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    w = np.array([weights[groups == g].sum() for g in uniq], dtype=float)
    order = np.lexsort((rng.random(len(uniq)), -w))
    load = np.zeros(n_folds)
    assign: Dict[str, int] = {}
    for i in order:
        f = int(np.argmin(load))
        assign[str(uniq[i])] = f
        load[f] += w[i] + 1e-9
    return assign


def record_wise_split(record_ids: Sequence, n_folds: int = 5, seed: int = 0,
                      weights: Optional[Sequence[float]] = None) -> List[Fold]:
    """Assign whole records to folds ignoring subject identity (leaky by design)."""
    rec = _as_array(record_ids).astype(str)
    w = np.ones(len(rec)) if weights is None else np.asarray(weights, float)
    assign = _balanced_group_assignment(rec, w, n_folds, seed)
    fold_of = np.array([assign[r] for r in rec])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_folds)]


def patient_wise_split(subject_ids: Sequence, n_folds: int = 5, seed: int = 0,
                       weights: Optional[Sequence[float]] = None) -> List[Fold]:
    """Assign whole subjects to folds, balancing total ``weights`` (e.g. seizure windows).

    Parameters
    ----------
    subject_ids : per-window subject identifier
    n_folds : number of folds (must be <= number of unique subjects)
    weights : per-window weight; pass the seizure label to balance positives
    """
    subj = _as_array(subject_ids).astype(str)
    n_subj = len(np.unique(subj))
    if n_folds > n_subj:
        raise ValueError(f"n_folds={n_folds} exceeds number of subjects ({n_subj})")
    w = np.ones(len(subj)) if weights is None else np.asarray(weights, float)
    assign = _balanced_group_assignment(subj, w, n_folds, seed)
    fold_of = np.array([assign[s] for s in subj])
    folds = [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_folds)]
    for tr, te in folds:
        assert_no_subject_leakage(subj, tr, te)
    return folds


def leave_one_patient_out(subject_ids: Sequence) -> List[Fold]:
    """One fold per subject."""
    subj = _as_array(subject_ids).astype(str)
    return [(np.where(subj != s)[0], np.where(subj == s)[0]) for s in np.unique(subj)]


def assert_no_subject_leakage(subject_ids: Sequence, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise ``AssertionError`` if any subject appears in both train and test."""
    subj = _as_array(subject_ids).astype(str)
    overlap = np.intersect1d(np.unique(subj[train_idx]), np.unique(subj[test_idx]))
    if overlap.size:
        raise AssertionError(f"subject leakage: {overlap.tolist()}")


def subject_overlap_fraction(subject_ids: Sequence, folds: List[Fold]) -> float:
    """Fraction of test windows whose subject also appears in the training set (0 = clean)."""
    subj = _as_array(subject_ids).astype(str)
    leaky, total = 0, 0
    for tr, te in folds:
        train_subj = set(subj[tr].tolist())
        leaky += int(np.sum([s in train_subj for s in subj[te]]))
        total += len(te)
    return leaky / max(total, 1)


def leakage_inflation(metric_leaky: Sequence[float], metric_clean: Sequence[float],
                      n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Absolute and relative inflation of a metric under a leaky split vs a clean split.

    Both inputs are per-fold (or per-subject) metric values; the inflation is the
    difference of means with a bootstrap percentile CI. Returns a dict with keys
    ``abs``, ``rel``, ``ci_low``, ``ci_high``.
    """
    a = np.asarray(metric_leaky, float)
    b = np.asarray(metric_clean, float)
    rng = np.random.default_rng(seed)
    diff = a.mean() - b.mean()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"abs": float(diff), "rel": float(diff / max(abs(b.mean()), 1e-12)),
            "ci_low": float(lo), "ci_high": float(hi)}
