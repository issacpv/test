"""Stimulus-aligned spike binning and trial x unit response matrices."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .loaders import SessionData


def bin_spikes(spike_times: np.ndarray, t_starts: np.ndarray, window: Tuple[float, float],
               bin_size: float) -> np.ndarray:
    """Spike counts of one unit in bins aligned to each event in ``t_starts``.

    Returns an array of shape ``(n_events, n_bins)`` where
    ``n_bins = round((window[1] - window[0]) / bin_size)``.
    """
    st = np.asarray(spike_times, float)
    t0, t1 = window
    n_bins = int(round((t1 - t0) / bin_size))
    out = np.zeros((len(t_starts), n_bins), dtype=np.int32)
    if st.size == 0 or n_bins == 0:
        return out
    edges = np.arange(n_bins + 1) * bin_size + t0
    for i, ts in enumerate(np.asarray(t_starts, float)):
        lo = np.searchsorted(st, ts + t0, "left")
        hi = np.searchsorted(st, ts + t1, "left")
        if hi > lo:
            out[i] = np.histogram(st[lo:hi] - ts, edges)[0]
    return out


def response_matrix(sess: SessionData, table: Optional[str] = None, window: Optional[Tuple[float, float]] = None,
                    bin_size: Optional[float] = None, unit_ids: Optional[Sequence[int]] = None,
                    trial_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build a ``(n_trials, n_units, n_bins)`` count tensor and the matching trial table.

    Parameters
    ----------
    table : name in ``sess.stimulus['table']`` to use (default: all rows)
    window : (t0, t1) relative to ``start_time``; default (0, median trial duration)
    bin_size : default = whole window (one bin)
    """
    stim = sess.stimulus
    if table is not None:
        stim = stim[stim["table"] == table]
    if trial_mask is not None:
        stim = stim[np.asarray(trial_mask, bool)]
    stim = stim.reset_index(drop=True)
    if len(stim) == 0:
        raise ValueError("no trials selected")
    if window is None:
        dur = float(np.median(stim["stop_time"] - stim["start_time"]))
        window = (0.0, dur)
    bin_size = bin_size or (window[1] - window[0])
    ids = list(sess.spike_times) if unit_ids is None else list(unit_ids)
    starts = stim["start_time"].to_numpy(float)
    R = np.stack([bin_spikes(sess.spike_times[u], starts, window, bin_size) for u in ids], axis=1)
    return R, stim


def trial_responses(R: np.ndarray, method: str = "rate", bin_size: Optional[float] = None) -> np.ndarray:
    """Collapse the time axis: ``(n_trials, n_units, n_bins)`` -> ``(n_trials, n_units)``."""
    s = R.sum(axis=-1).astype(float)
    if method == "rate" and bin_size is not None:
        return s / (R.shape[-1] * bin_size)
    return s


def time_blocks(trial_times: np.ndarray, n_blocks: int = 3, by: str = "time") -> np.ndarray:
    """Assign trials to ``n_blocks`` contiguous blocks by absolute time (equal trial counts)."""
    t = np.asarray(trial_times, float)
    order = np.argsort(t, kind="stable")
    labels = np.empty(len(t), dtype=int)
    for b, idx in enumerate(np.array_split(order, n_blocks)):
        labels[idx] = b
    return labels


def condition_means(X: np.ndarray, cond: np.ndarray, block: np.ndarray,
                    conditions: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Condition-averaged responses per block: returns ``(M, conditions)`` with M ``(n_blocks, n_cond, n_units)``.

    Missing (block, condition) cells are NaN.
    """
    conds = np.unique(cond) if conditions is None else np.asarray(conditions)
    blocks = np.unique(block)
    M = np.full((len(blocks), len(conds), X.shape[1]), np.nan)
    for bi, b in enumerate(blocks):
        for ci, c in enumerate(conds):
            sel = (block == b) & (cond == c)
            if sel.any():
                M[bi, ci] = X[sel].mean(axis=0)
    return M, conds


def zscore_units(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd


def balance_trials(cond: np.ndarray, block: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Boolean mask selecting an equal number of trials per (block, condition) cell."""
    rng = np.random.default_rng(0) if rng is None else rng
    cells: Dict[Tuple[int, int], np.ndarray] = {}
    for b in np.unique(block):
        for c in np.unique(cond):
            idx = np.where((block == b) & (cond == c))[0]
            cells[(b, c)] = idx
    n_min = min(len(v) for v in cells.values())
    keep = np.zeros(len(cond), dtype=bool)
    for idx in cells.values():
        keep[rng.choice(idx, n_min, replace=False)] = True
    return keep
