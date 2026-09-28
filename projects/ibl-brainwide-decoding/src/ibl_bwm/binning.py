"""Trial-aligned binned spike tensors and decoding targets.

`bin_spikes` produces a (n_trials, n_units, n_bins) count tensor for a set of units aligned to event
times; `window_features` collapses it to (n_trials, n_units) for a decoder. `targets_from_trials`
derives the four benchmark targets (choice, stimulus side, block prior, reward) with IBL conventions.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .loaders import SessionData

# Default alignment windows (event column, pre, post) per target, following the BWM decoding analyses:
# stimulus: 0–100 ms after stimulus onset; choice: ±100 ms around first movement; block prior: the 400 ms
# before stimulus onset; reward/feedback: 0–200 ms after feedback.
ALIGNMENTS: Dict[str, Tuple[str, float, float]] = {
    "stimulus": ("stimOn_times", 0.0, 0.10),
    "choice": ("firstMovement_times", -0.10, 0.10),
    "block": ("stimOn_times", -0.40, 0.0),
    "reward": ("feedback_times", 0.0, 0.20),
}


def bin_spikes(spike_times: np.ndarray, spike_clusters: np.ndarray, unit_ids: Sequence[int],
               align_times: np.ndarray, pre: float, post: float, bin_size: float = 0.02) -> np.ndarray:
    """Count spikes of `unit_ids` in bins of `bin_size` s from `pre` to `post` around each event.

    `spike_times` must be sorted ascending. Trials with NaN event time get all-zero rows (mask them
    out with `np.isnan(align_times)` downstream).
    Returns an int32 tensor (n_trials, n_units, n_bins).
    """
    unit_ids = np.asarray(unit_ids)
    unit_pos = {u: i for i, u in enumerate(unit_ids)}
    n_bins = int(np.round((post - pre) / bin_size))
    out = np.zeros((len(align_times), len(unit_ids), n_bins), dtype=np.int32)
    # spike -> unit index (-1 for spikes of units we do not want)
    uidx = np.full(len(spike_clusters), -1, dtype=np.int64)
    if len(unit_ids):
        lut_keys, lut_vals = np.array(list(unit_pos.keys())), np.array(list(unit_pos.values()))
        order = np.argsort(lut_keys)
        srt = lut_keys[order]
        pos = np.searchsorted(srt, spike_clusters)
        pos[pos >= len(srt)] = 0
        hit = srt[pos] == spike_clusters
        uidx[hit] = lut_vals[order][pos[hit]]
    for k, t0 in enumerate(np.asarray(align_times, dtype=float)):
        if np.isnan(t0):
            continue
        lo = np.searchsorted(spike_times, t0 + pre, side="left")
        hi = np.searchsorted(spike_times, t0 + post, side="left")
        if hi <= lo:
            continue
        u = uidx[lo:hi]
        b = ((spike_times[lo:hi] - (t0 + pre)) / bin_size).astype(np.int64)
        ok = (u >= 0) & (b >= 0) & (b < n_bins)
        np.add.at(out[k], (u[ok], b[ok]), 1)
    return out


def window_features(tensor: np.ndarray, sqrt: bool = True, per_bin: bool = False) -> np.ndarray:
    """(n_trials, n_units, n_bins) -> (n_trials, n_units) summed counts, or flattened per-bin features."""
    X = tensor.reshape(tensor.shape[0], -1) if per_bin else tensor.sum(axis=2)
    X = X.astype(float)
    return np.sqrt(X) if sqrt else X


def region_features(session: SessionData, region: str, target: str, bin_size: float = 0.02,
                    good_only: bool = True, alignments: Mapping[str, Tuple[str, float, float]] = ALIGNMENTS,
                    sqrt: bool = True, per_bin: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix for one (session, region, target) and the boolean mask of usable trials."""
    event, pre, post = alignments[target]
    units = session.units_in_region(region, good_only=good_only)
    t = session.trials[event].to_numpy(dtype=float)
    tensor = bin_spikes(session.spike_times, session.spike_clusters, units, t, pre, post, bin_size)
    X = window_features(tensor, sqrt=sqrt, per_bin=per_bin)
    valid = ~np.isnan(t)
    return X, valid


def targets_from_trials(trials: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Decoding targets with NaN where undefined.

    choice     : 1 = left choice (+1), 0 = right (−1); no-go (0) -> NaN
    stimulus   : 1 = stimulus on the left (contrastLeft not NaN), 0 = right; zero-contrast -> NaN
    contrast   : signed contrast (+left / −right), continuous
    block      : probabilityLeft (0.2 / 0.5 / 0.8) continuous; `block_bin` = 1 if 0.8, 0 if 0.2, NaN if 0.5
    reward     : 1 = rewarded (feedbackType +1), 0 = error
    """
    t = trials
    out: Dict[str, np.ndarray] = {}
    if "choice" in t:
        ch = t["choice"].to_numpy(dtype=float)
        out["choice"] = np.where(ch == 0, np.nan, (ch > 0).astype(float))
    if {"contrastLeft", "contrastRight"} <= set(t.columns):
        cl, cr = t["contrastLeft"].to_numpy(dtype=float), t["contrastRight"].to_numpy(dtype=float)
        left = ~np.isnan(cl)
        signed = np.where(left, np.nan_to_num(cl), -np.nan_to_num(cr))
        out["contrast"] = signed
        out["stimulus"] = np.where(signed == 0, np.nan, (signed > 0).astype(float))
    if "probabilityLeft" in t:
        p = t["probabilityLeft"].to_numpy(dtype=float)
        out["block"] = p
        out["block_bin"] = np.where(p == 0.5, np.nan, (p > 0.5).astype(float))
    if "feedbackType" in t:
        out["reward"] = (t["feedbackType"].to_numpy(dtype=float) > 0).astype(float)
    return out


def usable(X: np.ndarray, y: np.ndarray, valid: Optional[np.ndarray] = None,
           min_trials: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Drop trials with NaN target or invalid alignment; raise if too few remain."""
    m = ~np.isnan(y)
    if valid is not None:
        m &= valid
    if m.sum() < min_trials:
        raise ValueError(f"only {int(m.sum())} usable trials (< {min_trials})")
    return X[m], y[m]


def unit_yield_table(session: SessionData, min_units: int = 1) -> pd.DataFrame:
    """Per-region unit counts (all / good) for the yield-vs-decoding analysis."""
    regs = pd.Series(session.cluster_regions)
    tab = pd.DataFrame({"n_units": regs.value_counts(), "n_good": regs[session.cluster_good].value_counts()}).fillna(0)
    tab["eid"], tab["pid"], tab["lab"], tab["subject"] = session.eid, session.pid, session.lab, session.subject
    return tab[tab["n_units"] >= min_units].astype({"n_units": int, "n_good": int})
