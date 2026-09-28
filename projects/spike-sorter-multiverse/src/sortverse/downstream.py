"""Downstream 'scientific claims' computed identically for every sorter output.

Inputs are spike times (s) per unit plus stimulus tables (onset times, conditions); outputs are per-unit
metrics and per-population summaries that the multiverse compares across specifications.
"""
from __future__ import annotations

from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def trial_counts(spike_times: np.ndarray, onsets: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Spike counts in [onset + w0, onset + w1) for every onset (searchsorted, O(n log n))."""
    t = np.sort(np.asarray(spike_times, dtype=float))
    on = np.asarray(onsets, dtype=float)
    lo = np.searchsorted(t, on + window[0])
    hi = np.searchsorted(t, on + window[1])
    return hi - lo


def trial_rates(spike_times: np.ndarray, onsets: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return trial_counts(spike_times, onsets, window) / (window[1] - window[0])


def responsiveness_permutation(spike_times: np.ndarray, onsets: np.ndarray, evoked: Tuple[float, float] = (0.0, 0.25),
                               baseline: Tuple[float, float] = (-0.25, 0.0), n_perm: int = 500,
                               rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Permutation test of mean evoked minus baseline rate, shuffling evoked/baseline labels within trials."""
    rng = np.random.default_rng(0) if rng is None else rng
    ev = trial_rates(spike_times, onsets, evoked)
    bl = trial_rates(spike_times, onsets, baseline)
    obs = float(np.mean(ev - bl))
    diffs = ev - bl
    null = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        null[i] = np.mean(diffs * signs)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"evoked_minus_baseline": obs, "p": p, "mean_evoked_rate": float(ev.mean()),
            "mean_baseline_rate": float(bl.mean())}


def tuning_curve(spike_times: np.ndarray, onsets: np.ndarray, conditions: np.ndarray,
                 window: Tuple[float, float]) -> pd.DataFrame:
    """Mean and SEM rate per condition."""
    r = trial_rates(spike_times, onsets, window)
    df = pd.DataFrame({"condition": np.asarray(conditions), "rate": r})
    g = df.groupby("condition")["rate"]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})


def orientation_selectivity(mean_rates: np.ndarray, directions_deg: np.ndarray) -> Dict[str, float]:
    """Vector-based OSI and DSI (Mazurek, Kager & Van Hooser, 2014) and the classic ratio OSI.

    OSI_vec = |sum r exp(2i theta)| / sum r ; DSI_vec = |sum r exp(i theta)| / sum r ;
    OSI_ratio = (r_pref - r_orth) / (r_pref + r_orth) with orthogonal = pref + 90 deg (mean of both directions).
    Negative rates are clipped at 0."""
    r = np.clip(np.asarray(mean_rates, dtype=float), 0, None)
    th = np.deg2rad(np.asarray(directions_deg, dtype=float))
    s = r.sum()
    if s <= 0:
        return {"osi_vec": np.nan, "dsi_vec": np.nan, "osi_ratio": np.nan, "pref_dir_deg": np.nan, "pref_ori_deg": np.nan}
    osi = float(np.abs(np.sum(r * np.exp(2j * th))) / s)
    dsi = float(np.abs(np.sum(r * np.exp(1j * th))) / s)
    pref_dir = float(np.rad2deg(np.angle(np.sum(r * np.exp(1j * th)))) % 360)
    pref_ori = float((np.rad2deg(np.angle(np.sum(r * np.exp(2j * th)))) / 2.0) % 180)
    # ratio OSI on the sampled directions closest to pref and pref+90
    dirs = np.rad2deg(th) % 360
    i_pref = int(np.argmin(np.abs(np.angle(np.exp(1j * np.deg2rad(dirs - pref_dir))))))
    orth = (dirs[i_pref] + 90.0) % 360
    i_orth = np.argsort(np.abs(np.angle(np.exp(1j * np.deg2rad(dirs - orth)))))[:2]
    r_orth = r[i_orth].mean()
    r_pref = r[i_pref]
    osi_ratio = float((r_pref - r_orth) / (r_pref + r_orth)) if (r_pref + r_orth) > 0 else np.nan
    return {"osi_vec": osi, "dsi_vec": dsi, "osi_ratio": osi_ratio, "pref_dir_deg": pref_dir, "pref_ori_deg": pref_ori}


def noise_correlations(counts: np.ndarray, conditions: np.ndarray) -> Dict[str, float]:
    """Mean pairwise Pearson correlation of trial counts after z-scoring within condition.

    ``counts`` is (n_trials, n_units)."""
    X = np.asarray(counts, dtype=float).copy()
    cond = np.asarray(conditions)
    for c in np.unique(cond):
        m = cond == c
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd == 0] = 1.0
        X[m] = (X[m] - mu) / sd
    n_units = X.shape[1]
    if n_units < 2:
        return {"mean_noise_corr": np.nan, "n_pairs": 0}
    C = np.corrcoef(X.T)
    iu = np.triu_indices(n_units, k=1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    return {"mean_noise_corr": float(vals.mean()) if len(vals) else np.nan, "n_pairs": int(len(vals))}


def population_response_matrix(sorting: Mapping[Hashable, np.ndarray], onsets: np.ndarray, conditions: np.ndarray,
                               window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """(n_units, n_conditions) mean-rate matrix and the sorted condition labels."""
    conds = np.unique(conditions)
    M = np.zeros((len(sorting), len(conds)))
    for i, (u, t) in enumerate(sorting.items()):
        r = trial_rates(t, onsets, window)
        for j, c in enumerate(conds):
            M[i, j] = r[conditions == c].mean()
    return M, conds


def representational_drift(resp_early: np.ndarray, resp_late: np.ndarray, within_early: Optional[Tuple[np.ndarray, np.ndarray]] = None
                           ) -> Dict[str, float]:
    """Population-vector and per-unit similarity between two blocks of (n_units, n_stimuli) responses.

    drift_index = 1 - mean over stimuli of corr(population vector early, late).
    If ``within_early`` = (half A, half B) of the early block is given, the *excess* drift beyond within-block
    reliability is also returned: within_similarity - between_similarity.
    """
    A = np.asarray(resp_early, dtype=float)
    B = np.asarray(resp_late, dtype=float)
    assert A.shape == B.shape
    def _pv_sim(X, Y):
        vals = []
        for j in range(X.shape[1]):
            if X[:, j].std() > 0 and Y[:, j].std() > 0:
                vals.append(np.corrcoef(X[:, j], Y[:, j])[0, 1])
        return float(np.mean(vals)) if vals else np.nan
    def _unit_sim(X, Y):
        vals = []
        for i in range(X.shape[0]):
            if X[i].std() > 0 and Y[i].std() > 0:
                vals.append(np.corrcoef(X[i], Y[i])[0, 1])
        return float(np.mean(vals)) if vals else np.nan
    between = _pv_sim(A, B)
    out = {"pv_similarity": between, "drift_index": 1.0 - between, "unit_similarity": _unit_sim(A, B),
           "rate_change": float(np.mean(np.abs(B.mean(axis=1) - A.mean(axis=1)) / (A.mean(axis=1) + B.mean(axis=1) + 1e-9)))}
    if within_early is not None:
        w = _pv_sim(*within_early)
        out["within_similarity"] = w
        out["excess_drift"] = w - between
    return out


def waveform_duration_ms(template: np.ndarray, fs: float) -> Dict[str, float]:
    """Trough-to-peak duration (ms) on the peak channel of an (n_samples, n_channels) mean waveform."""
    T = np.asarray(template, dtype=float)
    ch = int(np.argmax(np.ptp(T, axis=0)))
    w = T[:, ch]
    trough = int(np.argmin(w))
    peak = trough + int(np.argmax(w[trough:])) if trough < len(w) - 1 else trough
    dur = (peak - trough) / fs * 1e3
    return {"peak_channel": ch, "trough_to_peak_ms": float(dur), "amplitude": float(np.ptp(w))}


def narrow_spiking_fraction(durations_ms: Sequence[float], threshold_ms: float = 0.4) -> float:
    d = np.asarray(durations_ms, dtype=float)
    d = d[np.isfinite(d)]
    return float(np.mean(d < threshold_ms)) if len(d) else np.nan


def synthetic_tuned_unit(rng: np.random.Generator, onsets: np.ndarray, directions: np.ndarray, pref_deg: float,
                         peak_rate: float = 20.0, base_rate: float = 2.0, kappa: float = 2.0,
                         window: Tuple[float, float] = (0.0, 0.5)) -> np.ndarray:
    """Poisson spike times for a von-Mises-tuned unit (for tests and nulls)."""
    times = []
    dur = window[1] - window[0]
    for on, d in zip(onsets, directions):
        rate = base_rate + peak_rate * np.exp(kappa * (np.cos(np.deg2rad(d - pref_deg)) - 1))
        n = rng.poisson(rate * dur)
        times.append(on + window[0] + rng.uniform(0, dur, n))
        # baseline spikes before onset
        nb = rng.poisson(base_rate * 0.5)
        times.append(on - 0.5 + rng.uniform(0, 0.5, nb))
    return np.sort(np.concatenate(times))
