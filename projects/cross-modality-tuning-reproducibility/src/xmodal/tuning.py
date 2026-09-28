"""Tuning metrics computed identically for imaging and electrophysiology responses.

The common currency is a trial-level response per stimulus presentation (spike count / rate, mean dF/F, or
event amplitude) with its condition labels. Metrics follow the Allen Brain Observatory definitions where they
exist so that released `cell_specimens` / `analysis_metrics` tables can be reproduced.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def condition_means(trial_resp: np.ndarray, conditions: np.ndarray) -> pd.DataFrame:
    """Mean, SEM and n per condition label."""
    df = pd.DataFrame({"condition": np.asarray(conditions), "resp": np.asarray(trial_resp, dtype=float)})
    g = df.groupby("condition")["resp"]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})


def osi_dsi(mean_rates: np.ndarray, directions_deg: np.ndarray) -> Dict[str, float]:
    """Vector OSI/DSI (1 - circular variance; Ringach et al., 2002; Mazurek et al., 2014) and ratio OSI/DSI.

    ratio OSI = (r_pref - r_orth) / (r_pref + r_orth); ratio DSI = (r_pref - r_null) / (r_pref + r_null)."""
    r = np.clip(np.asarray(mean_rates, dtype=float), 0, None)
    d = np.asarray(directions_deg, dtype=float) % 360
    th = np.deg2rad(d)
    s = r.sum()
    if s <= 0 or len(r) < 2:
        return {k: np.nan for k in ("osi_vec", "dsi_vec", "osi_ratio", "dsi_ratio", "pref_dir_deg", "pref_ori_deg")}
    osi_vec = float(np.abs(np.sum(r * np.exp(2j * th))) / s)
    dsi_vec = float(np.abs(np.sum(r * np.exp(1j * th))) / s)
    i_pref = int(np.argmax(r))
    pref = d[i_pref]

    def _nearest(target: float) -> float:
        diffs = np.abs(np.angle(np.exp(1j * np.deg2rad(d - target))))
        return float(r[int(np.argmin(diffs))])

    r_pref = float(r[i_pref])
    r_null = _nearest(pref + 180)
    r_orth = 0.5 * (_nearest(pref + 90) + _nearest(pref - 90))
    osi_ratio = (r_pref - r_orth) / (r_pref + r_orth) if (r_pref + r_orth) > 0 else np.nan
    dsi_ratio = (r_pref - r_null) / (r_pref + r_null) if (r_pref + r_null) > 0 else np.nan
    return {"osi_vec": osi_vec, "dsi_vec": dsi_vec, "osi_ratio": float(osi_ratio), "dsi_ratio": float(dsi_ratio),
            "pref_dir_deg": float(pref), "pref_ori_deg": float(pref % 180)}


def preferred_condition(mean_rates: np.ndarray, levels: np.ndarray) -> object:
    """Level with the maximal mean response (e.g., preferred temporal frequency)."""
    return np.asarray(levels)[int(np.argmax(np.asarray(mean_rates, dtype=float)))]


def lifetime_sparseness(mean_responses: np.ndarray) -> float:
    """Vinje & Gallant (2000): S = (1 - (sum r / n)^2 / (sum r^2 / n)) / (1 - 1/n); 0 = flat, 1 = one-hot."""
    r = np.clip(np.asarray(mean_responses, dtype=float), 0, None)
    n = len(r)
    if n < 2 or np.sum(r ** 2) == 0:
        return np.nan
    return float((1.0 - (r.mean() ** 2) / np.mean(r ** 2)) / (1.0 - 1.0 / n))


def responsiveness_permutation(evoked: np.ndarray, baseline: np.ndarray, n_perm: int = 1000,
                               rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Paired permutation test of mean(evoked - baseline) with sign flips."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = np.asarray(evoked, dtype=float) - np.asarray(baseline, dtype=float)
    obs = float(d.mean())
    null = np.array([np.mean(d * rng.choice([-1.0, 1.0], size=len(d))) for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"effect": obs, "p": p}


def responsiveness_anova_like(trial_resp: np.ndarray, conditions: np.ndarray, blank_resp: np.ndarray,
                              n_perm: int = 1000, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Is any condition's mean response different from blank? Max-|t| permutation over shuffled labels."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(trial_resp, dtype=float)
    c = np.asarray(conditions)
    b = np.asarray(blank_resp, dtype=float)
    allx = np.concatenate([x, b])
    lab = np.concatenate([c.astype(object), np.array(["blank"] * len(b), dtype=object)])

    def _stat(vals: np.ndarray, labels: np.ndarray) -> float:
        bl = vals[labels == "blank"]
        best = 0.0
        for lv in pd.unique(labels[labels != "blank"]):
            v = vals[labels == lv]
            se = np.sqrt(v.var(ddof=1) / len(v) + bl.var(ddof=1) / len(bl)) if len(v) > 1 and len(bl) > 1 else np.inf
            best = max(best, abs(v.mean() - bl.mean()) / se if se > 0 else 0.0)
        return best

    obs = _stat(allx, lab)
    null = np.array([_stat(allx, rng.permutation(lab)) for _ in range(n_perm)])
    return {"max_t": float(obs), "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}


def tuning_metrics(trial_resp: np.ndarray, directions_deg: np.ndarray, secondary: Optional[np.ndarray] = None,
                   secondary_levels: Optional[Sequence] = None) -> Dict[str, float]:
    """OSI/DSI at the preferred secondary level (e.g., TF), preferred secondary level, lifetime sparseness.

    Mirrors the Allen drifting-gratings analysis: preferred (direction, TF) is the condition with the maximal
    mean response; OSI/DSI are computed across directions at the preferred TF; sparseness over all conditions."""
    resp = np.asarray(trial_resp, dtype=float)
    dirs = np.asarray(directions_deg, dtype=float)
    if secondary is None:
        cm = condition_means(resp, dirs)
        out = osi_dsi(cm["mean"].to_numpy(), cm.index.to_numpy())
        out["lifetime_sparseness"] = lifetime_sparseness(cm["mean"].to_numpy())
        out["pref_secondary"] = np.nan
        return out
    sec = np.asarray(secondary)
    levels = list(secondary_levels) if secondary_levels is not None else list(pd.unique(sec))
    grid = pd.DataFrame({"resp": resp, "dir": dirs, "sec": sec}).groupby(["sec", "dir"])["resp"].mean().unstack("dir")
    grid = grid.reindex(levels)
    i, j = np.unravel_index(np.nanargmax(grid.to_numpy()), grid.shape)
    pref_sec = levels[i]
    out = osi_dsi(grid.loc[pref_sec].to_numpy(), grid.columns.to_numpy())
    out["lifetime_sparseness"] = lifetime_sparseness(grid.to_numpy().ravel())
    out["pref_secondary"] = pref_sec
    return out


def rank_preference_agreement(pref_a: np.ndarray, pref_b: np.ndarray, levels: Sequence) -> float:
    """Fraction of matched cells whose preferred level agrees (rank-based, modality-invariant candidate)."""
    a = np.asarray(pref_a)
    b = np.asarray(pref_b)
    return float(np.mean(a == b))


def synthetic_tuned_responses(rng: np.random.Generator, directions: np.ndarray, pref_deg: float, peak: float,
                              base: float, kappa: float = 3.0, noise_sd: float = 1.0) -> np.ndarray:
    """Gaussian-noise trial responses of a von-Mises-tuned unit (for tests)."""
    mu = base + peak * np.exp(kappa * (np.cos(np.deg2rad(directions - pref_deg)) - 1))
    return mu + noise_sd * rng.normal(size=len(directions))
