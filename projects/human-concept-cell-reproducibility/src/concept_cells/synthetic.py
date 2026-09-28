"""Synthetic single-unit sessions with known selective units, drift and blocked stimuli.

Used (a) by the tests and (b) to calibrate the empirical false-positive rate of every
criterion x null combination before real data are analysed.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .nwb_loader import SessionData


def make_synthetic_session(n_units: int = 40, n_stim: int = 5, trials_per_stim: int = 12, frac_selective: float = 0.25,
                           base_rate: float = 4.0, gain: float = 3.0, drift_amp: float = 0.0, blocked: bool = False,
                           iti: float = 2.5, window: float = 1.0, seed: int = 0) -> Dict[str, object]:
    """Create a :class:`SessionData` with ground-truth labels.

    Parameters
    ----------
    frac_selective : fraction of units whose rate is multiplied by ``gain`` for one stimulus.
    drift_amp : amplitude of a slow multiplicative rate drift (0 = stationary); the drift is a
        random smooth curve in [1 - drift_amp, 1 + drift_amp] over the session.
    blocked : if True, trials are ordered stimulus by stimulus (block design), which is where
        drift and i.i.d. shuffles disagree most.
    Returns ``{"session": SessionData, "truth": DataFrame(unit, selective, preferred)}``.
    """
    rng = np.random.default_rng(seed)
    n_trials = n_stim * trials_per_stim
    stim = np.repeat(np.arange(n_stim), trials_per_stim)
    if not blocked:
        stim = rng.permutation(stim)
    onsets = 5.0 + np.arange(n_trials) * (iti + window)
    n_sel = int(round(frac_selective * n_units))
    selective = np.zeros(n_units, dtype=bool)
    selective[:n_sel] = True
    preferred = rng.integers(0, n_stim, size=n_units)
    preferred[~selective] = -1

    # slow drift shared shape per unit (random cosine mixture)
    t_norm = np.linspace(0, 1, n_trials)
    spike_times = []
    for u in range(n_units):
        if drift_amp > 0:
            phase, freq = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.5)
            drift = 1.0 + drift_amp * np.cos(2 * np.pi * freq * t_norm + phase)
        else:
            drift = np.ones(n_trials)
        rate_u = rng.gamma(shape=4.0, scale=base_rate / 4.0)  # unit-specific base rate
        trial_rates = np.full(n_trials, rate_u) * drift
        if selective[u]:
            trial_rates[stim == preferred[u]] *= gain
        spikes = []
        for i in range(n_trials):
            # baseline period (1 s before) at base rate, stimulus window at trial rate
            n_base = rng.poisson(rate_u * drift[i] * 1.0)
            spikes.append(onsets[i] - 1.0 + rng.uniform(0, 1.0, n_base))
            n_resp = rng.poisson(trial_rates[i] * window)
            spikes.append(onsets[i] + rng.uniform(0, window, n_resp))
        spike_times.append(np.sort(np.concatenate(spikes)))

    trials = pd.DataFrame({"stim_id": stim, "onset": onsets, "trial_index": np.arange(n_trials)})
    if blocked:
        trials["block"] = stim
    unit_meta = pd.DataFrame({
        "region": rng.choice(["hippocampus", "amygdala"], size=n_units),
        "electrode": rng.integers(0, max(1, n_units // 4), size=n_units),
        "SNR": rng.uniform(2, 15, size=n_units),
    })
    session = SessionData(session_id=f"synthetic_{seed}", spike_times=spike_times, trials=trials, unit_meta=unit_meta,
                          dataset="synthetic", patient_id=f"P{seed}")
    session.validate()
    truth = pd.DataFrame({"unit": np.arange(n_units), "selective": selective, "preferred": preferred})
    return {"session": session, "truth": truth}


def make_units_table(n_sessions: int = 6, units_per_session: int = 30, base_prev: float = 0.3, session_sd: float = 0.5,
                     criteria: Optional[Dict[str, float]] = None, seed: int = 0) -> pd.DataFrame:
    """Synthetic wide units table for the prevalence module (logit-normal session effects).

    ``criteria`` maps criterion name -> multiplicative odds factor relative to ``base_prev``.
    """
    rng = np.random.default_rng(seed)
    criteria = criteria or {"anova": 1.0, "binwise_ranksum": 0.6, "response_strength": 0.3}
    rows = []
    for s in range(n_sessions):
        eff = rng.normal(0, session_sd)
        for u in range(units_per_session):
            row = {"dataset": "synthetic", "session": f"S{s}", "patient": f"P{s // 2}",
                   "region": rng.choice(["hippocampus", "amygdala"]), "unit": u}
            for name, factor in criteria.items():
                logit = np.log(base_prev / (1 - base_prev)) + np.log(factor) + eff
                row[name] = bool(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["make_synthetic_session", "make_units_table"]
