"""Semi-synthetic ICU cohort generator with *informative lab ordering*.

A latent severity ``z`` drives four things at once: the outcome, the probability
that each lab is ordered in each hour, the probability that an order is STAT
(as opposed to a routine morning draw), and the measured value. Two knobs
separate the information channels that the real analysis tries to disentangle:

* ``gamma``  - ordering informativeness: how strongly clinicians' order
  decisions track severity (the "leak" through the observation process).
* ``lam``    - value informativeness: how strongly the measured value tracks
  severity (physiology).

``site_shift`` adds a constant to the baseline ordering log-odds without
touching physiology, so an ordering-policy shift between sites (e.g. eICU
hospitals with different daily-lab intensity) can be simulated exactly.

The long-table output mimics the columns produced by ``lab_leak.sql`` on
MIMIC-IV / eICU so the same feature builder runs on real and synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass
class SimConfig:
    """Parameters of the semi-synthetic cohort (all log-odds unless stated)."""

    n_stays: int = 500
    n_labs: int = 6
    horizon_hours: int = 24
    gamma: float = 1.0
    lam: float = 0.8
    base_logit: float = -3.0
    morning_boost: float = 3.5
    morning_window: tuple[int, int] = (4, 8)
    site_shift: float = 0.0
    stat_logit: float = -1.0
    outcome_intercept: float = -1.5
    outcome_slope: float = 1.5
    value_noise: float = 1.0
    seed: int = 0


def simulate_cohort(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ``(long, stays)`` tables.

    Returns
    -------
    long : DataFrame with columns ``stay_id, t_hours, hod, itemid, valuenum, priority``
        one row per performed lab measurement (``hod`` = hour of day 0-23).
    stays : DataFrame with columns ``stay_id, admit_hour, severity, y``.
    """
    rng = np.random.default_rng(cfg.seed)
    n, H, K = cfg.n_stays, cfg.horizon_hours, cfg.n_labs
    z = rng.normal(size=n)
    admit_hour = rng.integers(0, 24, size=n)
    y = rng.random(n) < sigmoid(cfg.outcome_intercept + cfg.outcome_slope * z)
    stays = pd.DataFrame({"stay_id": np.arange(n), "admit_hour": admit_hour, "severity": z, "y": y.astype(int)})

    lab_offsets = rng.normal(0.0, 0.5, size=K)
    hours = np.arange(H)
    hod = (admit_hour[:, None] + hours[None, :]) % 24  # (n, H)
    lo, hi = cfg.morning_window
    morning = (hod >= lo) & (hod < hi)  # (n, H)

    # ordering log-odds: routine draws are (almost) severity-independent, ad hoc draws track severity
    sev_weight = np.where(morning, 0.25, 1.0)
    logit = (
        cfg.base_logit
        + cfg.site_shift
        + lab_offsets[None, None, :]
        + cfg.gamma * z[:, None, None] * sev_weight[:, :, None]
        + cfg.morning_boost * morning[:, :, None]
    )
    ordered = rng.random((n, H, K)) < sigmoid(logit)
    idx_stay, idx_hour, idx_lab = np.nonzero(ordered)
    m = idx_stay.size
    values = cfg.lam * z[idx_stay] + cfg.value_noise * rng.normal(size=m)
    stat_p = sigmoid(cfg.stat_logit + cfg.gamma * z[idx_stay] - 2.5 * morning[idx_stay, idx_hour])
    stat = rng.random(m) < stat_p
    long = pd.DataFrame(
        {
            "stay_id": idx_stay,
            "t_hours": idx_hour + rng.random(m),
            "hod": hod[idx_stay, idx_hour],
            "itemid": idx_lab,
            "valuenum": values,
            "priority": np.where(stat, "STAT", "ROUTINE"),
        }
    ).sort_values(["stay_id", "t_hours"], ignore_index=True)
    return long, stays


def two_site_cohorts(cfg: SimConfig, site_shift: float, seed_offset: int = 1000) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Source and target cohorts that differ only in ordering intensity (policy shift)."""
    src = simulate_cohort(cfg)
    tgt_cfg = SimConfig(**{**cfg.__dict__, "site_shift": cfg.site_shift + site_shift, "seed": cfg.seed + seed_offset})
    tgt = simulate_cohort(tgt_cfg)
    return {"source": src, "target": tgt}
