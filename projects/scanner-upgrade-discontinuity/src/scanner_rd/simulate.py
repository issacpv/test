"""Synthetic longitudinal cohort with a scanner upgrade.

Each subject has annual-ish sessions, a true linear trajectory (intercept + group slope), and a
scanner transition at a random session. The new scanner adds an additive offset (``jump``) and a
multiplicative scale (``scale``) to the measured feature, plus session noise. The generator
returns a long table compatible with :func:`scanner_rd.rdit.local_linear_rd`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def simulate_cohort(
    n_subjects: int = 200,
    sessions_per_subject: Sequence[int] = (3, 4, 5, 6),
    interval_years: float = 1.0,
    interval_jitter: float = 0.2,
    slope_by_group: Sequence[float] = (-40.0, -90.0),
    group_prob: Sequence[float] = (0.7, 0.3),
    baseline_mean: float = 3600.0,
    baseline_sd: float = 350.0,
    jump: float = 80.0,
    scale: float = 1.0,
    slope_change: float = 0.0,
    noise_sd: float = 40.0,
    transition_prob: float = 0.6,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Return a long table with ``subject, session, years, group, scanner, t_rel, post, y_true, y``.

    ``y`` is e.g. a hippocampal volume in mm^3; the defaults give ~1-2 %/year atrophy and a
    ~2 % scanner offset. ``t_rel`` is years relative to each transitioned subject's first
    post-upgrade session (NaN for subjects that never transition).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        g = rng.choice(len(slope_by_group), p=group_prob)
        n_s = int(rng.choice(sessions_per_subject))
        gaps = interval_years + rng.uniform(-interval_jitter, interval_jitter, size=n_s - 1)
        years = np.concatenate([[0.0], np.cumsum(gaps)])
        base = rng.normal(baseline_mean, baseline_sd)
        slope = slope_by_group[g] + rng.normal(0, 10)
        y_true = base + slope * years
        transitions = rng.random() < transition_prob and n_s >= 2
        k = int(rng.integers(1, n_s)) if transitions else n_s  # index of first post session
        for j in range(n_s):
            post = int(j >= k)
            t_rel = years[j] - years[k] if transitions else np.nan
            y = y_true[j]
            if post:
                y = scale * y + jump + slope_change * t_rel
            y += rng.normal(0, noise_sd)
            rows.append(
                {
                    "subject": f"S{i:04d}",
                    "session": j,
                    "years": years[j],
                    "group": g,
                    "scanner": "B" if post else "A",
                    "t_rel": t_rel,
                    "post": post,
                    "y_true": y_true[j],
                    "y": y,
                }
            )
    return pd.DataFrame(rows)
