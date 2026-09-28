"""Longitudinal cohort simulator with amyloid, growing WMH, an interaction on cognitive slope and
a progression hazard with additive interaction.

Per subject: amyloid ``A ~ Bernoulli(p_amyloid)``; baseline log-WMH ``W0 ~ N(mu_w + age
effect, sd)``; WMH grows linearly (faster in older subjects); cognition ``y(t) = b0 +
(slope0 + b_a A + b_w W(t) + b_aw A W(t)) t + random slope + noise``; progression hazard per
year ``expit(h0 + h_a A + h_w Wbin + h_aw A Wbin)`` with ``Wbin = 1[W0 > median]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def simulate_wmh_cohort(
    n_subjects: int = 400,
    n_visits: Sequence[int] = (3, 4, 5, 6),
    visit_interval: float = 1.2,
    p_amyloid: float = 0.35,
    slope0: float = -0.02,
    b_a: float = -0.05,
    b_w: float = -0.02,
    b_aw: float = -0.06,
    wmh_growth: float = 0.08,
    random_slope_sd: float = 0.04,
    noise_sd: float = 0.15,
    h0: float = -3.5,
    h_a: float = 0.8,
    h_w: float = 0.5,
    h_aw: float = 0.7,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a long table: ``subject, visit, time, age, sex, amyloid_pos, wmh_log, wmh_bin,
    cognition, cdr_event, event_time, event``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        A = int(rng.random() < p_amyloid)
        age0 = rng.uniform(60, 85)
        sex = int(rng.random() < 0.55)
        w0 = rng.normal(0.8 + 0.03 * (age0 - 70), 0.6)
        n_v = int(rng.choice(n_visits))
        times = np.concatenate([[0.0], np.cumsum(rng.uniform(0.7, 1.3, n_v - 1) * visit_interval)])
        slope_i = rng.normal(0, random_slope_sd)
        b0 = rng.normal(0, 0.5)
        wbin = int(w0 > 0.8)
        # progression: discrete annual hazard
        hazard = 1 / (1 + np.exp(-(h0 + h_a * A + h_w * wbin + h_aw * A * wbin)))
        t_end = times[-1]
        event, event_time = 0, t_end
        for yr in range(1, int(np.ceil(t_end)) + 1):
            if rng.random() < hazard:
                event, event_time = 1, float(yr)
                break
        for v, t in enumerate(times):
            w_t = w0 + wmh_growth * t * (1 + 0.02 * (age0 - 70))
            slope = slope0 + b_a * A + b_w * w_t + b_aw * A * w_t + slope_i
            y = b0 + slope * t + rng.normal(0, noise_sd)
            rows.append(
                {
                    "subject": f"S{i:04d}",
                    "visit": v,
                    "time": t,
                    "age": age0 + t,
                    "sex": sex,
                    "amyloid_pos": A,
                    "wmh_log": w_t,
                    "wmh_bin": wbin,
                    "cognition": y,
                    "event_time": event_time,
                    "event": event,
                }
            )
    return pd.DataFrame(rows)
