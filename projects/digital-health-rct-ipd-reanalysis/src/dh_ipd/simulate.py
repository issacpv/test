"""Simulate multi-trial individual-participant data for a digital-health RCT programme.

The generative model encodes the three problems the project studies:

* **engagement confounding** - an unobserved trait (``motivation``) raises both
  engagement with the app and the outcome, so within-arm engagement-outcome
  correlations overstate the causal effect of engaging;
* **principal strata** - every participant has a latent "would engage if
  offered the app" status ``S``, observed only in the treated arm, so
  complier-type effects are well defined for controls too;
* **informative attrition** - dropout depends on the (unobserved) outcome and is
  higher among treated non-engagers, i.e. missing not at random.

Outcomes are on a standardised improvement scale (higher = better).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_ipd(
    n_trials: int = 5,
    n_per_trial: int = 300,
    ate: float = 0.3,
    engagement_effect: float = 0.3,
    engagement_confounding: float = 1.0,
    hte_severity: float = 0.2,
    trial_sd: float = 0.15,
    dropout_base: float = -1.5,
    mnar_strength: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per participant with columns

    ``trial, id, z, baseline, age, sex, motivation, S, engaged, dose, y_full, dropout, y``.

    ``y`` is ``y_full`` with dropouts set to NaN.  ``engaged`` is ``S`` for
    treated participants and 0 for controls (never offered the app).
    ``dose`` is the number of app sessions (0 for controls).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_trials):
        n = n_per_trial
        u_t = rng.normal(0, trial_sd)  # trial intercept
        d_t = rng.normal(0, trial_sd / 2)  # trial-specific effect modifier
        baseline = rng.normal(0, 1, n)  # standardised symptom severity at baseline
        age = rng.normal(40, 12, n)
        sex = rng.integers(0, 2, n)
        motivation = rng.normal(0, 1, n)  # unobserved
        z = rng.integers(0, 2, n)
        lat = -0.1 + engagement_confounding * motivation + 0.2 * baseline + rng.normal(0, 0.8, n)
        S = (lat > 0).astype(int)  # would engage if treated
        engaged = S * z
        dose = np.where(engaged == 1, rng.poisson(8, n) + 3, np.where(z == 1, rng.poisson(1, n), 0))
        y_full = u_t + 0.5 * baseline + 0.4 * motivation + z * (ate + d_t + hte_severity * baseline) + z * S * engagement_effect + rng.normal(0, 1, n)
        drop_logit = dropout_base - mnar_strength * y_full + 0.6 * z * (1 - S)
        dropout = (rng.uniform(size=n) < 1 / (1 + np.exp(-drop_logit))).astype(int)
        y = np.where(dropout == 1, np.nan, y_full)
        frames.append(
            pd.DataFrame(
                {
                    "trial": f"T{t + 1:02d}",
                    "id": [f"T{t + 1:02d}-{i:04d}" for i in range(n)],
                    "z": z,
                    "baseline": baseline,
                    "age": age,
                    "sex": sex,
                    "motivation": motivation,
                    "S": S,
                    "engaged": engaged,
                    "dose": dose,
                    "y_full": y_full,
                    "dropout": dropout,
                    "y": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
