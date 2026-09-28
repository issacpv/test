"""Generative model with a known artifact path and a known trait path.

For subject *i* and run *r*::

    b_i        ~ N(0, I_k)                               latent brain factors
    m_i        ~ N(0, 1)                                 trait motion (standardised)
    m_ir       = m_i + pi * z_r + u_ir                    run motion; z_r = run index (instrument)
    y_i        = a' b_i + c * m_i + e_i                   phenotype (c: trait confounding)
    FC_ir      = W_b b_i + w_a * m_ir + noise             artifact edges load on run motion

``simulate_motion_cohort`` returns everything needed to evaluate the estimators, including the
"clean" FC without the artifact term so that the true artifact-mediated covariance can be
computed for any fitted weight vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionCohort:
    X_runs: np.ndarray        # (n, R, p) observed FC
    X_clean_runs: np.ndarray  # (n, R, p) FC without the artifact term
    y: np.ndarray             # (n,)
    motion_runs: np.ndarray   # (n, R)
    motion_trait: np.ndarray  # (n,)
    control_motion: np.ndarray  # (n,) negative-control exposure (trait + independent noise)
    instrument: np.ndarray    # (n, R) run index
    w_artifact: np.ndarray    # (p,)
    families: np.ndarray      # (n,)


def simulate_motion_cohort(
    n_subjects: int = 400,
    n_runs: int = 4,
    n_edges: int = 300,
    n_factors: int = 5,
    artifact_strength: float = 0.5,
    trait_confounding: float = 0.4,
    brain_effect: float = 1.0,
    instrument_strength: float = 0.3,
    run_noise: float = 0.5,
    edge_noise: float = 1.0,
    control_noise: float = 0.5,
    seed: int = 0,
) -> MotionCohort:
    """Draw a synthetic cohort (see module docstring)."""
    rng = np.random.default_rng(seed)
    n, R, p, k = n_subjects, n_runs, n_edges, n_factors
    b = rng.normal(size=(n, k))
    m_trait = rng.normal(size=n)
    z = np.tile(np.arange(R, dtype=float), (n, 1))
    m_runs = m_trait[:, None] + instrument_strength * z + run_noise * rng.normal(size=(n, R))
    a = rng.normal(size=k) / np.sqrt(k)
    y = brain_effect * b @ a + trait_confounding * m_trait + rng.normal(size=n) * 0.5
    W_b = rng.normal(size=(k, p)) / np.sqrt(k)
    w_a = np.zeros(p)
    art = rng.choice(p, size=p // 5, replace=False)
    w_a[art] = artifact_strength * rng.choice([-1.0, 1.0], size=len(art))
    noise = edge_noise * rng.normal(size=(n, R, p))
    X_clean = (b @ W_b)[:, None, :] + noise
    X = X_clean + m_runs[:, :, None] * w_a[None, None, :]
    control = m_trait + control_noise * rng.normal(size=n)
    families = np.arange(n) // 2  # pairs, to exercise grouped folds
    return MotionCohort(X, X_clean, y, m_runs, m_trait, control, z, w_a, families)
