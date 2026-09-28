"""Synthetic multi-subject two-class EEG-like data with subject-specific covariance shifts.

Class structure mimics lateralised event-related desynchronisation: class 0 boosts variance
on a "left-hemisphere" channel group, class 1 on a "right-hemisphere" group. Each subject
applies a random congruence transform ``T C T^T`` (SPD ``T = expm(shift * S)`` with symmetric
random ``S``) to both class covariances, which is exactly the kind of shift Riemannian
re-centering and Euclidean alignment are designed to remove.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .covariance import expm_spd


def class_covariances(n_channels: int = 8, class_sep: float = 1.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Two SPD class covariances sharing a random background structure."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_channels, n_channels))
    base = A @ A.T / n_channels + np.eye(n_channels)
    half = n_channels // 2
    g0 = np.ones(n_channels)
    g0[:half] += class_sep
    g1 = np.ones(n_channels)
    g1[half:] += class_sep
    C0 = np.diag(np.sqrt(g0)) @ base @ np.diag(np.sqrt(g0))
    C1 = np.diag(np.sqrt(g1)) @ base @ np.diag(np.sqrt(g1))
    return C0, C1


def make_subject(n_trials: int = 100, n_channels: int = 8, n_times: int = 128, shift: float = 0.5, class_sep: float = 1.0,
                 seed: int = 0, base_seed: int = 0, trial_noise: float = 0.1) -> Dict[str, np.ndarray]:
    """Trials for one subject: ``X (n, C, T)``, labels ``y``, the subject transform ``T``.

    ``shift`` scales the log of the subject transform (0 = identical to the population).
    ``trial_noise`` adds small per-trial random congruence perturbations (session-like variability).
    """
    rng = np.random.default_rng(seed)
    C0, C1 = class_covariances(n_channels, class_sep, base_seed)
    S = rng.normal(size=(n_channels, n_channels))
    S = (S + S.T) / 2
    S /= np.linalg.norm(S, "fro") / np.sqrt(n_channels)
    T = expm_spd(shift * S)
    y = np.tile([0, 1], n_trials // 2 + 1)[:n_trials]
    y = rng.permutation(y)
    X = np.empty((n_trials, n_channels, n_times))
    for i in range(n_trials):
        C = T @ (C0 if y[i] == 0 else C1) @ T
        if trial_noise > 0:
            P = rng.normal(size=(n_channels, n_channels))
            P = (P + P.T) / 2
            P /= np.linalg.norm(P, "fro") / np.sqrt(n_channels)
            Ti = expm_spd(trial_noise * P)
            C = Ti @ C @ Ti
        L = np.linalg.cholesky(C)
        X[i] = L @ rng.normal(size=(n_channels, n_times))
    return {"X": X, "y": y, "T": T}


def make_multi_subject_dataset(n_subjects: int = 6, n_trials: int = 80, n_channels: int = 8, n_times: int = 128,
                               shift: float = 0.5, class_sep: float = 1.0, seed: int = 0) -> List[Dict[str, np.ndarray]]:
    """A list of subject dictionaries sharing the same population class covariances."""
    return [make_subject(n_trials, n_channels, n_times, shift, class_sep, seed=seed * 1000 + s, base_seed=seed)
            for s in range(n_subjects)]


def make_results_table(n_datasets: int = 4, subjects_per_dataset: int = 10, budgets: Tuple[int, ...] = (0, 5, 10, 20, 40, 80),
                       methods: Tuple[str, ...] = ("scratch", "ra_mdm"), transfer_gain: float = 0.08, seed: int = 0):
    """Synthetic long results table with a known method effect, for testing the statistics."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_datasets):
        d_eff = rng.normal(0, 0.03)
        for s in range(subjects_per_dataset):
            s_eff = rng.normal(0, 0.05)
            asym = 0.85 + d_eff + s_eff
            for m in methods:
                gain = transfer_gain if m != "scratch" else 0.0
                for k in budgets:
                    acc = asym - (0.3 - gain) * (k + 1) ** (-0.6) + rng.normal(0, 0.02)
                    rows.append({"dataset": f"D{d}", "subject": f"S{s}", "method": m, "budget": k, "draw": 0,
                                 "accuracy": float(np.clip(acc, 0, 1))})
    return pd.DataFrame(rows)


__all__ = ["class_covariances", "make_subject", "make_multi_subject_dataset", "make_results_table"]
