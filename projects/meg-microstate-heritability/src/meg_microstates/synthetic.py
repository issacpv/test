"""Synthetic data: multichannel microstate recordings, ECG, and twin phenotypes with known h2."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def simulate_microstate_data(n_channels: int = 32, sfreq: float = 250.0, duration_s: float = 60.0, n_states: int = 4,
                             mean_duration_ms: float = 80.0, snr: float = 3.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Piecewise-stable topographies with random durations plus spatially white noise.

    Each segment is one template scaled by a slowly varying positive envelope (so GFP peaks occur
    inside segments). Returns ``data (n_channels, n_times)``, ``labels``, ``templates``.
    """
    rng = np.random.default_rng(seed)
    n_times = int(duration_s * sfreq)
    templates = rng.normal(size=(n_states, n_channels))
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    labels = np.empty(n_times, dtype=int)
    data = np.empty((n_channels, n_times))
    t = 0
    current = int(rng.integers(n_states))
    while t < n_times:
        seg = max(2, int(rng.gamma(shape=4.0, scale=mean_duration_ms / 4.0 / 1000 * sfreq)))
        seg = min(seg, n_times - t)
        env = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, seg))  # amplitude envelope with a peak mid-segment
        sign = rng.choice([-1.0, 1.0])
        data[:, t:t + seg] = sign * np.outer(templates[current], env) * snr
        labels[t:t + seg] = current
        t += seg
        current = int((current + rng.integers(1, n_states)) % n_states)  # never repeat the same state
    data += rng.normal(size=data.shape)
    return {"data": data, "labels": labels, "templates": templates, "sfreq": np.array(sfreq)}


def simulate_ecg(duration_s: float = 120.0, sfreq: float = 500.0, hr_bpm: float = 65.0, rr_sd_s: float = 0.04,
                 noise: float = 0.05, seed: int = 0) -> Dict[str, np.ndarray]:
    """Toy ECG: Gaussian R waves at RR intervals with Gaussian variability, small T waves, white noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n) / sfreq
    ecg = np.zeros(n)
    peaks = []
    beat = 0.5
    mean_rr = 60.0 / hr_bpm
    while beat < duration_s - 0.5:
        idx = int(round(beat * sfreq))
        peaks.append(idx)
        ecg += 1.0 * np.exp(-0.5 * ((t - beat) / 0.012) ** 2)      # R wave
        ecg += 0.25 * np.exp(-0.5 * ((t - beat - 0.25) / 0.05) ** 2)  # T wave
        beat += max(0.3, rng.normal(mean_rr, rr_sd_s))
    ecg += noise * rng.normal(size=n)
    return {"ecg": ecg, "peaks": np.asarray(peaks, dtype=int), "sfreq": np.array(sfreq)}


def simulate_twin_phenotypes(n_mz: int, n_dz: int, h2: float, c2: float = 0.0, seed: int = 0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Standardised twin-pair phenotypes with additive-genetic (h2), shared (c2) and unique variance."""
    if h2 < 0 or c2 < 0 or h2 + c2 > 1:
        raise ValueError("need h2, c2 >= 0 and h2 + c2 <= 1")
    rng = np.random.default_rng(seed)
    e2 = 1 - h2 - c2

    def pairs(n: int, r_a: float) -> np.ndarray:
        a_shared = rng.normal(size=(n, 1))
        a_unique = rng.normal(size=(n, 2))
        a = np.sqrt(r_a) * a_shared + np.sqrt(1 - r_a) * a_unique
        c = np.repeat(rng.normal(size=(n, 1)), 2, axis=1)
        e = rng.normal(size=(n, 2))
        return np.sqrt(h2) * a + np.sqrt(c2) * c + np.sqrt(e2) * e

    return pairs(n_mz, 1.0), pairs(n_dz, 0.5)


__all__ = ["simulate_microstate_data", "simulate_ecg", "simulate_twin_phenotypes"]
