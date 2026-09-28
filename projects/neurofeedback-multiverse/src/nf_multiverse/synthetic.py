"""Synthetic multichannel EEG for a neurofeedback session.

A 1/f background on every channel, an alpha oscillator whose amplitude grows
block by block (the "learning" ground truth), eye blinks that project mostly
onto frontal channels, and EMG bursts on temporal/frontal channels.  All
amplitudes are in microvolts.  The generator returns the clean alpha envelope
so that contingency between delivered feedback and the intended target can be
measured exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal as sps

DEFAULT_CHANNELS: Tuple[str, ...] = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2", "M1", "M2")

# anterior-posterior position 0 (front) .. 1 (back) for artifact projection
_AP: Dict[str, float] = {
    "Fp1": 0.0, "Fp2": 0.0, "F7": 0.2, "F3": 0.2, "Fz": 0.2, "F4": 0.2, "F8": 0.2,
    "T7": 0.5, "C3": 0.5, "Cz": 0.5, "C4": 0.5, "T8": 0.5,
    "P7": 0.75, "P3": 0.75, "Pz": 0.75, "P4": 0.75, "P8": 0.75, "O1": 1.0, "O2": 1.0, "M1": 0.6, "M2": 0.6,
}
_LATERAL = {"F7", "F8", "T7", "T8", "P7", "P8", "M1", "M2"}


@dataclass
class SimulatedSession:
    eeg: np.ndarray  # (n_channels, n_samples) microvolts
    ch_names: Tuple[str, ...]
    fs: float
    block_edges: List[Tuple[float, float]]  # (start_s, end_s) per training block
    baseline: Tuple[float, float]  # (start_s, end_s) of the pre-training baseline
    alpha_envelope: np.ndarray  # (n_samples,) clean target amplitude
    eog: np.ndarray  # (n_samples,) blink source
    emg: np.ndarray  # (n_samples,) muscle source
    meta: Dict[str, float] = field(default_factory=dict)


def pink_noise(n: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent noise with unit variance via spectral shaping."""
    white = rng.normal(size=n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(white) / f ** (exponent / 2)
    x = np.fft.irfft(spec, n)
    return (x - x.mean()) / (x.std() + 1e-12)


def _blink_train(n: int, fs: float, rng: np.random.Generator, rate_hz: float = 0.25, width_s: float = 0.12) -> np.ndarray:
    out = np.zeros(n)
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate_hz)
        if t * fs >= n:
            break
        c = int(t * fs)
        w = int(width_s * fs)
        idx = np.arange(max(0, c - 3 * w), min(n, c + 3 * w))
        out[idx] += np.exp(-0.5 * ((idx - c) / w) ** 2)
    return out


def _saccade_train(n: int, fs: float, rng: np.random.Generator, rate_hz: float = 0.5, dur_s: Tuple[float, float] = (0.2, 1.0), rise_s: float = 0.02) -> np.ndarray:
    """Step-like horizontal/vertical eye movements (corneo-retinal dipole shifts) with a short rise time."""
    out = np.zeros(n)
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate_hz)
        if t * fs >= n:
            break
        d = rng.uniform(*dur_s)
        s0, s1 = int(t * fs), min(n, int((t + d) * fs))
        out[s0:s1] += rng.choice([-1.0, 1.0])
    k = max(1, int(rise_s * fs))
    return np.convolve(out, np.ones(k) / k, mode="same")


def _emg_bursts(n: int, fs: float, rng: np.random.Generator, rate_hz: float = 0.1, dur_s: Tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """Broadband (15 Hz - Nyquist) muscle activity gated into bursts."""
    hf = rng.normal(size=n)
    b, a = sps.butter(4, [15 / (fs / 2), min(120, fs / 2 - 1) / (fs / 2)], btype="band")
    hf = sps.filtfilt(b, a, hf)
    hf /= hf.std() + 1e-12
    gate = np.zeros(n)
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate_hz)
        if t * fs >= n:
            break
        d = rng.uniform(*dur_s)
        gate[int(t * fs) : min(n, int((t + d) * fs))] = 1.0
    return hf * gate


def simulate_session(
    fs: float = 250.0,
    baseline_s: float = 30.0,
    block_s: float = 30.0,
    n_blocks: int = 8,
    learning_gain: float = 0.5,
    alpha_freq: float = 10.0,
    alpha_amp_uv: float = 8.0,
    background_uv: float = 10.0,
    eog_amp_uv: float = 0.0,
    emg_amp_uv: float = 0.0,
    target_channels: Sequence[str] = ("Pz", "P3", "P4"),
    ch_names: Sequence[str] = DEFAULT_CHANNELS,
    seed: int = 0,
) -> SimulatedSession:
    """Simulate one neurofeedback session (baseline + ``n_blocks`` training blocks).

    The alpha amplitude in block *b* is ``alpha_amp_uv * (1 + learning_gain * b / (n_blocks - 1))``
    plus slow random fluctuation; ``learning_gain = 0`` means no learning.
    """
    rng = np.random.default_rng(seed)
    ch_names = tuple(ch_names)
    n_ch = len(ch_names)
    total_s = baseline_s + n_blocks * block_s
    n = int(total_s * fs)
    t = np.arange(n) / fs
    eeg = np.vstack([pink_noise(n, rng) * background_uv for _ in range(n_ch)])

    # alpha oscillator with block-wise amplitude and slow drift
    env = np.ones(n) * alpha_amp_uv
    block_edges: List[Tuple[float, float]] = []
    for b in range(n_blocks):
        s0, s1 = baseline_s + b * block_s, baseline_s + (b + 1) * block_s
        block_edges.append((s0, s1))
        gain = 1.0 + learning_gain * (b / max(1, n_blocks - 1))
        env[int(s0 * fs) : int(s1 * fs)] = alpha_amp_uv * gain
    drift = 1 + 0.15 * sps.filtfilt(*sps.butter(2, 0.05 / (fs / 2)), rng.normal(size=n))
    env = env * np.clip(drift, 0.5, 1.5)
    phase = 2 * np.pi * alpha_freq * t + 0.3 * np.cumsum(rng.normal(size=n)) / np.sqrt(fs)
    alpha = env * np.sin(phase)
    w_alpha = np.array([1.0 if c in target_channels else (0.4 if _AP.get(c, 0.5) >= 0.7 else 0.1) for c in ch_names])
    eeg += w_alpha[:, None] * alpha[None, :]

    eog = (_blink_train(n, fs, rng) + 0.6 * _saccade_train(n, fs, rng)) * eog_amp_uv
    w_eog = np.array([np.exp(-3.0 * _AP.get(c, 0.5)) for c in ch_names])
    eeg += w_eog[:, None] * eog[None, :]

    emg = _emg_bursts(n, fs, rng) * emg_amp_uv
    w_emg = np.array([1.0 if c in _LATERAL else 0.15 for c in ch_names])
    eeg += w_emg[:, None] * emg[None, :]

    return SimulatedSession(eeg.astype(np.float32), ch_names, fs, block_edges, (0.0, baseline_s), env, eog, emg, {"learning_gain": learning_gain, "alpha_freq": alpha_freq, "seed": seed})
