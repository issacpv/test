"""Harmonisation of heterogeneous wearable gait recordings.

Pipeline: units -> m/s^2, polyphase resampling to a common rate, gravity
estimation by low-pass filtering, orientation-invariant channels
(magnitude, PCA-vertical component, horizontal magnitude) and windowing
with subject bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

G = 9.80665


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str        # 'accel', 'accel+gyro', 'vgrf', 'footswitch'
    placement: str       # 'foot', 'ankle', 'thigh', 'lower_back', 'wrist', 'pocket'
    fs_hz: float
    units: str           # 'g', 'mg', 'm/s2', 'N', 'binary'
    access: str


REGISTRY: dict[str, DatasetSpec] = {
    "gaitpdb": DatasetSpec("gaitpdb", "vgrf", "foot", 100.0, "N", "open (PhysioNet)"),
    "gaitndd": DatasetSpec("gaitndd", "footswitch", "foot", 300.0, "binary", "open (PhysioNet)"),
    "ltmm": DatasetSpec("ltmm", "accel", "lower_back", 100.0, "g", "open (PhysioNet)"),
    "pads": DatasetSpec("pads", "accel+gyro", "wrist", 100.0, "m/s2", "open (PhysioNet)"),
    "daphnet": DatasetSpec("daphnet", "accel", "ankle", 64.0, "mg", "open (UCI)"),
    "kaggle_tdcsfog": DatasetSpec("kaggle_tdcsfog", "accel", "lower_back", 128.0, "g", "Kaggle rules"),
    "kaggle_defog": DatasetSpec("kaggle_defog", "accel", "lower_back", 100.0, "g", "Kaggle rules"),
    "mpower": DatasetSpec("mpower", "accel", "pocket", 100.0, "g", "Synapse registration"),
    "ppmi_verily": DatasetSpec("ppmi_verily", "accel", "wrist", 0.0, "derived", "PPMI application + DUA"),
}


def to_m_s2(x: np.ndarray, units: str) -> np.ndarray:
    """Convert acceleration to m/s^2 from 'g', 'mg' or 'm/s2'."""
    x = np.asarray(x, float)
    if units == "m/s2":
        return x
    if units == "g":
        return x * G
    if units == "mg":
        return x * G / 1000.0
    raise ValueError(f"unknown units {units}")


def resample(x: np.ndarray, fs_in: float, fs_out: float, max_denominator: int = 1000) -> np.ndarray:
    """Polyphase resampling along axis 0 using a rational approximation of fs_out/fs_in."""
    if fs_in == fs_out:
        return np.asarray(x, float)
    frac = Fraction(fs_out / fs_in).limit_denominator(max_denominator)
    return resample_poly(np.asarray(x, float), frac.numerator, frac.denominator, axis=0)


def gravity_component(x: np.ndarray, fs: float, cutoff_hz: float = 0.5, order: int = 2) -> np.ndarray:
    """Low-pass estimate of the gravity vector per sample (``[T, 3]``)."""
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, float), axis=0)


def remove_gravity(x: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    return np.asarray(x, float) - gravity_component(x, fs, cutoff_hz)


def orientation_invariant(x: np.ndarray, fs: float) -> np.ndarray:
    """Three orientation-invariant channels from a ``[T, 3]`` accelerometer signal.

    Columns: dynamic magnitude, component along the mean gravity direction
    (vertical), magnitude of the horizontal residual.
    """
    x = np.asarray(x, float)
    g = gravity_component(x, fs)
    g_dir = g.mean(0)
    norm = np.linalg.norm(g_dir)
    if norm < 1e-9:
        # no gravity information (already dynamic): fall back to the principal axis
        u, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
        g_dir = vt[0]
    else:
        g_dir = g_dir / norm
    dyn = x - g
    vertical = dyn @ g_dir
    horizontal = dyn - np.outer(vertical, g_dir)
    return np.column_stack([np.linalg.norm(dyn, axis=1), vertical, np.linalg.norm(horizontal, axis=1)])


def window(x: np.ndarray, fs: float, win_s: float = 5.0, step_s: float = 2.5) -> np.ndarray:
    """Slice ``[T, C]`` into ``[n_win, L, C]`` windows (drops the incomplete tail)."""
    x = np.asarray(x, float)
    L = int(round(win_s * fs))
    S = int(round(step_s * fs))
    if len(x) < L:
        return np.empty((0, L, x.shape[1]))
    starts = np.arange(0, len(x) - L + 1, S)
    return np.stack([x[s: s + L] for s in starts])


def harmonize(x: np.ndarray, fs_in: float, units: str, fs_out: float = 50.0, invariant: bool = True) -> tuple[np.ndarray, float]:
    """Full harmonisation: units -> m/s^2, resample, orientation-invariant channels (or gravity-free axes)."""
    y = to_m_s2(x, units)
    y = resample(y, fs_in, fs_out)
    y = orientation_invariant(y, fs_out) if invariant else remove_gravity(y, fs_out)
    return y, fs_out
