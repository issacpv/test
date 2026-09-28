"""Feedback specificity: how much of the delivered signal is artifact, and how much is target.

``artifact_regressors`` builds per-window nuisance regressors from channels
that are *not* the target - frontal-polar low-frequency power (blinks, eye
movements), lateral high-frequency power (EMG), broadband RMS (movement /
electrode pops) - on the same window grid as the feedback signal.
``feedback_specificity`` reports the variance of the feedback values explained
by these regressors.  ``contingency`` measures the dependence between the
delivered feedback and a reference target signal (a cleaner offline estimate
of the same quantity, or the known ground truth in simulations).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from .feedback import _idx, band_power_windows


def artifact_regressors(
    eeg: np.ndarray,
    ch_names: Sequence[str],
    fs: float,
    window_s: float,
    step_s: float,
    eog_channels: Sequence[str] = ("Fp1", "Fp2"),
    emg_channels: Sequence[str] = ("F7", "F8", "T7", "T8"),
    emg_band: Tuple[float, float] = (30.0, 100.0),
) -> pd.DataFrame:
    """Per-window nuisance regressors (log power) aligned with the feedback grid."""
    eeg = np.asarray(eeg, float)
    eog = eeg[_idx(ch_names, eog_channels)].mean(0) if _idx(ch_names, eog_channels) else np.zeros(eeg.shape[1])
    emg_idx = _idx(ch_names, emg_channels)
    emg = eeg[emg_idx].mean(0) if emg_idx else np.zeros(eeg.shape[1])
    t, p_eog = band_power_windows(eog, fs, (0.5, 4.0), window_s, step_s, "hilbert")
    _, p_emg = band_power_windows(emg, fs, (emg_band[0], min(emg_band[1], fs / 2 - 1)), window_s, step_s, "bandpass_rms")
    w = int(round(window_s * fs))
    s = max(1, int(round(step_s * fs)))
    starts = np.arange(0, eeg.shape[1] - w + 1, s)
    rms = np.array([np.sqrt(np.mean(eeg[:, st : st + w] ** 2)) for st in starts])
    return pd.DataFrame({"time": t, "log_eog_power": np.log(p_eog + 1e-12), "log_emg_power": np.log(p_emg + 1e-12), "log_broadband_rms": np.log(rms + 1e-12)})


def feedback_specificity(values: np.ndarray, regressors: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """OLS of feedback values on nuisance regressors; returns R^2, adjusted R^2 and standardised betas."""
    cols = list(columns) if columns is not None else [c for c in regressors.columns if c != "time"]
    y = np.asarray(values, float)
    X = regressors[cols].to_numpy(float)
    n = min(len(y), len(X))
    y, X = y[:n], X[:n]
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    if len(y) < len(cols) + 3:
        return {"r2": np.nan, "adj_r2": np.nan, "n": float(len(y))}
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    A = np.c_[np.ones(len(ys)), Xs]
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ beta
    r2 = 1 - float(resid @ resid) / float(ys @ ys)
    k = len(cols)
    adj = 1 - (1 - r2) * (len(ys) - 1) / max(1, len(ys) - k - 1)
    out = {"r2": float(r2), "adj_r2": float(adj), "n": float(len(ys))}
    for c, b in zip(cols, beta[1:]):
        out[f"beta_{c}"] = float(b)
    return out


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Plug-in mutual information (nats) on quantile-binned variables."""
    xq = np.quantile(x, np.linspace(0, 1, bins + 1))
    yq = np.quantile(y, np.linspace(0, 1, bins + 1))
    xi = np.clip(np.searchsorted(xq, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(yq, y, side="right") - 1, 0, bins - 1)
    joint = np.zeros((bins, bins))
    np.add.at(joint, (xi, yi), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))


def contingency(feedback: np.ndarray, target: np.ndarray, decisions: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Dependence between delivered feedback and the reference target signal.

    Returns Spearman rho, plug-in mutual information, and - if binary reward
    ``decisions`` are given - the point-biserial correlation between reward and
    target and the reward rate.
    """
    f, t = np.asarray(feedback, float), np.asarray(target, float)
    n = min(len(f), len(t))
    f, t = f[:n], t[:n]
    m = np.isfinite(f) & np.isfinite(t)
    out: Dict[str, float] = {"n": float(m.sum())}
    if m.sum() < 5:
        return dict(out, spearman=np.nan, mutual_information=np.nan)
    rho, p = stats.spearmanr(f[m], t[m])
    out["spearman"] = float(rho)
    out["spearman_p"] = float(p)
    out["mutual_information"] = _mutual_information(f[m], t[m])
    if decisions is not None:
        d = np.asarray(decisions[:n], bool)[m]
        out["reward_rate"] = float(d.mean())
        out["reward_target_pointbiserial"] = float(stats.pointbiserialr(d.astype(int), t[m])[0]) if 0 < d.mean() < 1 else np.nan
    return out


def target_reference_signal(alpha_envelope: np.ndarray, fs: float, times: np.ndarray, window_s: float) -> np.ndarray:
    """Mean of a ground-truth envelope inside each feedback window (simulation helper)."""
    w = int(round(window_s * fs))
    out = np.empty(len(times))
    for i, tc in enumerate(times):
        st = int(round(tc * fs - w / 2))
        out[i] = float(np.mean(alpha_envelope[max(0, st) : st + w]))
    return out
