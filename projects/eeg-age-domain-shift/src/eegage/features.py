"""Spectral window features and an age-conditioned normative scaler.

``spectral_features`` maps harmonized windows of shape (n_windows, n_channels,
n_samples) to a (n_windows, n_channels * 14) matrix: log absolute and relative
power in five bands, spectral edge 95%, peak frequency in 3-13 Hz (a proxy for
the posterior dominant rhythm, which matures with age), line length, and Hjorth
mobility/complexity.

``AgeNormativeScaler`` is the label-free adaptation step: for each feature it
fits mean and log-SD as polynomial functions of log(age+1) on background windows
of training subjects, then z-scores every window given its subject's age.  This
mirrors normative modelling in neuroimaging (Marquand et al., 2016; Rutherford
et al., 2022) applied to EEG background statistics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}
PER_CHANNEL_FEATURES: List[str] = (
    [f"log_{b}" for b in BANDS] + [f"rel_{b}" for b in BANDS]
    + ["sef95", "peak_freq_3_13", "line_length", "hjorth_mobility", "hjorth_complexity"]
)


def feature_names(n_channels: int, channel_names: Optional[Sequence[str]] = None) -> List[str]:
    ch = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(n_channels)]
    return [f"{c}:{f}" for c in ch for f in PER_CHANNEL_FEATURES]


def _hjorth(x: np.ndarray) -> Tuple[float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0, v1, v2 = np.var(x) + 1e-12, np.var(dx) + 1e-12, np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return float(mob), float(comp)


def spectral_features(X: np.ndarray, fs: float, bands: Dict[str, Tuple[float, float]] = BANDS,
                      nperseg: Optional[int] = None) -> np.ndarray:
    """Per-window, per-channel spectral/time features. ``X``: (n_windows, n_channels, n_samples)."""
    X = np.asarray(X, dtype=float)
    n, c, L = X.shape
    nperseg = nperseg or min(L, int(2 * fs))
    f, P = sps.welch(X, fs=fs, nperseg=nperseg, axis=-1)  # (n, c, nf)
    df = f[1] - f[0]
    fmax = min(70.0, 0.5 * fs)
    tot = P[..., (f >= 0.5) & (f <= fmax)].sum(-1) * df + 1e-12
    feats = []
    for lo, hi in bands.values():
        sel = (f >= lo) & (f < min(hi, fmax))
        feats.append(np.log(P[..., sel].sum(-1) * df + 1e-12))
    for lo, hi in bands.values():
        sel = (f >= lo) & (f < min(hi, fmax))
        feats.append(P[..., sel].sum(-1) * df / tot)
    # spectral edge 95 %
    sel = (f >= 0.5) & (f <= fmax)
    cum = np.cumsum(P[..., sel], axis=-1)
    cum = cum / (cum[..., -1:] + 1e-12)
    idx = (cum >= 0.95).argmax(-1)
    feats.append(f[sel][idx])
    # peak frequency in 3-13 Hz
    sel2 = (f >= 3.0) & (f <= 13.0)
    feats.append(f[sel2][P[..., sel2].argmax(-1)])
    # line length
    feats.append(np.abs(np.diff(X, axis=-1)).mean(-1))
    mob = np.empty((n, c))
    comp = np.empty((n, c))
    for i in range(n):
        for j in range(c):
            mob[i, j], comp[i, j] = _hjorth(X[i, j])
    feats += [mob, comp]
    F = np.stack(feats, axis=-1)  # (n, c, n_feat)
    return F.reshape(n, c * F.shape[-1])


class AgeNormativeScaler:
    """Age-conditioned z-scoring: mean(age) and log-SD(age) are polynomials in log(age+1)."""

    def __init__(self, degree_mean: int = 2, degree_sd: int = 1, min_sd: float = 1e-6):
        self.degree_mean, self.degree_sd, self.min_sd = degree_mean, degree_sd, min_sd
        self.coef_mean_: Optional[np.ndarray] = None
        self.coef_sd_: Optional[np.ndarray] = None

    @staticmethod
    def _design(age: np.ndarray, degree: int) -> np.ndarray:
        la = np.log1p(np.asarray(age, float))
        return np.column_stack([la ** k for k in range(degree + 1)])

    def fit(self, F: np.ndarray, age: np.ndarray) -> "AgeNormativeScaler":
        F = np.asarray(F, float)
        age = np.asarray(age, float)
        Am = self._design(age, self.degree_mean)
        As = self._design(age, self.degree_sd)
        p = F.shape[1]
        self.coef_mean_ = np.zeros((Am.shape[1], p))
        self.coef_sd_ = np.zeros((As.shape[1], p))
        for j in range(p):
            ok = np.isfinite(F[:, j]) & np.isfinite(age)
            if ok.sum() <= Am.shape[1]:
                continue
            bm, *_ = np.linalg.lstsq(Am[ok], F[ok, j], rcond=None)
            self.coef_mean_[:, j] = bm
            resid = F[ok, j] - Am[ok] @ bm
            # log|resid| regression gives a smooth log-scale; add log(sqrt(pi/2)) for E|z| of a normal
            target = np.log(np.abs(resid) + 1e-12) + 0.5 * np.log(np.pi / 2)
            bs, *_ = np.linalg.lstsq(As[ok], target, rcond=None)
            self.coef_sd_[:, j] = bs
        return self

    def transform(self, F: np.ndarray, age: np.ndarray) -> np.ndarray:
        if self.coef_mean_ is None:
            raise RuntimeError("fit first")
        F = np.asarray(F, float)
        Am = self._design(age, self.degree_mean)
        As = self._design(age, self.degree_sd)
        mu = Am @ self.coef_mean_
        sd = np.maximum(np.exp(As @ self.coef_sd_), self.min_sd)
        return (F - mu) / sd

    def fit_transform(self, F: np.ndarray, age: np.ndarray) -> np.ndarray:
        return self.fit(F, age).transform(F, age)

    def expected(self, age: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normative mean and SD for each feature at the given ages (for centile plots)."""
        mu = self._design(age, self.degree_mean) @ self.coef_mean_
        sd = np.maximum(np.exp(self._design(age, self.degree_sd) @ self.coef_sd_), self.min_sd)
        return mu, sd


def per_record_standardize(F: np.ndarray, record_ids: Sequence[str]) -> np.ndarray:
    """Baseline adaptation: z-score each feature within each record."""
    F = np.asarray(F, float)
    out = F.copy()
    rec = np.asarray(record_ids)
    for r in np.unique(rec):
        m = rec == r
        mu = np.nanmean(F[m], axis=0)
        sd = np.nanstd(F[m], axis=0) + 1e-9
        out[m] = (F[m] - mu) / sd
    return out
