"""Per-epoch EEG/EOG features and a transparent feature-based sleep stager.

The stager is deliberately simple (gradient boosting on spectral features with temporal context) so that
its errors can be dissected; it is the 'source-domain-controlled' model of hypothesis H6. Pretrained deep
stagers (U-Sleep, YASA) are run outside this package and their posteriors ingested as (n_epochs, 5) arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), "beta": (16.0, 30.0),
}


def epoch_signal(x: np.ndarray, fs: float, epoch_len: float = 30.0) -> np.ndarray:
    """Reshape a 1-D signal into (n_epochs, n_samples) dropping the trailing partial epoch."""
    n = int(round(fs * epoch_len))
    n_epochs = len(x) // n
    return np.asarray(x[: n_epochs * n], dtype=float).reshape(n_epochs, n)


def _hjorth(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = np.diff(e, axis=1)
    d2 = np.diff(d1, axis=1)
    v0 = e.var(axis=1) + 1e-12
    v1 = d1.var(axis=1) + 1e-12
    v2 = d2.var(axis=1) + 1e-12
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return activity, mobility, complexity


def spectral_features(epochs: np.ndarray, fs: float, nperseg_s: float = 4.0) -> Tuple[np.ndarray, List[str]]:
    """Log absolute and relative band powers, spectral edge frequency and Hjorth parameters per epoch.

    Returns an (n_epochs, n_features) array and the feature names.
    """
    nperseg = int(round(fs * nperseg_s))
    f, pxx = welch(epochs, fs=fs, nperseg=min(nperseg, epochs.shape[1]), axis=1)
    total = np.trapezoid(pxx[:, (f >= 0.5) & (f <= 30.0)], f[(f >= 0.5) & (f <= 30.0)], axis=1) + 1e-12
    feats: List[np.ndarray] = []
    names: List[str] = []
    for name, (lo, hi) in BANDS.items():
        sel = (f >= lo) & (f < hi)
        bp = np.trapezoid(pxx[:, sel], f[sel], axis=1) + 1e-12
        feats.append(np.log(bp))
        names.append(f"log_{name}")
        feats.append(bp / total)
        names.append(f"rel_{name}")
    # spectral edge frequency (95%) within 0.5-30 Hz
    sel = (f >= 0.5) & (f <= 30.0)
    cum = np.cumsum(pxx[:, sel], axis=1)
    cum = cum / (cum[:, -1:] + 1e-12)
    sef = f[sel][np.argmax(cum >= 0.95, axis=1)]
    feats.append(sef)
    names.append("sef95")
    act, mob, comp = _hjorth(epochs)
    feats += [np.log(act), mob, comp]
    names += ["log_hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    # sigma/delta ratio is a compact N2 vs N3 discriminator
    feats.append(np.log((np.exp(feats[names.index("log_sigma")]) / np.exp(feats[names.index("log_delta")]))))
    names.append("log_sigma_delta_ratio")
    return np.column_stack(feats), names


def add_temporal_context(X: np.ndarray, radius: int = 2) -> np.ndarray:
    """Append centred rolling means (window 2*radius+1) of every feature, edge-padded."""
    if radius <= 0:
        return X
    pad = np.pad(X, ((radius, radius), (0, 0)), mode="edge")
    k = 2 * radius + 1
    cs = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), pad]), axis=0)
    roll = (cs[k:] - cs[:-k]) / k
    return np.hstack([X, roll])


def smooth_posteriors(P: np.ndarray, radius: int = 1) -> np.ndarray:
    """Centred moving-average smoothing of an (n_epochs, n_classes) posterior matrix, renormalised."""
    if radius <= 0:
        return P
    pad = np.pad(P, ((radius, radius), (0, 0)), mode="edge")
    k = 2 * radius + 1
    cs = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), pad]), axis=0)
    S = (cs[k:] - cs[:-k]) / k
    return S / S.sum(axis=1, keepdims=True)


@dataclass
class StagingBaseline:
    """Gradient-boosting stager on spectral features with temporal context.

    Classes are the 5-class convention (W=0, N1=1, N2=2, N3=3, R=4). ``predict_proba`` always returns a
    (n_epochs, 5) matrix with columns in that order, even if a class was absent from training.
    """

    context_radius: int = 2
    max_iter: int = 200
    learning_rate: float = 0.08
    random_state: int = 0
    model: Optional[HistGradientBoostingClassifier] = field(default=None, repr=False)
    classes_: Optional[np.ndarray] = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return add_temporal_context(np.asarray(X, dtype=float), self.context_radius)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "StagingBaseline":
        self.model = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=self.learning_rate,
                                                    class_weight="balanced", random_state=self.random_state)
        self.model.fit(self._prep(X), np.asarray(y), sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: np.ndarray, smooth_radius: int = 1) -> np.ndarray:
        assert self.model is not None, "call fit first"
        p = self.model.predict_proba(self._prep(X))
        full = np.zeros((p.shape[0], 5))
        for j, c in enumerate(self.classes_):
            full[:, int(c)] = p[:, j]
        return smooth_posteriors(full, smooth_radius)

    def predict(self, X: np.ndarray, smooth_radius: int = 1) -> np.ndarray:
        return np.argmax(self.predict_proba(X, smooth_radius), axis=1)


def features_from_record(eeg: np.ndarray, fs: float, eog: Optional[np.ndarray] = None,
                         epoch_len: float = 30.0) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix for one recording from a central EEG channel and (optionally) one EOG channel."""
    X, names = spectral_features(epoch_signal(eeg, fs, epoch_len), fs)
    if eog is not None:
        Xe, ne = spectral_features(epoch_signal(eog, fs, epoch_len), fs)
        n = min(len(X), len(Xe))
        X = np.hstack([X[:n], Xe[:n]])
        names = names + ["eog_" + k for k in ne]
    return X, names
