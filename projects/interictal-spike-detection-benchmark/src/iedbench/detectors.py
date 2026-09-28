"""Baseline IED detectors: envelope (Janca-style), matched filter and morphology classifier.

All detectors take a (n_channels, n_samples) array and return a list of
:class:`iedbench.events.Event` with a continuous ``score`` so that threshold
sweeps and sensitivity-at-FP-rate curves can be computed.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, signal as sps

from .events import Event, events_from_mask
from .morphology import FEATURE_NAMES, features_matrix


def bandpass(X: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(hi, 0.95 * nyq)
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, np.asarray(X, float), axis=-1)


def robust_scale(X: np.ndarray) -> np.ndarray:
    """Per-channel median/MAD scaling (used before cross-modality transfer)."""
    X = np.asarray(X, float)
    med = np.median(X, axis=-1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=-1, keepdims=True) * 1.4826 + 1e-9
    return (X - med) / mad


def envelope_detector(X: np.ndarray, fs: float, channel_names: Optional[Sequence[str]] = None,
                      band: Tuple[float, float] = (10.0, 60.0), k: float = 3.5, win_s: float = 5.0,
                      min_dur_ms: float = 20.0, max_dur_ms: float = 200.0) -> List[Event]:
    """Log-envelope z-score detector (in the spirit of Janca et al., 2015).

    Band-pass -> Hilbert envelope -> log -> running mean / SD over ``win_s`` -> z-score;
    contiguous runs with z > ``k`` and duration within [min, max] become events with
    score = max z.
    """
    X = np.asarray(X, float)
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(X.shape[0])]
    xf = bandpass(X, fs, *band)
    env = np.log(np.abs(sps.hilbert(xf, axis=-1)) + 1e-9)
    z = blockwise_robust_z(env, fs, win_s)
    return events_from_mask(z > k, fs, names, scores=z, min_dur_s=min_dur_ms / 1000.0, max_dur_s=max_dur_ms / 1000.0)


def blockwise_robust_z(env: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Robust z-score of a (n_channels, n_samples) array using median / MAD in ``win_s`` blocks.

    Block statistics are linearly interpolated to every sample, which keeps the cost
    linear in record length (a running median would not) while resisting the heavy
    lower tail of a log-envelope.
    """
    env = np.asarray(env, float)
    n_ch, n = env.shape
    b = max(8, int(win_s * fs))
    nb = int(np.ceil(n / b))
    pad = nb * b - n
    e = np.pad(env, ((0, 0), (0, pad)), mode="edge").reshape(n_ch, nb, b)
    med = np.median(e, axis=-1)
    mad = np.median(np.abs(e - med[..., None]), axis=-1) * 1.4826 + 1e-9
    centers = (np.arange(nb) + 0.5) * b
    t = np.arange(n)
    if nb == 1:
        return (env - med[:, :1]) / mad[:, :1]
    med_i = np.stack([np.interp(t, centers, m) for m in med])
    mad_i = np.stack([np.interp(t, centers, m) for m in mad])
    return (env - med_i) / mad_i


def synthetic_spike_template(fs: float, spike_ms: float = 40.0, slow_ms: float = 200.0, slow_amp: float = -0.4) -> np.ndarray:
    """Spike (Gaussian, FWHM ~ spike_ms) followed by an opposite-polarity slow wave."""
    t = np.arange(-0.1, 0.4, 1.0 / fs)
    sig = spike_ms / 1000.0 / 2.355
    spike = np.exp(-0.5 * (t / sig) ** 2)
    slow = slow_amp * np.exp(-0.5 * ((t - 0.15) / (slow_ms / 1000.0 / 2.355)) ** 2)
    tpl = spike + slow
    return (tpl - tpl.mean()) / (np.linalg.norm(tpl) + 1e-12)


def template_detector(X: np.ndarray, fs: float, channel_names: Optional[Sequence[str]] = None,
                      template: Optional[np.ndarray] = None, thr: float = 4.0, refractory_ms: float = 80.0) -> List[Event]:
    """Normalized matched filter: z-scored cross-correlation with a spike + slow-wave template."""
    X = np.asarray(X, float)
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(X.shape[0])]
    tpl = synthetic_spike_template(fs) if template is None else np.asarray(template, float)
    tpl = (tpl - tpl.mean()) / (np.linalg.norm(tpl) + 1e-12)
    L = tpl.size
    peak_offset = int(np.argmax(tpl))
    out: List[Event] = []
    for c in range(X.shape[0]):
        x = X[c]
        xc = np.correlate(x, tpl, mode="valid")
        # local energy normalisation
        e = np.sqrt(np.convolve(x ** 2, np.ones(L), mode="valid") + 1e-12)
        ncc = xc / e
        z = (ncc - np.median(ncc)) / (1.4826 * np.median(np.abs(ncc - np.median(ncc))) + 1e-12)
        pk, props = sps.find_peaks(z, height=thr, distance=max(1, int(refractory_ms / 1000.0 * fs)))
        for p, h in zip(pk, props["peak_heights"]):
            out.append(Event(names[c], (p + peak_offset) / fs, L / fs, float(h)))
    return out


def candidate_peaks(X: np.ndarray, fs: float, band: Tuple[float, float] = (5.0, 50.0), min_prom_mad: float = 3.0,
                    refractory_ms: float = 80.0) -> List[Tuple[int, int]]:
    """Candidate (channel, sample) extrema on the band-passed signal with prominence > ``min_prom_mad`` MADs."""
    xf = bandpass(X, fs, *band)
    cands: List[Tuple[int, int]] = []
    dist = max(1, int(refractory_ms / 1000.0 * fs))
    for c in range(xf.shape[0]):
        x = xf[c]
        mad = np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9
        for sign in (1.0, -1.0):
            pk, _ = sps.find_peaks(sign * x, prominence=min_prom_mad * mad, distance=dist)
            cands += [(c, int(p)) for p in pk]
    return sorted(set(cands))


class MorphologyClassifierDetector:
    """Candidate extrema -> IFCN-inspired morphology features -> logistic regression score."""

    def __init__(self, C: float = 1.0, min_prom_mad: float = 3.0, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.min_prom_mad = min_prom_mad
        self.pipe = Pipeline([("scale", StandardScaler()),
                              ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))])
        self.feature_names = list(FEATURE_NAMES)

    def _candidates_and_features(self, X: np.ndarray, fs: float):
        cands = candidate_peaks(X, fs, min_prom_mad=self.min_prom_mad)
        F = features_matrix(X, fs, cands) if cands else np.zeros((0, len(FEATURE_NAMES)))
        return cands, F

    def fit(self, records: Sequence[Tuple[np.ndarray, float, Sequence[Event], Sequence[str]]], tol_s: float = 0.1) -> "MorphologyClassifierDetector":
        """``records``: iterable of (X, fs, reference events, channel_names)."""
        Fs, ys = [], []
        for X, fs, ref, names in records:
            cands, F = self._candidates_and_features(X, fs)
            if not cands:
                continue
            names = list(names)
            ref_by_ch = {}
            for r in ref:
                ref_by_ch.setdefault(r.channel, []).append(r.t_s)
            y = np.zeros(len(cands), int)
            for i, (c, idx) in enumerate(cands):
                ts = np.asarray(ref_by_ch.get(names[c], []), float)
                if ts.size and np.min(np.abs(ts - idx / fs)) <= tol_s:
                    y[i] = 1
            Fs.append(F)
            ys.append(y)
        F = np.vstack(Fs)
        y = np.concatenate(ys)
        if len(np.unique(y)) < 2:
            raise ValueError("training candidates contain a single class; lower min_prom_mad or check labels")
        self.pipe.fit(np.nan_to_num(F), y)
        return self

    def detect(self, X: np.ndarray, fs: float, channel_names: Optional[Sequence[str]] = None, thr: float = 0.5) -> List[Event]:
        names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(np.asarray(X).shape[0])]
        cands, F = self._candidates_and_features(X, fs)
        if not cands:
            return []
        p = self.pipe.predict_proba(np.nan_to_num(F))[:, 1]
        return [Event(names[c], idx / fs, F[i, 0] / 1000.0, float(p[i])) for i, (c, idx) in enumerate(cands) if p[i] >= thr]

    def coefficients(self) -> dict:
        return dict(zip(self.feature_names, self.pipe.named_steps["lr"].coef_[0].tolist()))
