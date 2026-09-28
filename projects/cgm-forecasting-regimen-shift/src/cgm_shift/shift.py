"""Cross-regimen transfer diagnostics and controller-feedback decomposition.

- ``marginal_shift``: variance ratio, lag-1 autocorrelation difference and a
  windowed MMD between two regimens' glucose distributions.
- ``controller_feedback_r2``: fraction of AID glucose *change* explained by
  recent CGM + insulin - the deterministic, loop-induced component that makes
  AID glucose look more predictable independent of any learned transfer.
- ``variance_matched_surrogate``: rescale a source stream to a target's variance
  as a null for "the accuracy gain is just lower variance."
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def _autocorr1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else np.nan


def _rbf_mmd2(A: np.ndarray, B: np.ndarray, gamma: float | None = None) -> float:
    A, B = np.asarray(A, float), np.asarray(B, float)
    Z = np.vstack([A, B])
    if gamma is None:
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def k(X, Y):
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


def marginal_shift(glucose_a: np.ndarray, glucose_b: np.ndarray, win: int = 12) -> dict:
    """Summarise the marginal distribution shift between two regimens' CGM."""
    a, b = np.asarray(glucose_a, float), np.asarray(glucose_b, float)
    res = {
        "var_ratio": float(np.var(b) / (np.var(a) + 1e-9)),
        "autocorr_diff": _autocorr1(b) - _autocorr1(a),
    }
    # windowed MMD on standardized windows
    def windows(x):
        return np.stack([x[i:i + win] for i in range(0, len(x) - win, win)]) if len(x) > win else x[None, :]
    Wa, Wb = windows(a), windows(b)
    m = min(len(Wa), len(Wb), 200)
    if m >= 5:
        res["mmd2"] = _rbf_mmd2(Wa[:m], Wb[:m])
    else:
        res["mmd2"] = float("nan")
    return res


def controller_feedback_r2(glucose: np.ndarray, insulin: np.ndarray, history: int = 6) -> float:
    """R^2 of glucose *change* over the next step predicted from recent CGM+insulin.

    High R^2 under AID indicates the loop's dosing makes the next glucose change
    partly deterministic given the observed state - 'predictability' that is
    controller-induced rather than learned transfer.
    """
    g = np.asarray(glucose, float)
    ins = np.asarray(insulin, float)
    X, y = [], []
    for t in range(history, len(g) - 1):
        feats = list(g[t - history:t]) + list(ins[t - history:t])
        X.append(feats)
        y.append(g[t + 1] - g[t])
    if len(y) < 20:
        return float("nan")
    X, y = np.asarray(X), np.asarray(y)
    reg = LinearRegression().fit(X, y)
    return float(reg.score(X, y))


def variance_matched_surrogate(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale ``source`` to have the mean/variance of ``target`` (a shift null).

    If cross-regimen accuracy gains vanish after this rescaling, they were driven
    by variance, not genuine transfer.
    """
    s, t = np.asarray(source, float), np.asarray(target, float)
    s_std = s.std() + 1e-9
    return (s - s.mean()) / s_std * t.std() + t.mean()
